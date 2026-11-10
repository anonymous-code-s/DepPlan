"""BELLE's bi-level debate and composable five-operator executor."""
from __future__ import annotations

from ..prompts import belle as prompts
from .agent_utils import (answer_text, ask_json, count_step, instantiate, merge_passages,
                          parse_plan, payload, supported)


async def debate(ctx):
    classification = await ask_json(ctx, "belle_classifier", prompts.belle_type_classifier_prompt,
                                    question=ctx.question)
    question_type = classification.get("type")
    if question_type not in {"Inference", "Comparison", "Temporal", "Null"}:
        raise ValueError(f"Invalid BELLE question type: {question_type}")
    history, fast_history, slow_history = [], [], []
    previous_fast, previous_slow = {}, {}
    max_rounds = int(ctx.config.get("debate_rounds", 3))
    if max_rounds < 1:
        raise ValueError("BELLE requires at least one debate round")
    for round_number in range(1, max_rounds + 1):
        common = {"question": ctx.question, "question_type": question_type,
                  "operator_set": prompts.OPERATORS, "round": round_number}
        affirmative = await ask_json(
            ctx, "belle_affirmative", prompts.belle_affirmative_prompt, **common,
            own_history=[item["affirmative"] for item in history],
            previous_fast=previous_fast, previous_slow=previous_slow)
        negative = await ask_json(
            ctx, "belle_negative", prompts.belle_negative_prompt, **common,
            affirmative=affirmative, own_history=[item["negative"] for item in history],
            previous_fast=previous_fast, previous_slow=previous_slow)
        fast = await ask_json(ctx, "belle_fast", prompts.belle_fast_prompt, **common,
                              affirmative=affirmative, negative=negative,
                              fast_history=fast_history)
        slow = await ask_json(ctx, "belle_slow", prompts.belle_slow_prompt, **common,
                              affirmative=affirmative, negative=negative, fast=fast,
                              debate_history=history, slow_history=slow_history)
        fast_history.append(fast)
        slow_history.append(slow)
        mode = "soft" if round_number == max_rounds else "hard"
        judge = await ask_json(ctx, "belle_judge", prompts.belle_judge_prompt, **common,
                               affirmative=affirmative, negative=negative, fast=fast,
                               slow=slow, slow_history=slow_history, mode=mode)
        record = {"round": round_number, "affirmative": affirmative, "negative": negative,
                  "fast": fast, "slow": slow, "judge": judge, "mode": mode}
        history.append(record)
        ctx.event("belle_debate", **record)
        previous_fast, previous_slow = fast, slow
        if judge.get("ready") is True or mode == "soft":
            operators = judge.get("operators")
            if not isinstance(operators, list) or not operators:
                raise ValueError("BELLE judge must return a nonempty ordered operator list")
            if any(operator not in prompts.OPERATORS for operator in operators):
                raise ValueError(f"Unknown BELLE operator: {operators}")
            if len(operators) > int(ctx.config.get("max_plan_operators", 8)):
                raise ValueError("BELLE operator plan exceeds configured execution bound")
            return operators
    raise RuntimeError("BELLE debate did not produce a plan")


async def solve_operator(ctx, operator, question, state):
    if operator == "ADAPTIVE_STEP":
        route = await ask_json(ctx, "belle_adaptive_router", prompts.belle_adaptive_prompt,
                               question=question, state=state)
        selected = {"A": "COT", "B": "SINGLE_STEP", "C": "ITERATIVE_STEP"}.get(
            route.get("complexity"))
        if selected is None:
            raise ValueError("BELLE adaptive complexity must be A, B, or C")
        ctx.event("belle_adaptive_route", question=question, selected=selected, decision=route)
        return await solve_operator(ctx, selected, question, state)
    if operator == "COT":
        count_step(ctx, "belle_cot")
        return await ask_json(ctx, "belle_cot", prompts.belle_cot_prompt,
                               question=question, state=state)
    k = int(ctx.config.get("top_k", 5))
    mode = ctx.config.get("retriever", "bm25")
    if operator == "SINGLE_STEP":
        count_step(ctx, "belle_single_step")
        passages = ctx.retrieve(question, k=k, mode=mode)
        value = await ask_json(ctx, "belle_single", prompts.belle_single_prompt,
                               question=question, context=ctx.context(passages), state=state)
        value["passage_ids"] = [p["id"] for p in passages]
        return value
    if operator != "ITERATIVE_STEP":
        raise ValueError(f"Cannot solve a question with operator {operator}")
    query, passages, thoughts = question, [], []
    for iteration in range(1, int(ctx.config.get("operator_max_iterations", 6)) + 1):
        count_step(ctx, "belle_iterative_step")
        retrieved = ctx.retrieve(query, k=k, mode=mode)
        passages = merge_passages(passages, retrieved)
        value = await ask_json(ctx, "belle_iterative", prompts.belle_iterative_prompt,
                               question=question, context=ctx.context(passages),
                               state=state, thoughts=thoughts)
        thoughts.append(str(value.get("thought", "")))
        ctx.event("belle_iterative_step", question=question, iteration=iteration,
                  query=query, passage_ids=[p["id"] for p in retrieved], decision=value)
        if value.get("done") is True:
            value["passage_ids"] = [p["id"] for p in passages]
            return value
        query = str(value.get("query", "")).strip()
        if not query:
            raise ValueError("BELLE iterative operator needs a nonempty next query")
    # At the explicit iteration cap the accumulated evidence still gets answered.
    value = await ask_json(ctx, "belle_iterative_answer", prompts.belle_single_prompt,
                           question=question, context=ctx.context(passages),
                           state=state, thoughts=thoughts)
    value["passage_ids"] = [p["id"] for p in passages]
    return value


async def solve_subquestions(ctx, nodes, solver, prior_state):
    results = {}
    for node in nodes:
        if any(not supported(results[parent]) for parent in node["dependencies"]):
            result = {"answer": "Unknown", "supported": False, "blocked": True}
            query = node["template"]
        else:
            query = instantiate(node, results)
            result = await solve_operator(ctx, solver, query,
                                          {"prior_state": prior_state, "subanswers": results})
        result["query"] = query
        results[node["id"]] = result
        ctx.event("belle_subquestion", node=node["id"], dependencies=node["dependencies"],
                  solver=solver, result=result)
    return results


async def belle(ctx):
    operators = await debate(ctx)
    ctx.event("belle_operator_plan", operators=operators)
    state, pending_nodes = [], None
    for index, operator in enumerate(operators):
        if operator == "SUB_STEP":
            # SUB_STEP prepares dependent questions for the next selected solver.
            # It is also executable alone, using single-step for each subquestion.
            plan = await ask_json(ctx, "belle_decompose", prompts.belle_decompose_prompt,
                                   question=ctx.question, state=state)
            pending_nodes = parse_plan(plan, int(ctx.config.get("max_subquestions", 12)))
            state.append({"operator": operator, "plan": pending_nodes})
            if index == len(operators) - 1:
                results = await solve_subquestions(ctx, pending_nodes, "SINGLE_STEP", state)
                state.append({"operator": operator, "subanswers": results})
                pending_nodes = None
        elif pending_nodes is not None:
            results = await solve_subquestions(ctx, pending_nodes, operator, state)
            state.append({"operator": operator, "subanswers": results})
            # Keep the decomposition for subsequent solver refinements. Each
            # refinement instantiates the current pass's dependency answers.
        else:
            result = await solve_operator(ctx, operator, ctx.question, state)
            state.append({"operator": operator, "result": result})
        ctx.event("belle_operator", index=index + 1, operator=operator, result=state[-1])
    value = ctx.json(await ctx.ask(
        "belle_final", prompts.belle_final_prompt, payload(question=ctx.question, state=state),
        temperature=ctx.config.get("answer_temperature", 0),
        max_tokens=ctx.config.get("answer_max_tokens", 1024)))
    ctx.stop = "operator_plan_complete"
    return answer_text(value)
