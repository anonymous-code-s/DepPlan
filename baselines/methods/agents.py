"""Agent baselines under the shared model/candidate-corpus protocol.

See baselines/references/agent_parameters.md for source-specific settings and
implementation choices. Inference uses question text and candidate passages.
"""
from __future__ import annotations

import re

from ..prompts import coa as coa_prompts, per as per_prompts
from .agent_utils import answer_text, ask_json, count_step, instantiate, parse_plan, supported


def sentence_chunks(text, tokenizer, token_budget):
    """Algorithm 2: preserve source order, pack sentences, split oversized ones."""
    if token_budget < 1:
        raise ValueError("CoA's instruction and carry reservations leave no source budget")
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n(?=\[)", text):
        if not sentence.strip():
            continue
        combined = (current + " " + sentence).strip()
        if len(tokenizer.encode(combined, add_special_tokens=False)) <= token_budget:
            current = combined
            continue
        if current:
            chunks.append(current)
            current = ""
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        while len(tokens) > token_budget:
            chunks.append(tokenizer.decode(tokens[:token_budget], skip_special_tokens=True))
            tokens = tokens[token_budget:]
        current = tokenizer.decode(tokens, skip_special_tokens=True)
    if current:
        chunks.append(current)
    return chunks


async def coa(ctx):
    """Sequential workers cover the complete candidate text; manager reads last CU."""
    cfg = ctx.config
    window = int(cfg.get("worker_window_tokens", 8192))
    carry = int(cfg.get("communication_max_tokens", 1024))
    reserve = int(cfg.get("worker_output_tokens", 1024))
    wrapper = "Question: " + ctx.question + "\nCurrent source:\nPrevious summary:\n"
    overhead = len(ctx.tokenizer.encode(coa_prompts.prompt_worker + wrapper,
                                       add_special_tokens=False))
    # Carry/output/chat-template reservations are an explicit shared-API adaptation.
    budget = window - carry - reserve - overhead - int(cfg.get("chat_overhead_tokens", 64))
    chunks = sentence_chunks(ctx.context(ctx.passages), ctx.tokenizer, budget)
    ctx.event("coa_chunks", count=len(chunks), source_budget_tokens=budget,
              passage_ids=[p["id"] for p in ctx.passages])
    communication = ""
    for index, chunk in enumerate(chunks, 1):
        count_step(ctx, "coa_worker")
        communication = await ctx.ask(
            "coa_worker", coa_prompts.prompt_worker,
            f"Question: {ctx.question}\nCurrent source:\n{chunk}\n"
            f"Previous summary:\n{communication}", max_tokens=min(carry, reserve))
        ctx.event("coa_communication", worker=index, communication=communication)
    answer = await ctx.ask(
        "coa_manager", coa_prompts.prompt_manager,
        f"Question: {ctx.question}\nFinal summary:\n{communication}",
        temperature=cfg.get("answer_temperature", 0),
        max_tokens=cfg.get("answer_max_tokens", 1024))
    ctx.stop = "all_chunks_processed"
    return answer.strip()


async def per(ctx):
    """PER-QA-RAG: typed plan DAG -> dependent retrieval -> context-graph reasoner."""
    cfg = ctx.config
    classification = await ask_json(ctx, "per_classifier", per_prompts.perqa_type_prompt,
                                    question=ctx.question)
    question_type = str(classification.get("type", classification.get("Response", ""))).lower()
    if question_type not in {"bridge", "comparison"}:
        raise ValueError(f"Invalid PER question type: {question_type}")
    prompt = (per_prompts.perqa_bridge_prompt if question_type == "bridge"
              else per_prompts.perqa_comparison_prompt)
    planned = await ask_json(ctx, "per_planner", prompt, question=ctx.question)
    nodes = parse_plan(planned, int(cfg.get("max_plan_nodes", 16)))
    edges = [[parent, node["id"]] for node in nodes for parent in node["dependencies"]]
    ctx.event("per_plan", question_type=question_type, nodes=nodes, edges=edges)
    results = {}
    for node in nodes:
        count_step(ctx, "per_plan_node")
        if any(not results[parent]["supported"] for parent in node["dependencies"]):
            results[node["id"]] = {"answer": "Unknown", "supported": False,
                                    "query": node["template"], "blocked": True}
            ctx.event("per_blocked_node", node=node["id"], dependencies=node["dependencies"])
            continue
        query = instantiate(node, results)
        retrieval_query = query
        if cfg.get("use_hyde", True):
            hypothetical = await ctx.ask("per_hyde", per_prompts.perqa_hyde_prompt, query)
            # Official module.py get_hyde_query repeats the question five times.
            retrieval_query = " ".join([query] * int(cfg.get("hyde_query_repetitions", 5)))
            retrieval_query += " " + hypothetical.strip().splitlines()[0] if hypothetical.strip() else ""
        passages = ctx.retrieve(retrieval_query, k=int(cfg.get("top_k", 5)))
        result = await ask_json(ctx, "per_executor", per_prompts.perqa_executor_prompt,
                                question=query, context=ctx.context(passages))
        results[node["id"]] = {"query": query, "answer": answer_text(result),
                                "supported": per_supported(result),
                                "passage_ids": [p["id"] for p in passages]}
        ctx.event("per_context_node", node=node["id"], dependencies=node["dependencies"],
                  retrieval_query=retrieval_query, **results[node["id"]])
    # Official PER only aggregates complete candidate graphs. Failed branches
    # cannot be bypassed by asking the reasoner to invent missing intermediates.
    if not all(value["supported"] for value in results.values()):
        ctx.stop = "incomplete_context_graph"
        return "Unknown"
    linearized = "\n".join(f"{node['id']}: {results[node['id']]['query']} >> "
                           f"{results[node['id']]['answer']}" for node in nodes)
    reasoner = (per_prompts.perqa_bridge_reasoner_prompt if question_type == "bridge"
                else per_prompts.perqa_comparison_reasoner_prompt)
    value = ctx.json(await ctx.ask(
        "per_reasoner", reasoner,
        f"Question type: {question_type}\nQuestion: {ctx.question}\nContext:\n{linearized}",
        temperature=cfg.get("answer_temperature", 0),
        max_tokens=cfg.get("answer_max_tokens", 1024)))
    ctx.event("per_context_graph", nodes=results, edges=edges, complete=True)
    ctx.stop = "complete_context_graph"
    return answer_text(value)


def per_supported(result):
    """Official PERQARAGExecutor answer filtering, with optional JSON flag."""
    if not supported(result):
        return False
    answer = answer_text(result).casefold()
    if any(marker in answer for marker in ("[]", "[ ]", "not ", "there is no")):
        return False
    unsupported_words = {
        "unknown", "support", "describe", "unclear", "insufficient", "unspecified",
        "mention", "evidence", "undefined", "none", "available", "information", "found",
    }
    if any(word in unsupported_words for word in re.split(r"[\s,.?!]+", answer)):
        return False
    return not ("no " in answer and " no " not in answer)


from .gear import gear  # noqa: E402
from .belle import belle  # noqa: E402

METHODS = {"coa": coa, "gear": gear, "belle": belle, "per": per}
