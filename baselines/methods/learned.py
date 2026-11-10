"""DeepRAG checkpoint inference and ReasoningSearch NL-DAG execution."""

from __future__ import annotations

import re

from baselines.prompts import deeprag as deep_prompt
from baselines.prompts import r_search as dag_prompt


class LearnedProtocolError(ValueError):
    """A model response cannot execute under the method's control protocol."""


def _until(text: str, markers: tuple[str, ...]) -> str:
    positions = [text.find(marker) for marker in markers if marker in text]
    return text[: min(positions)] if positions else text


def _deep_answer(text: str) -> str:
    text = _until(text, deep_prompt.EOS + ("Follow up:", "So the final answer"))
    for tag in ("answer short", "answer long"):
        match = re.search(r"<" + tag + r">(.*?)(?:</" + tag + r">|$)", text, re.S)
        if match and match.group(1).strip():
            return match.group(1).strip()
    text = re.sub(r"</?answer (?:short|long)>", "", text).strip()
    if not text:
        raise LearnedProtocolError("DeepRAG produced an empty final answer")
    return text


async def deeprag(ctx) -> str:
    """Run the released SRAGSFTV2 policy, including its learned retrieve decision."""
    config = ctx.config
    if config.get("inference_mode", "released_checkpoint") != "released_checkpoint":
        raise ValueError("DeepRAG requires its released trained checkpoint")
    history: list[dict] = []
    dataset = str(config.get("dataset", getattr(ctx, "dataset", ""))).lower()
    dataset = {"hotpot": "hotpotqa"}.get(dataset, dataset)
    max_depth = int(config.get("max_depth_by_dataset", {"hotpotqa": 6}).get(
        dataset, config.get("max_depth", 10)
    ))
    if max_depth < 1:
        raise ValueError("DeepRAG max_depth must be positive")
    retries = int(config.get("empty_follow_up_attempts", 10))
    if retries < 1:
        raise ValueError("DeepRAG empty_follow_up_attempts must be positive")
    max_tokens = int(config.get("max_tokens", 4096))
    ctx.event("learned_variant", method="deeprag", mode="released_checkpoint",
              checkpoint=config.get("checkpoint", "xinyan233333/DeepRAG-8b"),
              revision=config.get("checkpoint_revision"))

    async def generate(assistant: str, stop: tuple[str, ...], *, extra_newline=False):
        prompt = deep_prompt.completion_prompt(ctx.tokenizer, ctx.question, assistant)
        if extra_newline:
            prompt += "\n"
        context_limit = int(config.get("max_context_tokens", 8192))
        input_tokens = len(ctx.tokenizer.encode(prompt, add_special_tokens=False))
        available = min(max_tokens, context_limit - input_tokens)
        if available < 1:
            raise LearnedProtocolError("DeepRAG history exhausted the checkpoint context window")
        return await ctx.complete(
            "deeprag", prompt, temperature=float(config.get("temperature", 0.0)),
            max_tokens=available, stop=list(stop), include_stop=True,
        )

    for depth in range(max_depth):
        history_text = deep_prompt.format_history(history)
        follow_up = ""
        for _ in range(retries):
            follow_up = await generate(
                history_text if history else "Follow up:", deep_prompt.FOLLOW_UP_STOP,
                extra_newline=bool(history),
            )
            if follow_up.strip():
                break
        if not follow_up.strip():
            raise LearnedProtocolError("DeepRAG exhausted empty follow-up retries")
        # Inspect control markers before cleaning them out of the continuation.
        if "So the final" in follow_up:
            ctx.stop = "answer_marker"
            ctx.event("deeprag_terminate", reason="model", depth=depth)
            break
        follow_up = _until(follow_up, deep_prompt.EOS).strip().removeprefix("Follow up:").strip()
        retrieval_match = re.search(r"(?:^|\s)Let(?:'s[^\n]*)?\s*$", follow_up)
        needs_retrieval = retrieval_match is not None
        if needs_retrieval:
            follow_up = follow_up[:retrieval_match.start()].strip()
        else:
            follow_up = _until(follow_up, ("Intermediate answer:",)).strip()
        if not follow_up:
            raise LearnedProtocolError("DeepRAG returned a control marker without a subquestion")
        step = {"follow_up": follow_up}
        prefix = history_text + "Follow up: " + follow_up + "\n"
        passage_ids = []
        if needs_retrieval:
            passages = ctx.retrieve(follow_up, k=int(config.get("top_k", 3)),
                                    mode=config.get("retriever", "bm25"))
            # Match the released numbered-context format, retaining titles as evidence.
            docs = "\n".join(
                f"[{i}] " + (str(p.get("title", "")) + " " + str(p["text"])).strip()
                for i, p in enumerate(passages, 1)
            )
            step["docs"] = docs
            passage_ids = [p["id"] for p in passages]
            prefix += "Let's search the question in Wikipedia.\nContext:\n" + docs + "\n"
        answer = await generate(prefix + "Intermediate answer: ", deep_prompt.INTERMEDIATE_STOP)
        answer = _until(answer, deep_prompt.INTERMEDIATE_STOP).strip()
        if not answer:
            raise LearnedProtocolError("DeepRAG produced an empty intermediate answer")
        step["answer"] = answer
        history.append(step)
        ctx.steps += 1
        ctx.event("deeprag_step", depth=depth + 1, query=follow_up,
                  decision="retrieve" if needs_retrieval else "parametric",
                  passage_ids=passage_ids, answer=answer)
    else:
        ctx.stop = "max_depth"
        ctx.event("deeprag_terminate", reason="max_depth", depth=max_depth)
    answer = await generate(
        deep_prompt.format_history(history) + "\nSo the final answer is: <answer long>",
        deep_prompt.FINAL_STOP,
    )
    return _deep_answer(answer if answer.startswith("<answer long>") else "<answer long>" + answer)


_SEARCH = re.compile(r"<search>(.*?)</search>", re.I | re.S)
_THINK = re.compile(r"<think>.+?</think>", re.I | re.S)
_NODE = re.compile(r"([A-Z])\s*:\s*(.+?)\s*[\[(（]\s*(General|News|Academy)\s*[\])）]\s*;?", re.I)
_EDGE = re.compile(r"([A-Z])\s*(?:->|→)\s*([A-Z])", re.I)


def parse_nl_dag(text: str, max_nodes: int = 6, max_query_words: int = 10):
    """Parse the paper's NL-DAG and reject incomplete, ambiguous, or cyclic plans."""
    match = _SEARCH.search(text)
    think = _THINK.search(text)
    if not match or not think or think.end() > match.start():
        raise LearnedProtocolError("Expected <think> followed by a complete <search> block")
    block = match.group(1).replace("：", ":").replace("；", ";")
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str]] = []
    section = None
    saw_edges = False
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "Nodes:":
            if section is not None:
                raise LearnedProtocolError("Repeated or misplaced Nodes section")
            section = "nodes"
            continue
        if line.startswith("Edges:"):
            if section != "nodes":
                raise LearnedProtocolError("Edges must follow Nodes")
            section, saw_edges = "edges", True
            line = line[len("Edges:"):].strip()
            if not line:
                continue
        if section == "nodes":
            node = _NODE.fullmatch(line)
            if not node:
                raise LearnedProtocolError("Invalid node or unsupported tool: " + line)
            label, query, tool = node.groups()
            label, query = label.upper(), query.strip()
            if label in nodes:
                raise LearnedProtocolError("Duplicate node: " + label)
            if max_query_words and len(query.split()) > max_query_words:
                raise LearnedProtocolError("Node " + label + " exceeds the query word limit")
            nodes[label] = {"query": query, "tool": tool.lower()}
        elif section == "edges":
            for part in line.split(";"):
                if not part.strip():
                    continue
                edge = _EDGE.fullmatch(part.strip())
                if not edge:
                    raise LearnedProtocolError("Invalid edge: " + part)
                pair = tuple(value.upper() for value in edge.groups())
                if pair not in edges:
                    edges.append(pair)
        else:
            raise LearnedProtocolError("Expected Nodes: before node definitions")
    if not saw_edges or not 1 <= len(nodes) <= max_nodes:
        raise LearnedProtocolError(f"Expected Edges: and between 1 and {max_nodes} nodes")
    parents = {label: set() for label in nodes}
    for source, target in edges:
        if source not in nodes or target not in nodes or source == target:
            raise LearnedProtocolError("Undefined node or self-loop in edge " + source + " -> " + target)
        parents[target].add(source)
    # Stable topological generations preserve source ordering within a frontier.
    remaining = set(nodes)
    levels = []
    while remaining:
        ready = [label for label in nodes if label in remaining and not (parents[label] & remaining)]
        if not ready:
            raise LearnedProtocolError("Search plan contains a cycle")
        levels.append(ready)
        remaining.difference_update(ready)
    return nodes, edges, levels, text[:match.end()]


def _rsearch_template(ctx, messages: list[dict]) -> str:
    return ctx.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


async def r_search(ctx) -> str:
    """Execute Shen/Shi ReasoningSearch; no relation to the other R-Search projects."""
    config = ctx.config
    mode = config.get("inference_mode", "llm_dag_adaptation")
    if mode not in {"llm_dag_adaptation", "trained_checkpoint"}:
        raise ValueError("Unknown R-Search inference_mode: " + str(mode))
    trained = mode == "trained_checkpoint"
    if trained and not config.get("trained_checkpoint"):
        raise ValueError("trained_checkpoint mode requires an explicit trained_checkpoint identity")
    ctx.event("learned_variant", method="r_search", mode=mode,
              checkpoint=config.get("trained_checkpoint") if trained else config.get("model"),
              source="wentao0429/ReasoningSearch", search_environment="candidate_passages")
    max_nodes = int(config.get("max_nodes", 6))
    max_query_words = int(config.get("max_query_words", 10))
    temperature = float(config.get("temperature", 0.6))
    answer_temperature = float(config.get("answer_temperature", 0.0))
    plan_tokens = int(config.get("plan_max_tokens", 1000))
    answer_tokens = int(config.get("answer_max_tokens", 2000))
    if max_nodes < 1 or max_query_words < 1:
        raise ValueError("R-Search node and query limits must be positive")
    messages = [
        {"role": "system", "content": dag_prompt.SOURCE_SYSTEM_PROMPT},
        {"role": "user", "content": ctx.question},
    ]
    if trained:
        raw = await ctx.complete(
            "r_search", _rsearch_template(ctx, messages + [{"role": "assistant", "content": "<think>\n"}]),
            temperature=temperature, max_tokens=plan_tokens, stop=["</search>"], include_stop=True,
        )
        plan = raw if raw.lstrip().startswith("<think>") else "<think>\n" + raw
    else:
        system = dag_prompt.PLAN_SYSTEM.format(max_nodes=max_nodes, max_query_words=max_query_words)
        plan = await ctx.ask("r_search", system, ctx.question,
                             temperature=temperature, max_tokens=plan_tokens)
    attempts = 0 if trained else int(config.get("plan_repair_attempts", 0))
    if attempts < 0:
        raise ValueError("R-Search plan_repair_attempts cannot be negative")
    for attempt in range(attempts + 1):
        try:
            nodes, edges, levels, plan = parse_nl_dag(plan, max_nodes, max_query_words)
            break
        except LearnedProtocolError as error:
            ctx.event("r_search_invalid_plan", attempt=attempt, error=str(error))
            if attempt == attempts:
                raise
            plan = await ctx.ask(
                "r_search", system,
                dag_prompt.REPAIR_USER.format(error=str(error), question=ctx.question, plan=plan),
                temperature=temperature, max_tokens=plan_tokens,
            )
    ctx.event("r_search_plan", nodes=nodes, edges=edges, levels=levels)
    result_parts = []
    for level in levels:
        # Shared candidate retrieval is synchronous. Independent queries have the same
        # frontier in the trace; execution order respects every DAG dependency.
        for label in level:
            node = nodes[label]
            passages = ctx.retrieve(node["query"], k=int(config.get("top_k", 3)),
                                    mode=config.get("retriever", "dense"))
            context = ctx.context(passages) if passages else "No passages were retrieved."
            result_parts.append(f"Node {label} ({node['tool']}) search results:\n{context}\n")
            ctx.steps += 1
            ctx.event("r_search_node", label=label, query=node["query"],
                      tool=node["tool"], passage_ids=[p["id"] for p in passages])
    observed = plan.rstrip() + "\n<result>\n" + "\n".join(result_parts) + "\n</result>"
    if trained:
        answer = await ctx.complete(
            "r_search", _rsearch_template(ctx, messages + [
                {"role": "assistant", "content": observed},
                {"role": "assistant", "content": "<answer>\n"},
            ]), temperature=answer_temperature, max_tokens=answer_tokens,
            stop=["</answer>"], include_stop=True,
        )
        if not answer.lstrip().startswith("<answer>"):
            answer = "<answer>" + answer
    else:
        answer = await ctx.ask(
            "r_search", dag_prompt.ANSWER_SYSTEM,
            "Question: " + ctx.question + "\n\n" + observed,
            temperature=answer_temperature, max_tokens=answer_tokens,
        )
    match = re.search(r"<answer>(.*?)</answer>", answer, re.I | re.S)
    if not match or not match.group(1).strip():
        raise LearnedProtocolError("R-Search did not produce a complete, nonempty <answer> block")
    ctx.stop = "answer"
    return match.group(1).strip()


METHODS = {"deeprag": deeprag, "r_search": r_search}
