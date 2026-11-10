"""Shared serialization and dependency validation for agent baselines."""
from __future__ import annotations

import json
import re


def payload(**fields):
    return json.dumps(fields, ensure_ascii=False)


def count_step(ctx, stage):
    limit = int(ctx.config.get("max_steps", 256))
    if ctx.steps >= limit:
        raise RuntimeError(f"{stage}: execution exceeded max_steps={limit}")
    ctx.steps += 1
    ctx.event("agent_step", stage=stage, step=ctx.steps)


async def ask_json(ctx, role, system, **fields):
    value = ctx.json(await ctx.ask(role, system, payload(**fields)))
    if not isinstance(value, dict):
        raise ValueError(f"{role} must return a JSON object")
    return value


def answer_text(value):
    if isinstance(value, str):
        return value.strip()
    for key in ("answer", "final_answer", "Response"):
        if key in value:
            answer = value[key]
            if isinstance(answer, list):
                answer = answer[0] if answer else "Unknown"
            return str(answer).strip()
    raise ValueError("Answer object does not contain an answer field")


def supported(value):
    if value.get("supported") is False:
        return False
    return answer_text(value).strip().casefold() not in {
        "", "unknown", "no-answer", "not support", "[not support]", "none", "n/a"
    }


def parse_plan(value, max_nodes=16):
    """Validate tags and topologically order a PER-style plan without flattening.

    Node order in generated JSON need not already be topological. Unknown tags,
    self-dependencies, cycles, duplicate tag declarations, and empty plans fail
    explicitly instead of silently reverting to an unrelated answering method.
    """
    plan = value.get("plan", value.get("Response", value))
    if not isinstance(plan, dict) or not 0 < len(plan) <= max_nodes:
        raise ValueError(f"Plan must contain 1–{max_nodes} nodes")
    nodes = {}
    for node_id, item in plan.items():
        if not re.fullmatch(r"Q[1-9]\d*", node_id):
            raise ValueError(f"Invalid plan node: {node_id}")
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{node_id} must contain [query, answer_tag]")
        query, tag = item
        if not isinstance(query, str) or not query.strip() or tag != f"<A{node_id[1:]}>":
            raise ValueError(f"Invalid query or answer tag for {node_id}")
        dependencies = list(dict.fromkeys(f"Q{n}" for n in re.findall(r"<A(\d+)>", query)))
        nodes[node_id] = {"id": node_id, "template": query, "tag": tag,
                          "dependencies": dependencies}
    ordered, done = [], set()
    while len(ordered) < len(nodes):
        ready = [node for key, node in nodes.items() if key not in done
                 and set(node["dependencies"]) <= done]
        if not ready:
            raise ValueError("Plan contains a cycle or a reference to an absent node")
        for node in ready:
            ordered.append(node)
            done.add(node["id"])
    return ordered


def instantiate(node, results):
    query = node["template"]
    for dependency in node["dependencies"]:
        query = query.replace(f"<A{dependency[1:]}>", results[dependency]["answer"])
    return query


def merge_passages(*groups):
    unique = {}
    for group in groups:
        for passage in group:
            unique.setdefault(str(passage["id"]), passage)
    return list(unique.values())
