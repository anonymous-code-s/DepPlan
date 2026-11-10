from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


PLACEHOLDER = re.compile(r"\[#\s*(\d+)\s*\]")


@dataclass(frozen=True)
class PlanStep:
    id: int
    sub_question: str
    depends_on: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalPlan:
    steps: tuple[PlanStep, ...]
    execution_order: tuple[int, ...]
    final_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "id": step.id,
                    "sub_question": step.sub_question,
                    "depends_on": list(step.depends_on),
                }
                for step in self.steps
            ],
            "execution_order": list(self.execution_order),
            "final_step": self.final_step,
        }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output contains no valid JSON object")


def placeholder_dependencies(question: str) -> set[int]:
    return {int(value) for value in PLACEHOLDER.findall(question)}


def _canonical_step(raw: dict[str, Any], fallback_id: int) -> PlanStep:
    step_id = int(raw.get("id", fallback_id))
    question = str(raw.get("sub_question", raw.get("question", ""))).strip()
    if not question:
        raise ValueError(f"Step {step_id} has an empty sub-question")
    raw_deps = raw.get("depends_on", [])
    if raw_deps is None:
        raw_deps = []
    dependencies = {int(value) for value in raw_deps}
    dependencies.update(placeholder_dependencies(question))
    if step_id in dependencies:
        raise ValueError(f'Step {step_id} refers to its own answer')
    return PlanStep(step_id, question, tuple(sorted(dependencies)))


def topological_order(steps: Iterable[PlanStep]) -> tuple[int, ...]:
    step_list = list(steps)
    ids = {step.id for step in step_list}
    indegree = {step.id: 0 for step in step_list}
    children = {step.id: [] for step in step_list}
    for step in step_list:
        for parent in step.depends_on:
            if parent not in ids:
                raise ValueError(f"Step {step.id} depends on missing step {parent}")
            indegree[step.id] += 1
            children[parent].append(step.id)
    ready = [step_id for step_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(step_list):
        raise ValueError("Plan dependencies contain a cycle")
    return tuple(order)


def parse_plan(raw: str | dict[str, Any], *, require_placeholder_edges: bool = False) -> RetrievalPlan:
    payload = extract_json_object(raw) if isinstance(raw, str) else raw
    if "plan" in payload and isinstance(payload["plan"], dict):
        payload = payload["plan"]
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Plan JSON must contain a non-empty 'steps' array")
    if require_placeholder_edges:
        for step in raw_steps:
            declared = {int(v) for v in (step.get('depends_on') or [])}
            referenced = placeholder_dependencies(str(step.get('sub_question',step.get('question',''))))
            if declared != referenced:
                raise ValueError('depends_on must equal exactly the placeholder references in the sub-question')
    steps = tuple(_canonical_step(step, idx) for idx, step in enumerate(raw_steps, start=1))
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("Plan step ids must be unique")
    canonical_order = topological_order(steps)
    requested_order = payload.get("execution_order")
    if requested_order is not None:
        requested = tuple(int(value) for value in requested_order)
        if set(requested) != set(ids) or len(requested) != len(ids):
            raise ValueError("execution_order must contain every step id exactly once")
        positions = {value: index for index, value in enumerate(requested)}
        if any(positions[parent] >= positions[step.id] for step in steps for parent in step.depends_on):
            raise ValueError("execution_order violates producer-consumer dependencies")
        canonical_order = requested
    final_step = int(payload.get("final_step", canonical_order[-1]))
    if final_step not in set(ids):
        raise ValueError("final_step is not a valid step id")
    return RetrievalPlan(steps, canonical_order, final_step)


def resolve_placeholders(question: str, answers: dict[int, str], unresolved: str = "[UNRESOLVED]") -> str:
    def replace(match: re.Match[str]) -> str:
        return answers.get(int(match.group(1)), unresolved)

    return PLACEHOLDER.sub(replace, question)


def plan_edges(plan: RetrievalPlan) -> set[tuple[int, int]]:
    return {(parent, step.id) for step in plan.steps for parent in step.depends_on}


def plan_from_dict(raw: dict[str, Any], *, allow_invalid_order: bool = False) -> RetrievalPlan:
    if not allow_invalid_order:
        return parse_plan(raw)
    steps = tuple(_canonical_step(step, idx) for idx, step in enumerate(raw["steps"], start=1))
    order = tuple(int(value) for value in raw.get("execution_order", [s.id for s in steps]))
    return RetrievalPlan(steps, order, int(raw.get("final_step", order[-1])))
