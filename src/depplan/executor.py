from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm import CallUsage, LLMClient, sum_usage
from .plan import RetrievalPlan, extract_json_object, resolve_placeholders
from .prompts import ANSWER_SYSTEM, FINAL_ANSWER_SYSTEM, SEMANTIC_JUDGE_SYSTEM
from .retrieval import DenseRetriever


@dataclass
class ExecutionResult:
    prediction: str
    steps: list[dict[str, Any]]
    usage: dict[str, Any]
    raw_final: str


def _answer_from_text(text: str) -> str:
    payload = extract_json_object(text)
    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise ValueError("Answer model returned an empty answer")
    return answer


class PlanExecutor:
    def __init__(
        self,
        client: LLMClient,
        retriever: DenseRetriever,
        *,
        top_k: int = 5,
        max_tokens: int = 512,
        compact_prompts: bool = False,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.compact_prompts = compact_prompts
        self.last_trace = []

    async def execute(
        self,
        question: str,
        embedding_record: dict[str, Any],
        plan: RetrievalPlan,
    ) -> ExecutionResult:
        by_id = {step.id: step for step in plan.steps}
        answers: dict[int, str] = {}
        trace: list[dict[str, Any]] = []
        self.last_trace = trace
        calls: list[CallUsage] = []
        for step_id in plan.execution_order:
            step = by_id[step_id]
            resolved = resolve_placeholders(step.sub_question, answers)
            evidence = self.retriever.retrieve(
                embedding_record, resolved, top_k=self.top_k
            )
            completion = await self.client.complete(
                role=f"answer_step_{step_id}",
                system=ANSWER_SYSTEM,
                user=(
                    f"Sub-question:\n{resolved}\n\nEvidence:\n"
                    + "\n\n".join(item["text"] for item in evidence)
                ),
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            calls.extend(completion.attempt_usage)
            answer = _answer_from_text(completion.text)
            answers[step_id] = answer
            trace.append(
                {
                    "step_id": step_id,
                    "sub_question_template": step.sub_question,
                    "resolved_sub_question": resolved,
                    "depends_on": list(step.depends_on),
                    "retrieved_contexts": [
                        {"index": item["index"], "score": item["score"], "text": item['text']}
                        for item in evidence
                    ],
                    "answer": answer,
                    "raw_answer_output": completion.text,
                    "unresolved_dependency": "[UNRESOLVED]" in resolved,
                }
            )

        if self.compact_prompts:
            from .prompt_compaction import compact_json, final_answer_payload
            final_user = compact_json(final_answer_payload(question, trace))
        else:
            final_user = (f"Original question:\n{question}\n\nExecuted trace:\n"
                          f"{json.dumps(trace, ensure_ascii=False)}")
        final = await self.client.complete(
            role="final_answer",
            system=FINAL_ANSWER_SYSTEM,
            user=final_user,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        calls.extend(final.attempt_usage)
        prediction = _answer_from_text(final.text)
        return ExecutionResult(prediction, trace, sum_usage(calls), final.text)


async def semantic_accuracy(
    client: LLMClient, question: str, reference: str | list[str], prediction: str
) -> tuple[int, CallUsage]:
    completion = await client.complete(
        role="semantic_accuracy_judge",
        system=SEMANTIC_JUDGE_SYSTEM,
        user=json.dumps(dict(question=question,
                             references=[reference] if isinstance(reference, str) else reference,
                             prediction=prediction), ensure_ascii=False),
        temperature=0.0,
        max_tokens=64,
    )
    payload = extract_json_object(completion.text)
    correct = payload.get('correct')
    if not (type(correct) is bool or type(correct) is int and correct in (0, 1)):
        raise ValueError('Semantic evaluator must return a binary correct field')
    return int(correct), completion.usage
