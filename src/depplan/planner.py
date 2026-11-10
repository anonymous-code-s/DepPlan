from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agents import PlanningAgent
from .llm import CallUsage, LLMClient, sum_usage
from .plan import RetrievalPlan, extract_json_object, parse_plan
from .prompts import (
    JUDGE_SYSTEM,
    PROPOSER_SYSTEM,
    SELF_REFINER_SYSTEM,
    VERIFIER_SYSTEM,
    planning_context,
)


@dataclass
class PlanningResult:
    plan: RetrievalPlan
    usage: dict[str, Any]
    raw_outputs: list[dict[str, str]]


def _verifier_payload(text: str) -> tuple[RetrievalPlan, list[str], str]:
    payload = extract_json_object(text)
    plan = parse_plan(payload.get("plan", payload))
    issues = [str(item) for item in payload.get("issues", [])]
    verdict = str(payload.get("verdict", "revise")).lower()
    return plan, issues, verdict


class MultiAgentPlanner:
    """Proposer-Verifier rounds followed by conditional candidate selection."""

    def __init__(self, client: LLMClient, *, temperature: float = 0.7, max_tokens: int = 1024,
                 compact_prompts: bool = False, use_verifier: bool = True,
                 use_judge: bool = True, rounds: int = 2):
        if type(rounds) is not int or rounds not in (1, 2, 3):
            raise ValueError("rounds must be 1, 2, or 3")
        self.rounds = rounds
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.compact_prompts = compact_prompts
        if use_judge and not use_verifier:
            raise ValueError('The Judge requires Proposer and Verifier candidates')
        self.use_verifier = use_verifier
        self.use_judge = use_judge

    async def plan(
        self,
        question: str,
        *,
        units: list[str] | None,
        strategy: str | None,
    ) -> PlanningResult:
        context = planning_context(question, units, strategy, indexed_units=self.compact_prompts)
        from .prompt_compaction import compact_json, previous_candidates
        render = compact_json if self.compact_prompts else lambda value: json.dumps(value, ensure_ascii=False)
        calls: list[CallUsage] = []
        raw: list[dict[str, str]] = []
        proposer = PlanningAgent('proposer', self.client, system=PROPOSER_SYSTEM,
                                 temperature=self.temperature, max_tokens=self.max_tokens)

        first = await proposer.complete(
            role="proposer_round_1",
            user=context,
        )
        calls.extend(first.attempt_usage)
        raw.append({"role": first.usage.role, "text": first.text})
        proposal_1 = extract_json_object(first.text)
        proposal_1 = proposal_1.get('plan', proposal_1)

        if not self.use_verifier:
            plan = parse_plan(proposal_1, require_placeholder_edges=True)
            return PlanningResult(plan, sum_usage(calls), raw)

        verifier = PlanningAgent('verifier', self.client, system=VERIFIER_SYSTEM,
                                 temperature=self.temperature, max_tokens=self.max_tokens)
        verify_1 = await verifier.complete(
            role="verifier_round_1",
            user=f"{context}\n\nCandidate plan:\n{render(proposal_1)}",
            source='proposer',
        )
        calls.extend(verify_1.attempt_usage)
        raw.append({"role": verify_1.usage.role, "text": verify_1.text})
        first_verdict = extract_json_object(verify_1.text)
        verified_1 = first_verdict.get('plan',first_verdict)
        issues_1 = first_verdict.get('issues',[])

        proposal_2, verified_2, issues_2 = proposal_1, verified_1, issues_1
        for round_number in range(2, self.rounds + 1):
            second = await proposer.complete(
                role=f"proposer_round_{round_number}",
                user=(f"{context}\n\n{previous_candidates(proposal_2, verified_2, issues_2, round_number=round_number-1)}"
                      if self.compact_prompts else (
                    f"{context}\n\nRound-{round_number-1} proposal:\n"
                    f"{json.dumps(proposal_2, ensure_ascii=False)}\n\n"
                    f"Verifier candidate:\n{json.dumps(verified_2, ensure_ascii=False)}\n\n"
                    f"Verifier issues:\n{json.dumps(issues_2, ensure_ascii=False)}\n"
                    "Return the minimally revised plan."
                )),
                source='verifier',
            )
            calls.extend(second.attempt_usage)
            raw.append({"role": second.usage.role, "text": second.text})
            proposal_2 = extract_json_object(second.text)
            proposal_2 = proposal_2.get('plan', proposal_2)
            verify_2 = await verifier.complete(
                role=f"verifier_round_{round_number}",
                user=f"{context}\n\nCandidate plan:\n{render(proposal_2)}",
                source='proposer',
            )
            calls.extend(verify_2.attempt_usage)
            raw.append({"role": verify_2.usage.role, "text": verify_2.text})
            second_verdict = extract_json_object(verify_2.text)
            verified_2 = second_verdict.get('plan', second_verdict)
            issues_2 = second_verdict.get('issues', [])
        if not self.use_judge:
            plan = parse_plan(verified_2, require_placeholder_edges=True)
            return PlanningResult(plan, sum_usage(calls), raw)
        valid_candidates = []
        validation = []
        for name,candidate in [('Proposer',proposal_2),('Verifier',verified_2)]:
            try:
                parsed = parse_plan(candidate,require_placeholder_edges=True)
                valid_candidates.append(parsed)
                validation.append(name+': valid')
            except (ValueError,KeyError,TypeError) as exc:
                validation.append(name+': invalid: '+str(exc))
        if not valid_candidates:
            raise ValueError('Neither final planning candidate is executable: '+'; '.join(validation))
        if len(valid_candidates)==2 and valid_candidates[0].to_dict()==valid_candidates[1].to_dict():
            return PlanningResult(valid_candidates[0],sum_usage(calls),raw)

        word = {1: 'one', 2: 'two', 3: 'three'}[self.rounds]
        judge_system = JUDGE_SYSTEM.replace('after two proposal-verification rounds.',
                                           f'after {word} proposal-verification rounds.')
        judge_agent = PlanningAgent('judge', self.client, system=judge_system,
                                    temperature=self.temperature, max_tokens=self.max_tokens)
        judge = await judge_agent.complete(
            role="judge",
            user=(
                f"{context}\n\nLatest Proposer plan:\n"
                f"{render(proposal_2)}\n\n"
                f"Latest Verifier plan:\n{render(verified_2)}\n\n"
                f"Remaining issues:\n{render(issues_2)}\n"
                f"Executable-candidate checks:\n{json.dumps(validation)}\nChoose a valid candidate unchanged."
            ),
        )
        calls.extend(judge.attempt_usage)
        raw.append({"role": judge.usage.role, "text": judge.text})
        final_plan = parse_plan(judge.text,require_placeholder_edges=True)
        if not any(final_plan.to_dict()==candidate.to_dict() for candidate in valid_candidates):
            raise ValueError('Judge must select one provided executable candidate unchanged')
        return PlanningResult(final_plan, sum_usage(calls), raw)


class EqualCallSelfRefiner:
    """One agent, five calls, matching the multi-agent planning-call budget."""

    def __init__(self, client: LLMClient, *, temperature: float = 0.7, max_tokens: int = 1024):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def plan(
        self,
        question: str,
        *,
        units: list[str] | None,
        strategy: str | None,
    ) -> PlanningResult:
        context = planning_context(question, units, strategy)
        calls: list[CallUsage] = []
        raw: list[dict[str, str]] = []
        current: RetrievalPlan | None = None
        agent = PlanningAgent('single_agent', self.client, system=SELF_REFINER_SYSTEM,
                              temperature=self.temperature, max_tokens=self.max_tokens)
        for call_index in range(1, 6):
            if current is None:
                instruction = "Generate the initial executable retrieval plan."
            else:
                instruction = (
                    "Critique the current plan internally, then return a corrected complete plan. "
                    "Do not output the critique.\nCurrent plan:\n"
                    f"{json.dumps(current.to_dict(), ensure_ascii=False)}"
                )
            completion = await agent.complete(
                role=f"single_agent_call_{call_index}",
                user=f"{context}\n\n{instruction}",
            )
            calls.extend(completion.attempt_usage)
            raw.append({"role": completion.usage.role, "text": completion.text})
            current = parse_plan(completion.text)
        assert current is not None
        return PlanningResult(current, sum_usage(calls), raw)

