PLAN_SCHEMA = """
Return JSON only:
{
  "steps": [
    {"id": 1, "sub_question": "...", "depends_on": []},
    {"id": 2, "sub_question": "... [#1] ...", "depends_on": [1]}
  ],
  "execution_order": [1, 2],
  "final_step": 2
}
Every dependency must be represented both in depends_on and by a matching [#i]
placeholder in the consumer sub-question. IDs start at 1. The dependency graph
must be acyclic, and every producer precedes its consumers.
depends_on must equal EXACTLY the set of [#i] references in that sub-question.
Use the upstream ANSWER as the needed entity in the next retrieval query.
Known named entities are constants: do not add a step asking for their names.
For example, to find the founder of the publisher of novel X, retrieve its
publisher, then ask "Who founded [#1]?". Do not repeat the full original
multi-hop question as the downstream retrieval query.
""".strip()

PROPOSER_SYSTEM = f"""
You are the Proposer in a retrieval-plan construction system. Construct the
smallest faithful executable plan for the multi-hop question. Include every
necessary information need, omit irrelevant steps, and follow any semantic
units and planning strategy supplied by the user.

{PLAN_SCHEMA}
""".strip()

VERIFIER_SYSTEM = f"""
You are the Verifier in a retrieval-plan construction system. Check semantic
coverage, producer-consumer dependencies, placeholder bindings, execution order,
strategy consistency, and redundancy. Return JSON only with keys verdict,
issues, and plan. verdict is either accept or revise. The plan must follow:

{PLAN_SCHEMA}
""".strip()

JUDGE_SYSTEM = f"""
You are the Judge in a retrieval-plan construction system. Select exactly one
of the two supplied final candidates after two proposal-verification rounds.
Return the chosen candidate unchanged. Do not repair, merge or invent a plan.
Prioritize coverage of the original question, valid dependencies, executable
order, and non-redundancy. Return only the final plan JSON in this schema:

{PLAN_SCHEMA}
""".strip()

SELF_REFINER_SYSTEM = f"""
You are a single retrieval-planning agent. Across repeated calls, generate,
critique, and revise one executable plan. Preserve all information needs in the
question and ensure valid producer-consumer dependencies. Return JSON only.

{PLAN_SCHEMA}
""".strip()

ANSWER_SYSTEM = """
Answer the retrieval sub-question using only the supplied evidence. Return JSON
only with keys answer and rationale. Keep answer short. If the evidence does not
support an answer, use "Unknown".
""".strip()

FINAL_ANSWER_SYSTEM = """
Answer the original multi-hop question using the executed plan trace and its
retrieved evidence. Return JSON only with keys answer and rationale. Keep answer
short and do not add unsupported facts.
""".strip()

SEMANTIC_JUDGE_SYSTEM = """
Decide whether the prediction is semantically equivalent to the reference answer
for the question. Return JSON only: {"correct": true} or {"correct": false}.
""".strip()


def planning_context(
    question: str, units: list[str] | None, strategy: str | None,
    *, indexed_units: bool = False,
) -> str:
    sections = [f"Original multi-hop question:\n{question}"]
    if units is not None:
        rendered = "\n".join(f"- {unit}" for unit in units) or "- None"
        if indexed_units:
            rendered = (
                "Each U stores one semantic fact. ctx lists the IDs of adjacent facts; "
                "read the fact together with those referenced facts as its local context.\n"
                + rendered
            )
        sections.append(
            f"AMR-derived semantic units (what must be resolved):\n{rendered}"
        )
    if strategy is not None:
        descriptions = {
            "sequential": "a dependency chain whose later steps consume earlier answers",
            "parallel": "independent branches followed by comparison or composition",
            "hierarchical": "a mixture of nested chains and parallel branches",
        }
        sections.append(
            "Planning strategy (how steps should be organized):\n"
            f"{strategy}: {descriptions[strategy]}"
        )
    return "\n\n".join(sections)
