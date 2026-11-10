"""BELLE's five operators and bi-level debate, ACL 2025 section 4 / Table 10.
Prompts preserve the published roles while serializing plans as executable JSON.
Adaptive-step uses prompted complexity classification, not a trained checkpoint.
"""
OPERATORS = {
    "COT": "Reason from model knowledge and existing state, without retrieval.",
    "SINGLE_STEP": "Retrieve once for the current question and answer from evidence.",
    "ITERATIVE_STEP": "Alternate evidence retrieval and a new reasoning step until answered.",
    "SUB_STEP": "Decompose into dependent atomic subquestions, then solve each in order.",
    "ADAPTIVE_STEP": "Route by complexity to no retrieval, single retrieval, or iterative retrieval.",
}
belle_type_classifier_prompt = """Classify the question as Inference (combine
linked facts), Comparison (compare entities), Temporal (reason about dates or
event order), or Null (unanswerable given the question's premise). Return JSON
{"type": "Inference", "reason": "brief explanation"}.
Examples:
Question: Where was the inventor of Machine A born? Output: {"type": "Inference"}
Question: Do City A and City B have the same population? Output: {"type": "Comparison"}
Question: Did Event A occur before Event B? Output: {"type": "Temporal"}
Question: What was the name of the event that never happened? Output: {"type": "Null"}
"""
_DEBATE = """Select a useful combination from the five supplied operators.
Match the operators to the question type and specify their execution order.
Reason about strengths and deficiencies of the plan, not the answer.
It is not necessary to agree on every point. Return a JSON object with a
"view" explanation and an "operators" list of operator names.
When SUB_STEP precedes a solver, that solver applies to every subquestion.
"""
belle_affirmative_prompt = _DEBATE + """You are the affirmative debater. Use
your previous arguments and preceding fast and slow assessments to propose or
defend an operator plan. Change your view when the evidence warrants it."""
belle_negative_prompt = _DEBATE + """You are the negative debater. Examine the
current affirmative proposal, your previous arguments, and preceding fast and
slow assessments. Identify deficiencies and propose a better plan."""
belle_fast_prompt = _DEBATE + """You are the fast debater. Assess current
affirmative and negative arguments using your previous fast assessments.
Identify which current operator choices are reasonable and why."""
belle_slow_prompt = _DEBATE + """You are the slow debater. Integrate all
historical discussion, the current sides, and the fast assessment. Preserve
correct earlier arguments and detect unjustified changes or oscillation.
Update the stable plan and the reasons for keeping or revising it."""
belle_judge_prompt = """Judge affirmative, negative, fast, and slow views and
produce an executable operator plan. Return JSON: {"ready": true,
"operators": ["SUB_STEP", "SINGLE_STEP"], "reason": "brief assessment"}.
In hard mode, set ready=false if another debate round is necessary. In soft
mode (the final round), extract the best plan from the slow debate history and
set ready=true. Use only the five supplied operator names. A SUB_STEP followed
by a solver applies that solver to each dependent subquestion; a final SUB_STEP
uses single-step retrieval for its subquestions. Preserve operator order."""
belle_decompose_prompt = """Decompose into atomic factual subquestions with
explicit answer dependencies. Return JSON {"plan": {"Q1": ["first question",
"<A1>"], "Q2": ["question about <A1>", "<A2>"]}}. Use an acyclic graph and
do not invent intermediate entities. Existing answers may help resolve entities.
Include every fact needed for the final answer."""
belle_cot_prompt = """Work through factual and logical steps using model
knowledge and supplied state. No retrieval is available for this operator.
Return JSON {"reason": "brief derivation", "answer": "concise answer"}.
Examples (the stated premises are sufficient for these illustrative questions):
Question: A was invented by B, who was born in C. Where was A's inventor born?
Output: {"reason": "A's inventor is B; B was born in C.", "answer": "C"}
Question: A has ten members and B has seven. Which group has more members?
Output: {"reason": "Ten exceeds seven.", "answer": "A"}
Question: Event A was in 1902 and event B in 1911. Which happened first?
Output: {"reason": "1902 precedes 1911.", "answer": "A"}
Question: The supplied account does not identify a founder. What is their name?
Output: {"reason": "The premises give no identity.", "answer": "Unknown"}
"""
belle_single_prompt = """Answer the current question from retrieved evidence
and existing state. Return JSON {"answer": "short answer", "reason": "brief
supporting evidence"}. Return Unknown if unsupported."""
belle_iterative_prompt = """Continue reasoning from accumulated passages and
earlier steps. Produce one further reasoning step. If enough information is
present, return JSON {"done": true, "thought": "brief step", "answer": "short
answer", "query": ""}. Otherwise return {"done": false, "thought": "brief
step", "answer": "", "query": "focused query for missing evidence"}.
Resolve intermediate entities in the next query."""
belle_adaptive_prompt = """Choose the retrieval complexity needed for the
question given existing state: A = answer without new retrieval, B = one
retrieval suffices, C = several dependent retrieval steps are needed.
Return JSON {"complexity": "A", "reason": "brief reason"}, choosing A, B or C."""
belle_final_prompt = """Combine completed operator results to answer the
original question. Follow dependencies and perform any final comparison.
Return JSON {"answer": "concise final answer"}. Do not invent unsupported facts."""
