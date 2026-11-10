# Baseline prompts

Each module contains the complete prompts for one method. The active workflow in `baselines/methods/` assembles the question, retrieved passages and interaction history. `calls.jsonl` records the exact resulting request for every generation.

| Method | Module | Active prompt stages |
|---|---|---|
| Direct prompting | [direct_prompting.py](direct_prompting.py) | Six-example concise QA |
| CoT | [cot.py](cot.py) | `cot_runtime_prompt`: six-example reasoning and final answer |
| IRCoT | [ircot.py](ircot.py) | `ircot_runtime_prompt`, `ircot_reader_prompt`: next sentence and separate final reader |
| ReAct | [react.py](react.py) | `react_runtime_prompt`: six official demonstrations and action history |
| Self-Ask | [self_ask.py](self_ask.py) | `self_ask_runtime_prompt`: follow-up generation and answer synthesis |
| Plan-and-Solve | [plan_and_solve.py](plan_and_solve.py) | `plan_solve_runtime_prompt`: zero-shot plan/solve; concise answer extraction |
| GenGround | [genground.py](genground.py) | `deduction_runtime_prompt`, `grounding_runtime_prompt` |
| DEC | [dec.py](dec.py) | Runtime decomposition, history-aware rewrite, keyword extraction |
| DeepRAG | [deeprag.py](deeprag.py) | Released instruction, model chat template, partial assistant continuation and control markers |
| R-Search | [r_search.py](r_search.py) | NL-DAG plan and evidence-conditioned answer; invalid plans fail without repair under the default evaluation protocol; optional original trained-model protocol |
| CoA | [coa.py](coa.py) | Worker communication and final manager |
| GeAR | [gear.py](gear.py) | Graph extraction, proximal triples, gist reading, answerability, rewrite and final QA |
| BELLE | [belle.py](belle.py) | Type classifier, affirmative/negative/fast/slow/judge debate, five operators and final QA |
| PER | [per.py](per.py) | Official few-shot type classifier, bridge/comparison planners, HyDE, retrieval executor and type-specific reasoners |

The concise passage-reader and final-chain synthesis prompts shared by classic methods are in [classic.py](../methods/classic.py). The table lists the active prompt functions; the classic modules also expose template constants for compatibility. ReAct and PER retain their upstream MIT notices. Published prompt sources and authored adaptations are recorded in the [parameter references](../references/).
