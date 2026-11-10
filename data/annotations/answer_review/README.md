# Answer review records

This collection contains 180 saved answers: 20 per method and 60 per dataset. GPT-4.1-mini is recorded as `gpt41_mini`; the two human annotators use the aliases `annotator_1` and `annotator_2`. Their roles and the automated evaluator's settings are in `evaluators.json`. Each evaluator receives the same correctness prompt, without method names or other evaluators' decisions.

| File | Contents |
|---|---|
| `cases.jsonl` | Question IDs, text, references, predictions, generation seeds where available, and answer-level EM/F1 |
| `judgments.csv` | All 540 decisions, joined to answers by `case_id` |
| `evaluators.json` | Evaluator identities, versions and settings |
| `agreement.csv` | Pairwise agreement and Cohen's kappa overall, by method and by dataset |
| `judge_prompt.txt` | Shared correctness prompt |
| `manifest.json` | Sampling information, record counts and source-file hashes |

Labels are `1` for correct and `0` for incorrect. `evaluator_id` identifies the scorer in `evaluators.json`; `response` records the yes/no decision. EM and F1 in `cases.jsonl` are fractions in [0,1]. Missing generation metadata remains `null`. Methods may share questions, so an answer record is identified by its method and question together.

Agreement uses paired labels on the same answers. The CSV reports the number of pairs, agreements, agreement percentage, Cohen's kappa and the two disagreement counts. The original labels and outputs are retained without adjudication.
