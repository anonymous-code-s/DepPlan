# Experimental results

| Directory | Experiment |
|---|---|
| [main_results](main_results/README.md) | Final-answer performance of DepPlan and baselines |
| [ablations](ablations/README.md) | Full model and component ablations |
| [classification](classification/README.md) | Strategy-classifier results and evaluation commands |
| [parsing_quality](parsing_quality/README.md) | Parsing quality by dataset and overall Smatch scores |

## Per-seed records

Each row in `per_seed.csv` identifies one method or variant, dataset and decoding seed. Seeds are 42, 123 and 2026. CSV files use UTF-8, decimal points and integer counts without thousands separators.

| Dataset key | Benchmark | Development questions |
|---|---|---:|
| `hotpot` | HotpotQA | 7405 |
| `musique` | MuSiQue | 2417 |
| `2wiki` | 2WikiQA | 12576 |

| Field | Meaning |
|---|---|
| `method` / `variant` | Method identifier or ablation configuration |
| `dataset` | Dataset key from the table above |
| `seed` | Decoding seed |
| `questions` | Evaluation denominator, including failed questions |
| `em_correct` | Integer count of exact-match answers |
| `em_percent` | `100 * em_correct / questions` |
| `f1_percent` | Mean answer-level token F1, expressed as a percentage |
| `acc_correct` | Integer count of semantically correct answers |
| `acc_percent` | `100 * acc_correct / questions` |

Semantic accuracy fields apply to the main comparison. Preserve the supplied precision of F1 scores. An unavailable measurement is an empty field, not zero.

## Aggregation

Group records by method or variant and dataset. Compute EM and semantic accuracy directly from integer correct counts and denominators, then calculate the mean and sample standard deviation across the three runs (`ddof=1`). Aggregate F1 from its per-seed scores. Round only the displayed summaries.

The `full` ablation rows reuse the `depplan` main-result runs. Both summary tables are calculated from these shared per-seed records.
