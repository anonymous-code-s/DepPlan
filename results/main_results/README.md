# Main results

The main comparison covers 15 methods, three development sets and decoding seeds 42, 123 and 2026: 135 per-seed records.

| File | Contents |
|---|---|
| [per_seed.csv](per_seed.csv) | Integer correct counts and per-seed EM, F1 and semantic accuracy |
| [mean_sample_sd.csv](mean_sample_sd.csv) | Mean and sample standard deviation for each method and dataset |

EM and semantic accuracy are recomputed from correct counts and dataset sizes before aggregation. F1 is aggregated from the supplied per-seed values at their recorded precision. Sample standard deviations use `ddof=1`. The tables below display percentages to two decimal places; the summary CSV retains additional calculation precision.

## HotpotQA

7,405 questions per seed.

| Method | EM (%) | F1 (%) | Acc (%) |
|---|---:|---:|---:|
| Direct prompting | 23.65 ± 0.10 | 32.95 ± 0.24 | 33.22 ± 0.21 |
| CoT | 23.47 ± 0.23 | 33.18 ± 0.24 | 33.33 ± 0.28 |
| IRCoT | 34.75 ± 0.27 | 45.65 ± 0.39 | 44.70 ± 0.26 |
| ReAct | 38.70 ± 0.17 | 51.67 ± 0.35 | 55.57 ± 0.26 |
| Self-Ask | 31.25 ± 0.26 | 42.15 ± 0.23 | 49.54 ± 0.18 |
| Plan-and-Solve | 33.02 ± 0.16 | 43.82 ± 0.42 | 51.47 ± 0.10 |
| GenGround | 35.14 ± 0.19 | 45.18 ± 0.42 | 47.45 ± 0.23 |
| DEC | 31.75 ± 0.19 | 40.72 ± 0.35 | 37.37 ± 0.16 |
| DeepRAG | 37.37 ± 0.18 | 50.09 ± 0.38 | 48.79 ± 0.18 |
| R-Search | 35.21 ± 0.29 | 46.05 ± 0.36 | 48.17 ± 0.21 |
| CoA | 46.59 ± 0.24 | 59.33 ± 0.36 | 64.71 ± 0.24 |
| GeAR | 52.99 ± 0.40 | 66.62 ± 0.67 | 69.93 ± 0.36 |
| BELLE | 50.63 ± 0.34 | 64.69 ± 0.43 | 67.78 ± 0.41 |
| PER | 44.45 ± 0.25 | 58.15 ± 0.45 | 61.76 ± 0.37 |
| **DepPlan** | **60.81 ± 0.27** | **74.92 ± 0.29** | **78.61 ± 0.33** |

## MuSiQue

2,417 questions per seed.

| Method | EM (%) | F1 (%) | Acc (%) |
|---|---:|---:|---:|
| Direct prompting | 4.88 ± 0.17 | 14.01 ± 0.15 | 9.71 ± 0.21 |
| CoT | 4.04 ± 0.09 | 12.32 ± 0.15 | 10.32 ± 0.10 |
| IRCoT | 11.46 ± 0.12 | 23.47 ± 0.35 | 22.19 ± 0.23 |
| ReAct | 24.51 ± 0.28 | 34.53 ± 0.39 | 37.75 ± 0.24 |
| Self-Ask | 10.23 ± 0.09 | 21.38 ± 0.16 | 19.89 ± 0.13 |
| Plan-and-Solve | 9.56 ± 0.26 | 14.52 ± 0.17 | 14.65 ± 0.15 |
| GenGround | 10.25 ± 0.21 | 15.52 ± 0.30 | 15.18 ± 0.08 |
| DEC | 13.47 ± 0.38 | 17.62 ± 0.50 | 19.20 ± 0.29 |
| DeepRAG | 20.89 ± 0.30 | 32.21 ± 0.38 | 32.74 ± 0.32 |
| R-Search | 15.82 ± 0.13 | 24.48 ± 0.29 | 23.90 ± 0.17 |
| CoA | 29.84 ± 0.30 | 37.74 ± 0.42 | 35.47 ± 0.38 |
| GeAR | 31.43 ± 0.44 | 41.98 ± 0.71 | 42.13 ± 0.37 |
| BELLE | 24.45 ± 0.29 | 34.86 ± 0.43 | 35.15 ± 0.32 |
| PER | 19.94 ± 0.23 | 28.99 ± 0.30 | 28.59 ± 0.29 |
| **DepPlan** | **39.35 ± 0.54** | **47.89 ± 0.56** | **48.49 ± 0.62** |

## 2WikiQA

12,576 questions per seed.

| Method | EM (%) | F1 (%) | Acc (%) |
|---|---:|---:|---:|
| Direct prompting | 25.70 ± 0.18 | 30.41 ± 0.25 | 29.60 ± 0.18 |
| CoT | 25.20 ± 0.21 | 29.95 ± 0.15 | 29.68 ± 0.18 |
| IRCoT | 27.66 ± 0.10 | 35.21 ± 0.14 | 36.68 ± 0.19 |
| ReAct | 29.87 ± 0.20 | 38.82 ± 0.36 | 39.64 ± 0.28 |
| Self-Ask | 26.83 ± 0.15 | 33.76 ± 0.26 | 48.12 ± 0.18 |
| Plan-and-Solve | 23.97 ± 0.23 | 27.73 ± 0.38 | 28.37 ± 0.09 |
| GenGround | 26.69 ± 0.12 | 35.22 ± 0.29 | 37.57 ± 0.15 |
| DEC | 23.88 ± 0.16 | 33.04 ± 0.25 | 35.00 ± 0.10 |
| DeepRAG | 32.66 ± 0.11 | 41.60 ± 0.26 | 42.90 ± 0.15 |
| R-Search | 45.21 ± 0.27 | 54.12 ± 0.48 | 55.60 ± 0.25 |
| CoA | 56.21 ± 0.33 | 63.70 ± 0.44 | 61.72 ± 0.21 |
| GeAR | 58.26 ± 0.40 | 68.49 ± 0.59 | 73.19 ± 0.45 |
| BELLE | 41.07 ± 0.34 | 54.72 ± 0.57 | 54.18 ± 0.22 |
| PER | 27.88 ± 0.10 | 31.24 ± 0.50 | 31.72 ± 0.11 |
| **DepPlan** | **66.43 ± 0.30** | **74.28 ± 0.31** | **76.70 ± 0.43** |

## Method identifiers

| Method | CSV identifier |
|---|---|
| Direct prompting | `direct_prompting` |
| CoT | `cot` |
| IRCoT | `ircot` |
| ReAct | `react` |
| Self-Ask | `self_ask` |
| Plan-and-Solve | `plan_and_solve` |
| GenGround | `genground` |
| DEC | `dec` |
| DeepRAG | `deeprag` |
| R-Search | `r_search` |
| CoA | `coa` |
| GeAR | `gear` |
| BELLE | `belle` |
| PER | `per` |
| DepPlan | `depplan` |

See the [results guide](../README.md) for column definitions and aggregation rules.
