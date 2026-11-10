# AMR parsing quality

The parsing comparison reports Smatch F1 (%) on 100 manually annotated AMR graphs: 30 from HotpotQA, 40 from MuSiQue and 30 from 2WikiQA. Each dataset score is corpus-level Smatch F1. Overall is the sample-weighted mean of the three dataset scores, using weights 0.3, 0.4 and 0.3.

| Structure source | HotpotQA | MuSiQue | 2WikiQA | Overall |
|---|---:|---:|---:|---:|
| CAIL | 76.63 | 75.34 | 77.55 | 76.39 |
| T5-base | 78.96 | 77.60 | 79.74 | 78.65 |
| Graphene | 80.75 | 78.95 | 81.05 | 80.12 |
| LLM-based (GPT-4o-mini) | 72.01 | 70.71 | 73.17 | 71.84 |
| SPRING | 82.85 | 80.97 | 84.09 | 82.47 |

[smatch.csv](smatch.csv) contains the overall scores and annotation-set size. The [checkpoint guide](../../docs/checkpoints.md) identifies the graph sources used in the comparison.

## Scoring

Smatch scoring takes aligned predicted and gold AMR files in PENMAN format. Run [scripts/score_smatch.py](../../scripts/score_smatch.py) for each dataset, then compute the sample-weighted overall score. The [scoring instructions](../../docs/reproduction.md#6-score-amr-parsing) describe the command and input format.
