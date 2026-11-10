# Strategy-classifier results

The five classifiers are evaluated against silver strategy labels on the same 500 held-out questions from the official training splits. These questions are disjoint from the shared 1,000-question training pool. The GPT-4o-mini source contains 823 valid training graphs. Graph failures remain in the evaluation denominator and contribute false negatives to Macro-F1.

| Structure source | Training graphs | Correct / 500 | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| SPRING | 1,000 | 426 | 85.20 | 82.53 |
| CAIL | 1,000 | 414 | 82.80 | 78.57 |
| T5-base | 1,000 | 415 | 83.00 | 79.80 |
| Graphene, three-source ensemble | 1,000 | 419 | 83.80 | 80.78 |
| GPT-4o-mini AMR | 823 | 327 | 65.40 | 64.01 |

[summary.csv](summary.csv) contains the scores for the original 200-question subset, the additional 300 questions and the combined evaluation set. Each source's JSONL file contains the individual predictions, silver labels and graph status. [reproduction.json](reproduction.json) records a CPU rerun of the five checkpoints against the saved predictions.

## Evaluation commands

Recompute scores from the saved predictions:

```bash
python -m depplan score-classification \
  --input results/classification/spring.jsonl
```

To evaluate a checkpoint on prepared question graphs, follow the [graph preparation instructions](../../docs/reproduction.md#2-restore-question-texts-and-prepare-graphs), then run:

```bash
python -m depplan evaluate-classifier \
  --checkpoint checkpoints/SPRING \
  --input outputs/reference_evaluation_graphs.jsonl \
  --device cpu --output outputs/spring_predictions.jsonl
```

Select the checkpoint matching the graph source. Add `--encoder /path/to/bge-base-en-v1.5` to use a local BGE installation. The [checkpoint guide](../../docs/checkpoints.md) provides model and training settings, and the [data guide](../../docs/data.md) describes the question manifests.
