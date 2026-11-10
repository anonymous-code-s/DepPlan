# Classifier sample manifests

We provide the sample IDs and silver labels in two files:

- `data/classifier/train_manifest.jsonl`: IDs and silver labels for the 1,000-question training pool.
- `data/classifier/evaluation_manifest.jsonl`: IDs and silver labels for the disjoint 500-question classifier evaluation set.

Both sets originate in the official training splits of HotpotQA, MuSiQue and 2WikiMultiHopQA. The manifests contain `dataset`, `example_id`, `source_split`, `silver_strategy` and `graph_status`. Evaluation records additionally include `cohort`. The commands below restore question text from the official datasets; graph parsing and feature extraction follow the reproduction guide.

| Dataset | Training questions | Sequential | Parallel | Hierarchical |
|---|---:|---:|---:|---:|
| HotpotQA | 300 | 150 | 150 | 0 |
| MuSiQue | 400 | 175 | 0 | 225 |
| 2WikiQA | 300 | 75 | 75 | 150 |
| Total | 1,000 | 400 | 225 | 375 |

The evaluation set combines a 200-question cohort with 300 questions sampled using official question-type quotas: 90 from HotpotQA, 120 from MuSiQue and 90 from 2WikiQA. It contains 306 sequential, 100 parallel and 94 hierarchical silver labels. Scores describe this stratified evaluation set. The result CSV reports each cohort and their combination.

`graph_status` records valid and failed graphs for each source. GPT-4o-mini has 823 valid training graphs and 428 valid evaluation graphs; Graphene has 499 valid evaluation graphs. Source keys identify the graph parser, including `spring` for `checkpoints/SPRING`. Failed evaluation graphs remain in the scoring denominator.

Restore question text by joining `example_id` to `_id` in HotpotQA/2WikiQA or to `id` in MuSiQue. `scripts/restore_classifier_questions.py` performs this join using separately downloaded official training files and preserves manifest order and labels. See [reproduction instructions](reproduction.md).

Dataset text and pretrained dependencies retain their original terms and are obtained separately.

## Evaluation inputs

| Evaluation | Included files | Additional inputs |
|---|---|---|
| Saved classification scores | Manifests, predictions and summaries | None |
| Classifier inference | GCN checkpoints and configuration | Source-matched question AMRs and BGE encoder |
| Multi-hop QA | Pipeline, prompts and configuration | Dataset questions, candidate passages, question AMRs and model service |
| Smatch | Scoring script and overall scores | Gold AMRs and matching parser outputs |

## Answer review records

[`data/annotations/answer_review/`](../data/annotations/answer_review/README.md) contains 180 saved method answers and their 540 judgments. Case IDs link question text, references and predictions to the individual decisions. Evaluator identities and versions are recorded separately, alongside the shared prompt, sampling manifest and pairwise agreement summaries.
