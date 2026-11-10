# Ablation results

| File | Contents |
|---|---|
| `per_seed.csv` | Per-seed correct counts and EM/F1 scores |
| `mean_sample_sd.csv` | Mean and sample standard deviation across the three decoding runs |

`per_seed.csv` is the source for the ablation summaries. The full model uses the same DepPlan records as `../main_results/per_seed.csv`. EM is computed from correct counts and dataset sizes; F1 retains its recorded two-decimal precision. Summaries use unrounded count-based EM and sample standard deviations (`ddof=1`).

Each variant–dataset pair has one row per decoding seed: 42, 123 and 2026. The evaluation sizes are 7,405 for HotpotQA, 2,417 for MuSiQue and 12,576 for 2WikiQA.

| Variant | CSV identifier |
|---|---|
| DepPlan (Full) | `full` |
| w/ Length-Matched Text Surrogate | `text_surrogate` |
| w/o Semantic Units | `no_semantic_units` |
| w/o Planning Strategy | `no_planning_strategy` |
| w/ Direct LLM Strategy | `direct_llm_strategy` |
| w/ Majority-Class Strategy | `majority_class_strategy` |
| w/ Proposer Only | `proposer_only` |
| w/ Proposer + Verifier | `proposer_verifier` |

Identifiers match the [ablation runner](../../docs/ablations.md). See the [results guide](../README.md) for column definitions.
