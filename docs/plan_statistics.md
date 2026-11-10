# Plan, intermediate-answer and paired statistical evaluation

We evaluate plan structure, intermediate answers and paired differences using
saved per-question outputs. The commands below use the run records and question
IDs to calculate the metrics and statistical comparisons in our paper.

## Plan and intermediate-answer quality

Run from the repository root after installation:

```bash
python -m depplan.experiments.plan_quality \
  --predictions outputs/musique_seed42/predictions.jsonl \
    outputs/musique_seed123/predictions.jsonl \
    outputs/musique_seed2026/predictions.jsonl \
  --gold /path/to/musique_ans_v1.0_dev.jsonl \
  --dependency-mode both --output outputs/plan_quality
```

`--dependency-mode sequence` reproduces the common representation used for the
cross-method plan comparison. `native` uses the recorded dependency graph. `both`
scores the same sub-questions, alignment, and execution order under both edge
representations. The default encoder is
`sentence-transformers/all-MiniLM-L6-v2`, replaceable with a local model directory
through `--encoder`. The default similarity threshold is 0.70.

Every method must contain exactly one record for every question in the supplied
gold file, for each of seeds 42, 123 and 2026. Failed runs must have explicit
question records. An exact subset of the gold file can be supplied for a cohort
analysis. `--seeds 2026` is available for a single-run diagnostic, which has no
sample SD. Missing seeds or IDs, duplicate records, mismatched question text,
unknown dependency references, and cyclic graphs cause errors.

The gold input is the official MuSiQue format, including `id`, `question` and
`answer` in every `question_decomposition` step. Gold dependencies are the
producer-consumer references `#1`, `#2`, etc., resolved against the explicit
decomposition step IDs. Supplied answer aliases are retained. Question-level
IDs may have the `musique:` prefix or use the original ID.

The scorer implements these definitions:

| Quantity | Definition |
|---|---|
| Alignment | Maximum-total-cosine one-to-one assignment, followed by the 0.70 threshold; matching is not rerun after removing low-similarity pairs. |
| Coverage | Accepted gold matches divided by the number of gold steps. |
| Sequence dependencies | Directed edges between adjacent nodes in the recorded execution order. |
| Native dependencies | The recorded `depends_on` edges, with no added sequential edges. |
| Dependency F1 | `2 * correct_aligned_edges / (predicted_edges + gold_edges)`. All predicted edges remain in the denominator, including edges with unmatched endpoints. |
| Ordering Accuracy | Fraction of all transitively gold-constrained ordered step pairs that appear in the right order. A pair with either endpoint unmatched counts as incorrect. Independent gold pairs do not enter the denominator. |
| Intermediate-answer F1 | Token F1 of matched step answers, zero for missing answers or unmatched gold steps; average over all gold steps in that question. A failed QA question contributes zero even if a partial execution trace contains answers. |

A missing or empty plan receives zero for all four metrics. When a nonempty
plan and gold graph both have no edges, dependency F1 is one; when the gold
graph has no constrained pairs, ordering accuracy is one. A valid plan retained
after answer-generation failure can still receive
plan-quality scores; its intermediate-answer F1 is zero.

All four output metrics have the `_percent` suffix and lie in [0,100]. They are
first averaged equally over questions, then summarized as mean and sample SD
across decoding seeds. Outputs include both overall and 2/3/4-hop groups:

- `configuration.json`: input hashes, models, thresholds and metric definitions;
- `per_question.jsonl`: alignments, accepted matches, gold/predicted edges,
  constrained-pair counts, gold-step answer F1, and all four metrics;
- `per_seed.csv` and `mean_sample_sd.csv`: question-level aggregation, including
  failed-question counts and hop groups.

### Saved output contracts

DepPlan's `plan.steps`, `plan.execution_order`, and execution `steps[].step_id`
and `steps[].answer` are consumed directly. Answers are joined by ID, not by
array position. Original sub-question templates are used for alignment.

The baseline runner's Self-Ask `trace` records are also accepted directly:
`kind="follow_up"` supplies an observed question and answer. Their recorded order
supports sequence evaluation; a native graph is unavailable unless explicitly
provided.

Other baseline output formats must supply an `evaluation_plan` derived from
their saved generated questions, intermediate answers and recorded execution
order. The expected record format is:

```json
{
  "method": "baseline_name",
  "dataset": "musique",
  "example_id": "original-question-id",
  "seed": 42,
  "status": "success",
  "question": "Original main question?",
  "evaluation_plan": {
    "steps": [
      {"id": 1, "sub_question": "Who wrote Example Book?", "answer": "Ada"},
      {"id": 2, "sub_question": "Where was its author born?", "answer": null}
    ],
    "execution_order": [1, 2]
  }
}
```

The record uses the exact saved question wording. `answer: null` means no saved
intermediate answer and receives zero; it must not be filled from the final
answer or gold labels. Include `depends_on` for **every** step to enable native
evaluation. Omit that field if a native graph was not recorded. For an absent
plan set `evaluation_plan` to `null`. Preserve the true failure `status`, one
record per question, the decoding seed, and the same cohort across methods.
Convert other baseline traces to this format before scoring.

## Nested paired bootstrap

The statistical entry point accepts actual per-question JSONL observations from
the QA, cost, plan-quality, or sub-question-quality runners. A family file names
every comparison in the correction family. The scorer validates the complete
family before calculating the Bonferroni correction.

For example, create the complete nine-comparison main-result family:

```bash
python - <<'PY'
import json
from pathlib import Path
comparisons = [
    {"name": f"{dataset}_{metric}", "dataset": dataset,
     "method_a": "depplan", "method_b": "gear", "metric": metric, "scale": 100}
    for dataset in ["hotpot", "musique", "2wiki"]
    for metric in ["em", "f1", "semantic_correct"]
]
out = Path('outputs/main_family.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"name": "main_vs_gear", "comparisons": comparisons}, indent=2))
PY

python -m depplan.experiments.bootstrap \
  --input outputs/main_per_question.jsonl \
  --family outputs/main_family.json \
  --replicates 10000 --bootstrap-seed 2026 \
  --output outputs/main_significance
```

`--input` accepts multiple files, so a combined JSONL is not required. Each
record needs `method` (or `variant`), `dataset`, `example_id`, and `seed`.
`variant="full"` is normalized to `method="depplan"`. The default three matched
seeds are 42, 123 and 2026. The bootstrap seed controls resampling and is separate
from the three decoding seeds.

For each replicate, sample three matched run pairs with replacement. Within
each selected pair, resample N question IDs with replacement using **the same
indices for both methods**. Independently resample the instances when a run
pair is drawn twice. Average the three paired differences. The observed delta
is the mean difference of the three complete run pairs. Report the marginal
2.5th and 97.5th percentiles of the bootstrap differences, using NumPy linear
percentile interpolation.

Let `b` be the smaller of the replicate counts at or below zero and at or above
zero. The two-sided value is `min(1, 2*(b+1)/(B+1))`; zeros enter both tail
counts. For a family containing m comparisons, the adjusted value is
`min(1, m*p_raw)`. `significant_0_05` is true when the adjusted value is below
0.05. Intervals remain marginal 95% intervals; they are not Bonferroni-adjusted.

Each comparison supports:

| Field | Meaning |
|---|---|
| `name` | Unique comparison name. |
| `method_a`, `method_b` | Delta is A minus B. |
| `dataset` | `hotpot`, `musique`, `2wiki`, or `all` for a joint cohort such as the 600-question selector comparison. |
| `metric` | Shared numeric field name, with dot paths supported (for example `usage.total_tokens`). |
| `metric_a`, `metric_b` | Override the field for each side when runners have different schemas, such as `online_seconds` versus `time_seconds`. Use a field that includes AMR time when comparing total pipeline time. |
| `scale` | Positive output-unit multiplier: 100 for EM/F1 fractions, 0.001 for tokens to k, 1 for seconds or already-percent metrics. |
| `where_a`, `where_b` | Exact field-value filters; use `{"dependency_mode":"sequence"}` for plan-score files containing both representations. |
| `weight_a`, `weight_b` | Optional fields for per-question observation weights, described below. |

For sub-question quality, the reported run mean weights generated sub-questions
equally. Use `per_question.jsonl` with the mean quality metric and
`weight_a="subquestions"`, `weight_b="subquestions"`. Each resampled pair then
recomputes the weighted run mean on each side before taking the difference.
Questions with zero sub-questions keep their sampling slot with weight zero;
their unavailable (`null`) mean is accepted only with that explicit zero weight.
A complete run or resampled replicate with zero total weight returns an error. Leave weights
unspecified for ordinary per-question EM, F1, runtime and token means.

Both methods and all three seeds must have exactly the same question IDs after
the declared filters. Duplicate observations, unexpected seeds, missing values,
non-finite metrics and missing fields cause errors. Recorded QA failures may
carry zero answer scores. Missing observations and unavailable API usage remain
missing values; they require completion before the comparison can run. Input
must contain paired per-question records.

The output includes `comparisons.json`, `comparisons.csv`, input hashes and
full settings in `configuration.json`, and all actual bootstrap draws in
`bootstrap_differences.npz`. The NPZ arrays `comparison_0`, `comparison_1`, etc.
follow family order and use the requested output units. Each comparison records
its derived resampling seed, tail counts, family size, raw/adjusted p-values,
cohort size, and matched decoding seeds.

The 42-comparison ablation family, seven-comparison sub-question family,
six-comparison pipeline-cost family and two-comparison selector-cost family
use separate family files, following the correction groups in our paper.
