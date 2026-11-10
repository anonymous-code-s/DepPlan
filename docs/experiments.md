# Sensitivity and diagnostic experiments

Run these commands from the repository root after installation. The implementations are in `src/depplan/experiments/`; JSON configurations are in `configs/experiments/`. Model IDs can be replaced with local model directories. API calls use `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment.

| Experiment | Entry point | Inputs | Model requests |
|---|---|---|---|
| Coverage threshold | `depplan.experiments.coverage` | Saved plans and MuSiQue gold decompositions | None |
| Interaction rounds | `depplan.experiments.rounds` | Fixed question graphs, passages and GCN checkpoint | Planning and QA |
| Retrieval cutoff | `depplan.experiments.retrieval_cutoff` | Fixed questions and per-seed saved plans | QA only |
| Sub-question quality | `depplan.experiments.subquestion_quality` | Generated sub-questions and optional gold decompositions | None; local scoring models |
| Offline preparation cost | `depplan.experiments.offline_cost` | Frozen training graphs, labels and teacher prompt | Silver-label generation |

Our experiments vary one component or setting while keeping the remaining configuration fixed. The runners save their effective settings, input hashes, per-question records and summaries to a new output directory. Multi-seed summaries report the mean and sample standard deviation (`ddof=1`) of per-seed metrics. A single-seed input has no sample SD. EM is aggregated from binary question scores. Failed QA attempts remain in the denominator with zero EM/F1; their elapsed time and recorded token usage remain in the cost totals. Missing token usage is `null`, with the sum of known usage retained separately as a lower bound. Token totals include input and output tokens from all attempts.

`results/` contains the reported results. Running these commands writes new measurements under `outputs/`. The `--config` option loads experiment settings; explicit command-line options override them. `--qa-config configs/qa.json` selects the shared planning and execution settings for rounds and retrieval-cutoff runs.

## Fixed question sets

Prepare questions and cached AMRs as described in [reproduction.md](reproduction.md). `scripts/prepare_qa.py` retains official question types for HotpotQA and 2WikiQA and decomposition lengths for MuSiQue. `parse-amr --batch-size 4` preserves this metadata and records `raw_amr` and per-question `amr_seconds`; measured batch time is divided by its actual question count. This includes parsing and graph decoding, excluding model loading and output serialization. The graph file and measured parsing costs are reused across seeds.

To create one fixed, proportionally stratified set of 1,000 questions per dataset:

```bash
python -m depplan.experiments.sampling \
  --input outputs/hotpot_graphs.jsonl outputs/musique_graphs.jsonl outputs/2wiki_graphs.jsonl \
  --per-dataset 1000 --seed 2026 --output outputs/rounds_cohort
```

Sampling uses official question types for HotpotQA/2WikiQA and 2/3/4-hop strata for MuSiQue. Largest-remainder allocation preserves the requested sample size. The output contains `questions.jsonl` and an ID-level `manifest.json`. Reuse that question file for all conditions and decoding seeds. To reproduce a particular run, use its saved question IDs. Sampling and decoding seeds are separate settings.

## Coverage threshold sensitivity

```bash
python -m depplan.experiments.coverage \
  --config configs/experiments/coverage.json \
  --predictions outputs/musique_seed42/predictions.jsonl \
    outputs/musique_seed123/predictions.jsonl outputs/musique_seed2026/predictions.jsonl \
  --gold /path/to/musique_ans_v1.0_dev.jsonl \
  --output outputs/coverage
```

The scorer builds a cosine-similarity matrix between generated and gold sub-questions with all-MiniLM-L6-v2. It computes a maximum-total-similarity one-to-one assignment once per question, then applies thresholds 0.50–0.90 to those same assignments. Coverage is matched gold steps divided by all gold steps, averaged over questions. A missing plan scores zero. Threshold changes do not trigger planning, retrieval or answer generation.

Every method/seed must contain exactly the IDs in the gold file, including records for failed plans. For a subset, pass the matching subset of gold records. Outputs are `per_question.jsonl` (assignments and threshold scores), `per_seed.csv`, and `mean_sample_sd.csv`, including overall and hop-specific summaries. A single `predictions.jsonl` can be used for a single-seed scan.

## Agent interaction rounds

```bash
python -m depplan.experiments.rounds \
  --config configs/experiments/rounds.json --qa-config configs/qa.json \
  --input outputs/rounds_cohort/questions.jsonl \
  --checkpoint checkpoints/SPRING --device cuda:0 \
  --output outputs/rounds
```

Each round consists of a Proposer generation or revision followed by a Verifier response. The final Judge selects between the latest candidates when needed. One, two and three rounds use the same AutoGen roles, prompts, classifier, retrieval and answer scorer. The two-round setting is the full pipeline used by `run-qa`. API retry limits remain fixed across round settings; no extra plan-repair calls are added.

Results are organized as `dataset/rounds_N/seed_N/`. Each cell contains complete call logs, predictions, configuration and a summary. Root-level CSV files report EM/F1, online and total seconds, steps, LLM calls, token usage and failures. Online time covers strategy selection, semantic units, passage encoding, planning and execution. Total time adds the question's measured AMR preprocessing time. Supply `amr_seconds` in the question records or a separate `--amr-timings` JSONL with `dataset`, `example_id`, and `amr_seconds`; without measured AMR times, total time stays unavailable. Scoring and model loading are excluded.

Add `--dry-run` to validate the grid without loading models or making requests. `--resume` continues completed question prefixes only when settings, code and inputs match. An interruption within a question requires a new output directory; the partial call log is retained. The rounds experiment uses EM/F1 by default; optional `--semantic-accuracy --judge-model <model> --judge-base-url <endpoint>` enables independent semantic scoring, with separate evaluator costs.

## Retrieval top-k

```bash
python -m depplan.experiments.retrieval_cutoff \
  --config configs/experiments/retrieval_cutoff.json --qa-config configs/qa.json \
  --input outputs/musique_graphs.jsonl \
  --plans outputs/musique_seed42/predictions.jsonl \
    outputs/musique_seed123/predictions.jsonl outputs/musique_seed2026/predictions.jsonl \
  --output outputs/retrieval_cutoff
```

The input plans determine the seed set. All cutoffs reuse the same saved plan for a given question and seed. Only retrieval and answer execution are repeated at k = 1, 3, 5, 10, 15, 20. A plan-generation failure remains a failed question at every cutoff. The runner checks question IDs and texts and records each plan's hash in the output.

`mean_qa_seconds` covers retrieval, intermediate-answer generation and final-answer generation. It excludes AMR parsing, classifier inference, plan construction, cached passage encoding and answer scoring. Token usage includes QA calls only. These costs have a different scope from end-to-end runtime. The output grid is `dataset/seed_N/k_N/`, with per-seed and mean/sample-SD CSV summaries. `--dry-run` and `--resume` follow the same conventions as the rounds runner.

## Sub-question quality and token length

DepPlan prediction files expose sub-questions through `plan.steps[].sub_question`. Other saved workflows can supply a normalized JSONL with one record per question:

```json
{"method":"belle","dataset":"musique","example_id":"example-1","seed":2026,"subquestions":["Who wrote Example Book?","Where was its author born?"]}
```

Keep the generated question text, including dependency placeholders, without rewriting it. Use an empty list for questions without generated sub-questions. Do not substitute reasoning statements or answers for questions. Each method's question IDs must match across seeds.

```bash
python -m depplan.experiments.subquestion_quality \
  --config configs/experiments/subquestion_quality.json \
  --predictions outputs/musique_seed2026/predictions.jsonl outputs/belle_subquestions.jsonl \
  --gold /path/to/musique_ans_v1.0_dev.jsonl \
  --output outputs/subquestion_quality
```

Length counts GPT-2 tokens per sub-question, without added special tokens. Fluency-NLL is the mean next-token negative log-likelihood within each sub-question; a one-token question has length but no next-token NLL. Overlength inputs are rejected rather than truncated. MuSiQue similarity is each generated sub-question's maximum cosine similarity to the gold sub-questions, without a coverage threshold. Per-seed scores average generated sub-questions equally; `per_question.jsonl` also retains question-level means and counts. Questions with no sub-questions are counted separately and excluded from sub-question quality averages.

Well-formedness can be evaluated with `--wellformed-model /path/to/classifier --positive-label-id 1` using the classifier's positive-class probability. Alternatively, `--wellformed-scores scores.jsonl` imports existing scores with fields `method`, `dataset`, `example_id`, `seed`, zero-based `subquestion_index`, and `score` in [0,1]. The import must cover exactly the supplied sub-questions. Provide an external well-formedness classifier or saved scores; the metric is omitted when neither is supplied. `--length-only` skips NLL and semantic similarity and loads only the tokenizer.

Outputs are `subquestions.jsonl`, `per_question.jsonl`, `per_seed.csv` and `mean_sample_sd.csv`. These files retain the individual observations used by the [paired-analysis tools](plan_statistics.md#nested-paired-bootstrap).

## One-off silver-label and classifier-training cost

Restore the frozen 1,000-question training pool using `data/classifier/train_manifest.jsonl`, then prepare graphs as described in [reproduction.md](reproduction.md#2-restore-question-texts-and-prepare-graphs). Graphs must retain the original `silver_strategy` labels and either `raw_amr` or `amr_root` with `amr_triples`.

```bash
python -m depplan.experiments.offline_cost \
  --config configs/experiments/offline_cost.json \
  --input outputs/reference_training_graphs.jsonl \
  --training-manifest data/classifier/train_manifest.jsonl \
  --prompt prompts/silver_strategy.txt \
  --device cuda:0 --output outputs/offline_cost
```

The runner checks all question IDs and original labels against the manifest before making requests. The fixed question-plus-AMR prompt generates fresh labels with greedy decoding. `--workers` controls concurrency against one configured endpoint. Labeling time is elapsed wall-clock time for the workload, including retries; it is not the sum of overlapping request latencies. Logs retain both request-level latency and workload time, all token usage, failures and agreement with the original labels.

For the training-time measurement, we reuse the original frozen labels. The trainer measures 30 epochs after BGE feature preparation, with batch size 16 and AdamW, and writes the new weights and history to `outputs/offline_cost/retrained_classifier/`. `labeling_cost.json`, `retrained_classifier/timing.json` and `summary.json` report the two cost components separately. Use `--dry-run` to validate inputs without requests or training.
