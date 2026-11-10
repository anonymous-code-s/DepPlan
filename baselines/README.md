# Baseline reproduction

We implement fourteen baseline workflows with a shared dataset reader, logged model client and answer scorer. Each method has its own prompts, retrieval procedure and stopping conditions. The configuration files distinguish published parameters from defaults selected for this implementation.

## Layout

```text
baselines/
├── run.py                       # CLI, per-question records and seed summaries
├── core.py                      # Logged chat/completion calls and method context
├── retrieval.py                 # Dense, BM25 and hybrid candidate retrieval
├── data.py                      # Official JSON/JSONL input normalization
├── methods/                     # Executable method workflows
├── prompts/                     # Complete prompts, one module per method
├── configs/
│   ├── common.json              # Shared service and evaluation settings
│   ├── methods/                 # Fourteen method configurations and parameter sources
│   ├── endpoints.example.json   # Per-method model services
│   ├── implementations.csv      # Implemented variants and adaptations
│   └── original_settings.csv    # Original-paper experimental settings
├── references/
│   ├── classic_parameters.md    # Direct, CoT, IRCoT, ReAct, Self-Ask, PS, GenGround, DEC
│   ├── agent_parameters.md      # CoA, GeAR, BELLE, PER
│   ├── learned_parameters.md    # DeepRAG, R-Search
│   └── published_results.csv    # Published scores and source locations
└── export_comparison.py         # Published scores and our results
```

## Configurations

`configs/common.json` is merged with the selected method's JSON. Each operational field has a `parameter_sources` record: `paper` or `code` for a published setting, `choice` for a selected default, and `adaptation` for a changed model, prompt or retrieval environment. Command-line overrides are also recorded. The source notes link to papers and commit-pinned code and explain disagreements between published defaults.

The following are the effective default settings. Token limits apply to each generation call, with stage-specific limits where shown. Retrieval counts apply to a query, not to the whole question trajectory.

| Method ID | Temperature | Output token limit | Retrieval | Termination / bounds |
|---|---:|---|---|---|
| `direct_prompting` | 0 | 128 | None | One answer call |
| `cot` | 0 | 512 | None | One reasoning-and-answer call |
| `ircot` | 0 | 300 | BM25, top 6 | Answer marker, 8 reasoning steps, 15 collected passages, or empty generation; then a separate QA reader |
| `react` | 0 | 100 | Search top 1; Lookup within current page | `Finish`, or 7 action turns |
| `self_ask` | 0 | 256 | Dense top 3 | Final-answer marker, or 8 controller turns |
| `plan_and_solve` | 0 | 256 reasoning / 32 answer | None | Plan-and-solve generation followed by answer extraction |
| `genground` | 0 | 512 generation / 128 final | Dense top 10, batches of 3 | Final answer, or 8 deduction turns; grounding stops at the first supporting batch |
| `dec` | 0 | 512 planning / 128 keyword or answer | Dense top 10; top 2 plus candidates containing every extracted keyword | Completion of the decomposed plan, capped at 8 steps |
| `deeprag` | 0 | Up to 4096 within remaining context | BM25 top 3 | Learned final marker, or depth 6 on HotpotQA / 10 otherwise |
| `r_search` | 0.6 plan / 0 answer | 1000 plan / 2000 answer | Dense top 3 per node | Valid NL-DAG execution and final answer; at most 6 nodes; invalid plan fails without repair |
| `coa` | 0 | 1024 worker / 1024 manager | Sequential access to all supplied context | All source chunks processed, then manager |
| `gear` | 0 | 1024 | Hybrid top 10 per round; final QA top 5 | Answerability decision, or 2 rounds on HotpotQA/2WikiQA and 4 on MuSiQue |
| `belle` | 0 | 1024 | BM25 top 5 | Debate agreement or 3 debate rounds, then operator execution; iterative operator capped at 6 rounds |
| `per` | 0 | 1024 | HyDE + dense top 5 per plan node | All nodes executed; only a complete supported context graph reaches the reasoner |

All default runs use seeds 42, 123 and 2026. The shared generator is Qwen2.5-32B-Instruct, except DeepRAG, which requires the released `xinyan233333/DeepRAG-8b` weights. BGE-base-en-v1.5 supplies dense embeddings. Retrieval uses each question's supplied candidate passages. Direct prompting, CoT and Plan-and-Solve receive no passages. CoA reads the complete candidate context in source order.

Final-answer generation is greedy for every method. DepPlan's Proposer–Verifier–Judge workflow uses planning temperature 0.7; baseline planning retains the method settings in the table, including R-Search's temperature 0.6. Under the default evaluation protocol, invalid R-Search plans and invalid ReAct actions fail without extra LLM repair calls. Their attempted generations remain in cost accounting. GeAR passes previous queries to its rewriter, but query repetition itself is not a stopping rule.

For DEC, we use Qwen to extract keywords that filter candidate documents. For R-Search, we implement the public NL-DAG workflow with Qwen; the referenced repository provides the training code without a trained checkpoint. An optional mode supports an externally supplied trained checkpoint and endpoint. DeepRAG uses the released weights at a pinned revision, whose calibration stage is unspecified. The [implementation records](configs/implementations.csv) and parameter references describe these choices.

The operational defaults are an 8192-token context, a 180-second request timeout, a 900-second question timeout, and at most three attempts for transient API errors. Oversized contexts and invalid plans are recorded as failures. Validated method-specific caps remain in effect; the agent workflows additionally have a 256-execution-step guard. These operational limits are selected defaults.

## Run

Install from the repository root with `python -m pip install -e .`. Set `OPENAI_BASE_URL` and, if required by the service, `OPENAI_API_KEY`. The service must expose the model name used by the selected configuration.

Inspect any method without model loading:

```bash
python -m baselines.run --method gear --list
```

Validate an official development file without inference:

```bash
python -m baselines.run --input data/musique_dev.jsonl \
  --dataset musique --validate-only
```

Run one method on three seeds:

```bash
python -m baselines.run --method dec \
  --input data/musique_dev.jsonl --dataset musique \
  --seeds 42 123 2026 --output outputs/baselines
```

Replace `dec` with any ID in the configuration table. `--dataset hotpot` accepts the official HotpotQA JSON array; `--dataset 2wiki` accepts the official 2Wiki JSON array. Prepared JSONL can instead include `dataset`, `example_id`, `question`, `passages` and `answers`; each passage contains `id`, `title` and `text`. References and supporting annotations are not passed to the method.

For all methods, serve DeepRAG separately and use the endpoint mapping:

```bash
python -m baselines.run --input data/musique_dev.jsonl --dataset musique \
  --endpoints baselines/configs/endpoints.example.json \
  --seeds 42 123 2026 --output outputs/baselines
```

Methods and seeds run sequentially. DeepRAG's endpoint must support raw `/completions` with stop strings retained. Its tokenizer revision matches the pinned checkpoint. The example maps DeepRAG to port 8001; other methods use `OPENAI_BASE_URL`. A per-method mapping may set `base_url`, `api_key_env`, `model` and `tokenizer`. Credentials are read from environment variables.

Use `--config settings.json` for a one-method override; use `--model`, `--tokenizer`, `--encoder` and `--device` to select deployed aliases or local model directories. To use a local tokenizer with no remote revision, set `tokenizer_revision` to `null` in the override file. `--limit` selects the first specified number of records for a development check. `--resume` requires the same inputs, configuration, endpoint and source fingerprint, and skips already recorded questions, including recorded failures.

EM and F1 use the shared dataset-specific scorer. Semantic accuracy uses an independent evaluator with the shared fixed prompt, temperature 0, and a 64-token output budget. Set `JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY`, or pass `--judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL"`. The evaluator requires its own model and endpoint settings. When unconfigured, semantic correctness and accuracy are `null`, with `judge_status="disabled"`; `--no-semantic-accuracy` explicitly disables it even if environment variables are set.

With semantic evaluation enabled, a QA failure receives zero on all answer metrics. An evaluator error or nonbinary response records an error and leaves semantic correctness `null`. Complete-run accuracy is reported once every question has a valid judgment. Evaluator calls, time, and token usage are recorded separately from inference cost.

The optional endpoint-map fields `model_revision`, `model_sha256`, `vllm_version`, `dtype`, `tensor_parallel_size`, and `hardware` are retained as `declared_serving_metadata` in the run manifest. They should describe the service actually deployed. Each successful API call also records the returned model name, response ID, creation timestamp, system fingerprint, and request ID when provided. Resume checks cover baseline source/configuration files and the shared answer scorer and semantic-judge prompts.

## Outputs and cost accounting

Each `outputs/baselines/<method>/<dataset>/seed_<seed>/` directory contains:

- `manifest.json`: effective configuration, source/input fingerprints, question IDs and endpoint identity.
- `predictions.jsonl`: final answers, reference scores, status, elapsed time, method steps, retrieval traces and aggregate model usage.
- `calls.jsonl`: exact prompts, outputs, generation parameters, attempts, token usage and latency for every model call.
- `judge_calls.jsonl`: answer-scoring calls when an independent evaluator is configured.
- `summary.json`: per-seed accuracy, failures and average inference costs.

`mean_sample_sd.csv` reports seed means and sample standard deviations. Failures remain in the EM/F1 denominator. Missing API usage produces an unknown total alongside `known_total_tokens`. Inference timers cover question-specific indexing/encoding, method calls, retrieval, retries, final answer generation, and synchronous call logging. Tokenizer/model loading and answer scoring are excluded. GeAR graph construction is included and separately identified in the trace. `scoring_time_seconds` covers lexical and semantic scoring; `judge_time_seconds`, `judge_llm_calls`, and `judge_*tokens` isolate evaluator cost. Per-seed summaries and aggregate CSVs keep these scoring costs separate from inference fields.

`steps` counts each method's execution units: classic controller turns, DeepRAG reasoning depth, R-Search/PER plan nodes, CoA workers, GeAR retrieval rounds, or BELLE primitive solver iterations. `llm_calls` separately includes all planning, extraction, debate, answer and retry calls. Use the call trace for comparisons requiring a common action definition.

## Published-score comparison

[Published results](references/published_results.csv) retain original backbones, metric definitions and exact table locations. [Per-seed results](../results/main_results/per_seed.csv) and their summaries are stored under `results/`; new runs write to `outputs/`. Generate the comparison with published scores using:

```bash
python baselines/export_comparison.py
```

The command joins our result CSVs with the published-score records.
