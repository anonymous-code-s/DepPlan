# Component ablations

The component study uses the same QA pipeline as `run-qa`. The full model and seven variants share question IDs, candidate passages, graph inputs, retrieval settings, plan validation and answer scoring. Each condition runs with seeds 42, 123 and 2026. EM/F1 are always computed. Configure `JUDGE_MODEL`, `JUDGE_BASE_URL` and `JUDGE_API_KEY` (or `--judge-*`) to compute Acc with an independent evaluator; without that configuration Acc remains `null`. See [evaluation setup](reproduction.md#5-run-multi-hop-qa).

## Conditions

| CLI name | Variant | Change from the full model |
|---|---|---|
| `full` | DepPlan (Full) | AMR semantic units, GCN strategy, Proposer–Verifier–Judge |
| `text_surrogate` | w/ Length-Matched Text Surrogate | Replace the semantic-unit text with length-matched question text |
| `no_semantic_units` | w/o Semantic Units | Omit the semantic-unit block; retain the AMR-based classifier |
| `no_planning_strategy` | w/o Planning Strategy | Omit strategy guidance and skip classifier inference |
| `direct_llm_strategy` | w/ Direct LLM Strategy | Predict the strategy from the question alone using the QA backbone |
| `majority_class_strategy` | w/ Majority-Class Strategy | Use the most frequent silver label in the classifier training manifest |
| `proposer_only` | w/ Proposer Only | Execute the first Proposer plan; no Verifier or Judge |
| `proposer_verifier` | w/ Proposer + Verifier | Keep both proposal–verification rounds; execute the final Verifier plan |

The full workflow uses two proposal–verification rounds. When the final valid candidates differ, the Judge selects one unchanged. Identical valid candidates require no Judge request. All variants use the same executable-plan validation, including exact agreement between dependency IDs and answer placeholders.

Planning roles are AutoGen AgentChat agents, implemented in `src/depplan/agents.py`. `planner.py` routes their messages in the fixed Proposer–Verifier–Proposer–Verifier order and invokes the Judge conditionally. Each turn receives the relevant candidate and feedback as its context. Role ablations remove the corresponding agents. The shared OpenAI-compatible client records prompts, decoding settings and usage. Each run records the installed AutoGen versions in `configuration.json`.

### Semantic-unit control

The text surrogate repeats and truncates the original question to match the semantic-unit token budget. It retains the numbered-list layout, unit count, prompt position and shared instructions. Matching uses the generation model's tokenizer on the complete initial planning context, with an absolute tolerance of one token. A mismatch raises an error. Each prediction records the reference and control token counts and the tokenizer identifier.

The AMR-based GCN remains active in this condition, isolating the semantic-unit guidance supplied to the planner. The classifier is also retained when semantic units are removed.

### Strategy controls

Direct selection uses `Qwen2.5-32B-Instruct`, temperature 0 and a 128-token output budget. Its question-specific input is the original question text. The prompt is defined in `src/depplan/experiments/ablations/variants.py`.

The majority strategy is calculated from `data/classifier/train_manifest.jsonl`: 400 sequential, 225 parallel and 375 hierarchical questions. This selector always chooses sequential. The manifest hash and counts are saved with each majority-control run.

## Run the study

Prepare the official development sets and their question graphs using the commands in [reproduction](reproduction.md). The runner reads every question in each supplied file; it applies no implicit sampling limit.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
python -m depplan run-ablations \
  --input outputs/hotpot_graphs.jsonl outputs/musique_graphs.jsonl outputs/2wiki_graphs.jsonl \
  --checkpoint checkpoints/SPRING \
  --config configs/qa.json \
  --suite-config configs/ablations.json \
  --device cuda:0 \
  --output outputs/ablations
```

Set `OPENAI_API_KEY` if the service requires authentication. `--encoder` and `--tokenizer` accept local model directories. The tokenizer must match the served language model. CLI arguments override the corresponding suite settings. The API is used sequentially, one question and one condition at a time.

Validate the run grid without loading models or calling an API:

```bash
python -m depplan run-ablations \
  --input outputs/hotpot_graphs.jsonl \
  --checkpoint checkpoints/SPRING \
  --config configs/qa.json --suite-config configs/ablations.json \
  --output outputs/ablations --dry-run
```

To run selected conditions, add, for example, `--variants full no_semantic_units --seeds 42`. To continue a run with identical inputs and configuration, add `--resume`. Completed question records, including failures, are retained. Resume stops if an interrupted question has API call records but no final prediction record.

## Records and summaries

```text
outputs/ablations/
├── experiment.json
├── hotpot/
│   ├── full/seed_42/
│   │   ├── configuration.json
│   │   ├── calls.jsonl
│   │   ├── judge_calls.jsonl  # when independent semantic evaluation is configured
│   │   ├── predictions.jsonl
│   │   └── summary.json
│   └── ...
├── musique/
├── 2wiki/
└── reports/
    ├── per_seed.csv
    ├── mean_sample_sd.csv
    └── summary.json
```

`calls.jsonl` retains the actual system/user prompts, model outputs, request roles, attempt indices, usage, request latency and retry wait times. Attempts belonging to the same logical request share a `request_index`. `predictions.jsonl` stores question IDs, input hashes, selected strategies, supplied guidance, plans, completed execution steps, final answers, EM/F1 and component times. Reference answers are used only after generation for scoring. A failed question receives zero EM/F1 and remains in the denominator; completed calls and partially executed steps remain recorded. Missing API token usage is represented by `null`, with known token counts reported separately.

The online timer covers strategy selection (node encoding and GCN inference), semantic-unit construction, passage encoding, planning and execution, including failed API attempts and retry waits. It excludes model loading, cached AMR parsing and answer scoring. Total runtime adds the measured `amr_seconds` from each input to every seed; it is unavailable if any AMR timing is missing. Plan construction includes semantic-unit preparation, and plan execution includes passage encoding and retrieval. Remaining online coordination overhead is recorded separately. CUDA work is synchronized at timing boundaries. QA token totals include all API-reported input and output tokens, including usage returned with errors. LLM calls count physical attempts. Independent evaluator time/tokens remain separate. Component summaries retain failed attempts even when a later attempt succeeds.

The shared configuration allows at most three API attempts in total: the initial request and up to two retries for connection errors, timeouts, HTTP 408/429, or HTTP 5xx responses. Retries preserve the prompt, model, seed and decoding parameters, with waits of one and two seconds. SDK-level retries are disabled. Authentication errors, malformed model outputs and invalid plans do not trigger API retries. The configured limit is recorded for every condition.

`plan_repair_attempts` is fixed at zero for the full model and all seven variants. Plan validation operates within each variant's configured agent workflow, without additional generation calls.

Summaries are generated at the end of the suite. They can also be recomputed independently:

```bash
python -m depplan aggregate-ablations \
  --input outputs/ablations --output outputs/ablations/reports
```

`per_seed.csv` includes `variant`, `dataset`, `seed`, `questions`, integer `em_correct`/`acc_correct`, EM/F1/Acc percentages, evaluation coverage, failures, mean online/AMR/total/component times, executed steps, QA calls/tokens and separate evaluator costs. EM and Acc use the full question count. Acc stays unavailable if an evaluator decision is missing. The aggregate file calculates means and sample standard deviations (`ddof=1`) from unrounded run metrics, retaining missing values across seeds. A single seed has no sample SD and is exported with an empty SD field. Aggregation checks run completeness, seed uniqueness, question cohorts and shared settings before computing summaries from saved predictions.

## Prompts

Planning and answer prompts are in `src/depplan/prompts.py`. All condition-specific changes are defined in `src/depplan/experiments/ablations/variants.py`; the context renderer is shared. Export the complete prompt specification with:

```bash
python scripts/export_ablation_prompts.py --output outputs/ablation_prompts.json
```
