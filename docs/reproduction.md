# Reproduction

Run commands from the repository root after `python -m pip install -e .`. The default device is CPU. Supply `--device cuda:0` for classifier inference or training on a GPU; `--encoder` accepts an existing BGE directory.

## 1. Recompute classification metrics

```bash
python -m depplan score-classification --input results/classification/gpt4o_mini_amr.jsonl
```

This recomputes metrics from the saved prediction files and runs locally on the files in this repository.

## 2. Restore question texts and prepare graphs

Download the official training files for HotpotQA, MuSiQue and 2WikiMultiHopQA. Restore the evaluation questions in manifest order:

```bash
python scripts/restore_classifier_questions.py \
  --manifest data/classifier/evaluation_manifest.jsonl \
  --hotpot-train /path/to/hotpot_train_v1.1.json \
  --musique-train /path/to/musique_ans_v1.0_train.jsonl \
  --wiki-train /path/to/2WikiMultiHopQA/train.json \
  --output outputs/evaluation_questions.jsonl
```

Prepare AMRs with a locally installed compatible parser. For the reference checkpoint, use a SPRING parser directory supported by amrlib (`parse_spring`). The directory must contain the parser weights and configuration; `checkpoints/SPRING` contains the strategy GCN, not the AMR parser. For example:

```bash
python -m depplan parse-amr --input outputs/evaluation_questions.jsonl \
  --model-dir /path/to/amrlib-spring \
  --structure-source spring --device cuda:0 --batch-size 4 \
  --output outputs/reference_evaluation_graphs.jsonl
python -m depplan evaluate-classifier \
  --checkpoint checkpoints/SPRING \
  --input outputs/reference_evaluation_graphs.jsonl \
  --output outputs/reference_predictions.jsonl --device cpu
```

The reference checkpoint uses SPRING AMR graphs, identified by `structure_source: spring`. For other classifiers, prepare graphs with the matching source described in [model provenance](checkpoints.md). Graph records require `example_id`, `dataset`, `amr_triples`, `structure_source`, `silver_strategy` and `status`. GPT-4o-mini AMR generation uses the prompt in `prompts/gpt4o_mini_amr.txt`.

The classifier takes concept embeddings and graph adjacency as inputs and uses silver labels for scoring. Inference writes per-question predictions and aggregate metrics. Step 1 recomputes the saved results; the commands above evaluate newly generated graphs.

## 3. Train a new classifier

Restore questions using `data/classifier/train_manifest.jsonl` instead of the evaluation manifest, then prepare source-matched graphs. For example, after creating CAIL training graph records:

```bash
python -m depplan train-classifier \
  --input outputs/cail_training_graphs.jsonl \
  --output outputs/cail_retrained --seed 2026 --device cuda:0
```

Training defaults to seed 2026, as specified by `retraining_seed` in the checkpoint configuration. It uses frozen BGE features, 30 epochs, batch size 16 and AdamW. The output directory must be empty. Keep failed graph records with `status: graph_failed`; the trainer excludes them and records the effective training count. New runs save their actual seed, input and source hashes, feature digest, final weight digest and history in `training_provenance.json`, `config.json` and Safetensors metadata. Pass `--encoder-revision` to pin a remote encoder revision; its requested value is recorded separately. The recorded seed describes the new training run; `retraining_seed` is a default for retraining, not metadata for the distributed weights. See [checkpoint configuration](checkpoints.md) for the architecture and optimizer settings.

## 4. Inspect semantic units

```bash
python -m depplan linearize \
  --input outputs/reference_evaluation_graphs.jsonl \
  --output outputs/semantic_units.jsonl
```

The template JSON provides fixed relation realizations. The linearizer additionally handles entity names, numbered `:op` edges, negation and unknown relations. Each semantic unit contains one central relation and its variable-sharing one-hop context. Core arguments retain the predicate sense and ARG index, including inverse edges; see [semantic-unit rendering](semantic_units.md).

## 5. Run multi-hop QA

The QA runner accepts one JSON record per question with `dataset` (`hotpot`, `musique`, or `2wiki`), `example_id`, `question`, `passages`, `answers`, `amr_triples` and `structure_source`. Candidate passage order and boundaries are preserved.

Prepare a downloaded official dataset file and parse its questions:

```bash
python scripts/prepare_qa.py --dataset hotpot \
  --input /path/to/hotpot_dev_distractor_v1.json \
  --output outputs/hotpot_questions.jsonl
python -m depplan parse-amr --input outputs/hotpot_questions.jsonl \
  --model-dir /path/to/amrlib-spring \
  --structure-source spring --device cuda:0 --batch-size 4 \
  --output outputs/hotpot_graphs.jsonl
```

The parser processes four questions per call by default, with a smaller final batch. Measured batch parsing and decoding time is divided by the actual batch size and saved as `amr_seconds` for each question. Batch IDs, batch duration and configured/actual batch sizes are retained. Failed parsing attempts also retain their cost. Model loading and output serialization are excluded. Reuse this graph file across the three decoding seeds.

Start your OpenAI-compatible Qwen2.5-32B-Instruct service with the [FP16/four-GPU launcher](serving.md), then run:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
python -m depplan run-qa --input outputs/hotpot_graphs.jsonl \
  --checkpoint checkpoints/SPRING --seed 2026 \
  --output outputs/hotpot_seed2026
```

If the service requires authentication, set `OPENAI_API_KEY` in the environment. Use separate output directories for seeds 42, 123 and 2026. `configs/qa.json` documents the default settings: two proposal–verification rounds, conditional final candidate selection, planning temperature 0.7, greedy answer decoding, and BGE cosine retrieval with top-k 5.

To evaluate semantic accuracy with GPT-4.1-mini, configure its model or deployment ID, endpoint and API key:

```bash
export JUDGE_MODEL='<evaluator-model-or-deployment-id>'
export JUDGE_BASE_URL='<evaluator-base-url>'
export JUDGE_API_KEY='<evaluator-api-key>'
python -m depplan run-qa --input outputs/hotpot_graphs.jsonl \
  --checkpoint checkpoints/SPRING --seed 2026 \
  --output outputs/hotpot_seed2026_scored
```

Equivalent CLI options are `--judge-model`, `--judge-base-url` and `--judge-api-key`. The evaluator uses its own client, the fixed semantic-judging prompt and temperature 0. Without a model setting, `semantic_correct` and aggregate Acc remain `null`; `--no-semantic-accuracy` explicitly disables evaluation. A configured model requires an independent endpoint. A failed QA attempt scores zero. An invalid or failed evaluator response is recorded as missing; aggregate Acc remains unavailable until every question is scored, rather than dropping such questions from the denominator. Evaluator calls, tokens and elapsed time are stored separately in `judge_calls.jsonl` and `judge_*` fields.

Score already generated answers without repeating QA:

```bash
python -m depplan score-qa \
  --input outputs/hotpot_seed2026/predictions.jsonl \
  --output outputs/hotpot_seed2026_evaluation
```

This reads `JUDGE_*` or the same CLI options, preserves the input file and writes new scored predictions and summaries. Use `--resume`; add `--retry-failed` to explicitly retry evaluator failures while accumulating their costs.

AutoGen AgentChat 0.7.4 runs the planning roles through custom `BaseChatAgent` agents. The coordinator passes candidate plans and verification feedback between roles; the Judge is called only when final candidates differ. Both `run-qa` and the ablation suite use this implementation. Model requests use the shared logged OpenAI-compatible client.

`calls.jsonl` records each request's role, prompts, output, attempt index, reported tokens, request latency and retry wait, together with response/model identifiers when the server returns them. The shared configuration permits up to three identical API attempts for transient connection errors, timeouts, rate limits and server errors. SDK-level retries are disabled; the two retry waits are one and two seconds. There are no extra plan-repair calls or question reruns. `predictions.jsonl` records plans, executed steps, answers, EM/F1, semantic correctness and timing. API/plan failures stay in the denominator with zero EM/F1. Failed attempts retain elapsed time and reported usage; unavailable token usage is `null`.

The online timer includes node encoding and GCN inference, semantic-unit construction, passage encoding, planning, execution and retry waits. It excludes AMR preprocessing, model loading and answer scoring. `total_seconds` automatically adds the saved `amr_seconds` to online latency in each seed. Missing AMR measurements leave total time `null`. Plan-construction time includes semantic-unit preparation; plan-execution time includes passage encoding, retrieval and answer generation. Remaining coordination time is explicitly recorded as overhead.

The [component-ablation suite](ablations.md) uses this same pipeline for Full and all seven variants. It reads the same prepared graph files and exports per-seed integer correct counts, EM/F1, usage and sample-standard-deviation summaries. `run-qa` and the `full` ablation invoke the same implementation.

The [supplementary experiment guide](experiments.md) covers Coverage thresholds, interaction rounds, retrieval cutoffs, sub-question scoring and offline preparation cost. The round runner adds measured per-question AMR times to online latency when those measurements are supplied.

## 6. Score AMR parsing

```bash
python scripts/score_smatch.py \
  --predicted /path/to/predicted.amr --gold /path/to/gold.amr
```

Both files must contain aligned PENMAN graphs, separated by blank lines, in the same question order. The command reports corpus-level Smatch with a fixed matching seed. To reproduce the aggregation in our paper, score each dataset separately and take the sample-weighted mean with weights 0.3, 0.4 and 0.3. See [data availability](data.md#evaluation-inputs) for the required files.
