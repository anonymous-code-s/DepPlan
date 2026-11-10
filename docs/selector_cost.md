# Online strategy-selection cost

We compare the selection cost of the CPU GCN, direct Qwen and the
training-majority selector with `python -m depplan.experiments.selector_cost`. The default cohort contains exactly 200 development
questions from each of HotpotQA, MuSiQue and 2WikiQA. All three selectors run on
the same ordered 600 questions with decoding seeds 42, 123 and 2026.

Prepare this cohort explicitly from the development sets, then prepare its AMRs
with the parser associated with the selected classifier checkpoint. Each JSONL
record requires `dataset` (`hotpot`, `musique`, `2wiki`), `example_id`, `question`,
`structure_source`, and `amr_triples`. `status` defaults to `success`; an explicit
`graph_failed` record remains in the cohort. `source_split`, when supplied, must
be `dev`, `development` or `validation`. The runner checks the dataset quotas,
unique IDs, source-matched graphs and absence of overlap with the training
manifest. The GCN receives question graphs; direct Qwen receives question text;
the majority selector uses the training-label distribution.

```bash
python -m depplan.experiments.selector_cost \
  --input outputs/selector_600_graphs.jsonl \
  --checkpoint checkpoints/SPRING \
  --training-manifest data/classifier/train_manifest.jsonl \
  --output outputs/selector_cost --dry-run

python -m depplan.experiments.selector_cost \
  --input outputs/selector_600_graphs.jsonl \
  --checkpoint checkpoints/SPRING \
  --training-manifest data/classifier/train_manifest.jsonl \
  --encoder /path/to/bge-base-en-v1.5 \
  --base-url http://127.0.0.1:8000/v1 \
  --output outputs/selector_cost
```

The dry run validates the cohort and configuration before inference. Use the serving setup in
[serving.md](serving.md) for the Qwen endpoint. `--questions-per-dataset` can be
set to change the cohort size; the selected value is recorded in the run
configuration. All three decoding seeds are required. The output
directory must be empty, preserving earlier measurements and interrupted logs.

The GCN and BGE models load before timing starts. For each question in every
seed, the timed GCN operation includes fresh CPU node encoding and classifier
inference. This experiment runs the classifier on the CPU. Direct Qwen uses the existing question-only strategy
prompt, temperature 0 and a default 128-token output limit. It makes one logical
selection request per question. The shared client permits at most three
identical physical attempts for transient failures and records retry waits and
API-reported usage. Invalid strategy outputs fail the question without repair
calls. Majority selection uses the unique most frequent strategy in the frozen
training manifest and makes no LLM requests.

Latency measures selection and response validation. It excludes model loading,
AMR parsing, answer scoring and writing the per-question result. Failed
selections retain their measured time and all reported API usage. An AMR failure
affects the GCN selection; the question-only and constant selectors can still
run on that same question. Their records share the same cohort and input hash.
Failed selections remain in the per-question denominator. Logical requests
and physical LLM calls are reported separately. Missing token usage stays
`null`, with known token totals retained separately.

The root `configuration.json` records the exact cohort, input-file checksums,
effective settings, code hash, checkpoint and classifier-config hashes, training
manifest hash, prompt hash, CPU thread count and endpoint without credentials.
`cohort.jsonl` retains the actual question/AMR inputs. Every selector/seed
directory contains its configuration, per-question `predictions.jsonl` and
summary; direct-Qwen runs also retain shared-client `calls.jsonl` with all
attempts. Each prediction records its input, effective-configuration, code and
checkpoint hashes. Its `method` field equals `selector` (`gcn`, `direct_qwen` or
`majority`), allowing the shared run loader to retain distinct matched methods.
A configuration digest is computed from its JSON object
before adding the digest's own `configuration_sha256` field.

`per_seed.csv` reports each selector's per-question latency, calls, input/output/
total tokens, known usage and failures, both over all 600 questions and for each
dataset. `mean_sample_sd.csv` computes means and sample standard deviations
(`ddof=1`) across the three unrounded per-seed means. Use the retained
per-question records for [paired statistical comparisons](plan_statistics.md#nested-paired-bootstrap).
