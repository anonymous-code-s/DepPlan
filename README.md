# DepPlan

DepPlan is a framework for multi-hop question answering that combines semantic guidance with dependency-aware planning. It derives local semantic units from a question's AMR graph and uses a graph classifier to select a planning strategy. AutoGen Proposer, Verifier and Judge agents construct a plan whose dependencies guide retrieval and intermediate-answer generation.

This repository accompanies our paper on DepPlan. We provide the QA pipeline, component ablations, baseline implementations, trained strategy classifiers, prompts and evaluation tools. Per-seed results and summaries are organized under [results/](results/README.md).

## Installation

Python 3.10 or later is required. Install a PyTorch build compatible with your CPU or CUDA environment, then install DepPlan:

```bash
python -m pip install -e .
python -m depplan --help
```

For AMR parsing and Smatch evaluation, install `python -m pip install -e '.[parsing]'`. [requirements.txt](requirements.txt) pins the dependency versions, including AutoGen AgentChat/Core 0.7.4.

The QA pipeline uses SPRING for AMR parsing, BGE-base-en-v1.5 for embeddings, and Qwen2.5-32B-Instruct through an OpenAI-compatible service. Prepare these external models using the [setup guide](docs/reproduction.md) and [Qwen serving instructions](docs/serving.md).

## Data preparation and quick start

Prepare an official development file and its question AMRs by following the [data preparation instructions](docs/reproduction.md#5-run-multi-hop-qa). The resulting JSONL contains each question, candidate passages, reference answers and AMR triples.

Run the QA pipeline against the Qwen service:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
python -m depplan run-qa --input outputs/hotpot_graphs.jsonl \
  --checkpoint checkpoints/SPRING --config configs/qa.json \
  --seed 42 --output outputs/hotpot_seed42
```

Set `OPENAI_API_KEY` if your service requires authentication. The runner saves plans, answers, EM/F1, model calls, token usage and timing in the output directory. [configs/qa.json](configs/qa.json) defines the shared QA settings; the experiments use decoding seeds 42, 123 and 2026 with separate output directories.

Semantic accuracy uses an independent GPT-4.1-mini evaluator configured through `JUDGE_MODEL`, `JUDGE_BASE_URL` and `JUDGE_API_KEY`. See [answer evaluation](docs/reproduction.md#5-run-multi-hop-qa) for configuration and saved-answer scoring commands.

## Strategy checkpoints

The five GCN checkpoints predict sequential, parallel or hierarchical planning strategies from question graphs. Choose the checkpoint corresponding to the graph source:

| Structure source | Checkpoint directory |
|---|---|
| SPRING | [checkpoints/SPRING](checkpoints/SPRING) |
| CAIL | [checkpoints/cail](checkpoints/cail) |
| T5-base | [checkpoints/t5_base](checkpoints/t5_base) |
| Graphene | [checkpoints/graphene](checkpoints/graphene) |
| GPT-4o-mini AMR | [checkpoints/GPT-4o-mini](checkpoints/GPT-4o-mini) |

Each directory contains Safetensors weights, a configuration file and training history. The [checkpoint guide](docs/checkpoints.md) describes the architecture, training settings and graph sources. [Classification results](results/classification/README.md) provide scores and evaluation commands, while the [data guide](docs/data.md) explains the sample-ID and silver-label manifests.

## Experiments and results

The following guides describe each experiment's inputs, configuration and commands. The [results guide](results/README.md) indexes the saved results and explains their formats and aggregation.

| Experiment | Guide and results |
|---|---|
| Main QA comparison | [QA workflow](docs/reproduction.md#5-run-multi-hop-qa), [per-seed results](results/main_results/README.md) |
| Component ablations | [Seven ablation variants](docs/ablations.md), [per-seed results](results/ablations/README.md) |
| Baseline reproduction | [Fourteen workflows, settings and prompts](baselines/README.md), [published-score comparison](baselines/results/published_comparison.csv) |
| Parsing quality | [Smatch results and scoring](results/parsing_quality/README.md) |
| Sensitivity and sub-question analysis | [Coverage thresholds, interaction rounds, retrieval top-k, sub-question quality and length](docs/experiments.md) |
| Plan quality and statistical comparisons | [Dependency metrics, intermediate-answer F1 and paired bootstrap](docs/plan_statistics.md) |
| Computational cost | [Strategy selection](docs/selector_cost.md), [silver-label generation and classifier training](docs/experiments.md) |
| Answer evaluation | [Selected answers, judgments and agreement summaries](data/annotations/answer_review/README.md) |

## Prompts and semantic units

The [agent prompts](src/depplan/prompts.py) define proposal, verification, plan selection and answer generation. Additional prompts cover [silver strategy labels](prompts/silver_strategy.txt), [direct strategy selection](src/depplan/experiments/ablations/variants.py) and [GPT-4o-mini AMR generation](prompts/gpt4o_mini_amr.txt).

The [AMR template library](src/depplan/semantic_units/amr_templates.json) and [linearizer](src/depplan/semantic_units/linearization.py) construct semantic units from graph relations and their local contexts. The [rendering guide](docs/semantic_units.md) describes core arguments, inverse relations and entity names.

Core arguments follow the predicate frame and argument index defined by the
[AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md).
