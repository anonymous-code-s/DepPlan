# DeepRAG and R-Search: implementations and settings

Our implementations are in `baselines/methods/learned.py`;
their JSON configs contain a `parameter_sources` record for each operational field.
`code` and `paper` identify published values; `choice` identifies a default we select;
`adaptation` identifies a changed model, prompt, or execution environment.

| Method | Primary source | Available artifact | Executed variant |
| --- | --- | --- | --- |
| DeepRAG | Guan et al., [DeepRAG: Thinking to Retrieve Step by Step for Large Language Models](https://arxiv.org/abs/2502.01142), ICLR 2026; [official repository](https://github.com/gxy-gxy/DeepRAG/tree/1665780b7b6317375d56e71c3ddc52d6f706c380) | Public, ungated [xinyan233333/DeepRAG-8b](https://huggingface.co/xinyan233333/DeepRAG-8b/tree/ece5f36cb1c4b279aad0503b5b6295b73869b677), revision `ece5f36cb1c4b279aad0503b5b6295b73869b677`; four safetensors shards and a Llama tokenizer/config | The released Llama-3-8B checkpoint using `SRAGSFTV2` continuation and learned retrieval decisions, with BM25 restricted to the shared candidate passages |
| R-Search | Shen and Shi, [Reasoning-guided structured planning and search in large language models](https://doi.org/10.1016/j.ipm.2026.104732), IPM 63 (2026), 104732; [preprint](https://arxiv.org/abs/2506.08352); [ReasoningSearch repository](https://github.com/wentao0429/ReasoningSearch/tree/07f619128223872219af1f14c395038963393df1) | Public training and NL-DAG execution code, commit `07f619128223872219af1f14c395038963393df1`; no trained checkpoint published | Default: Qwen2.5-32B-Instruct NL-DAG adaptation. Optional: an externally supplied trained R-Search endpoint, still using candidate-set retrieval |

We implement R-Search from `wentao0429/ReasoningSearch`. Its output has four
sections: `think`, `search`, `result`, and `answer`. The search plan uses the
original `Nodes:` / `Edges:` natural-language syntax.

## DeepRAG

We use `xinyan233333/DeepRAG-8b`, linked from the
[official repository](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/README.md),
at the revision listed above. The released weights are used without additional
training. Their calibration stage is unspecified, recorded as
`checkpoint_stage="unspecified_by_release"` in the configuration.

| Setting | Adapter value | Provenance / rationale |
| --- | --- | --- |
| Checkpoint | `xinyan233333/DeepRAG-8b` at the immutable revision above | Released artifact; HF config identifies `LlamaForCausalLM`, 32 layers, 8B-family vocabulary |
| Inference | Raw `/completions` with the checkpoint's chat template and an unfinished assistant message | [SRAGSFTV2](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/src/algorithms/sragsftv2.py), `_create_chat_prompt` |
| Control | `Follow up:`, `Intermediate answer:`, `Let`, `So the final`; final `<answer long>` / `<answer short>` | Same source; `Let` requests retrieval, `So the final` requests final synthesis. Stop strings must be returned by the serving endpoint or recovered from its `stop_reason` / `matched_stop` |
| Temperature / top-p | `0.0` / `0.95` | [inference command](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/scripts/inference/run.sh) and [remote generator](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/src/algorithms/basic.py) |
| Generation length | Up to 4,096 new tokens per call, capped to remaining context | Inference command and `BasicGenerator.remote_generate` |
| Context limit | 8,192 tokens | [released checkpoint config](https://huggingface.co/xinyan233333/DeepRAG-8b/blob/ece5f36cb1c4b279aad0503b5b6295b73869b677/config.json) |
| Maximum reasoning depth | HotpotQA: 6; other datasets: 10 | `SRAGSFTV2.__init__`; dataset comes from the runner context, or explicit config `dataset` |
| Empty follow-up retries | 10 | `_generate_follow_up`; exhaustion raises an explicit protocol error instead of advancing with an empty subquestion |
| Retrieval | BM25, top 3 | [argument defaults](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/src/main.py); local adaptation restricts the corpus to candidate passages |
| Prompt | Released instruction and history format | [instruction.txt](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/ICL-prompt/instruction.txt). The original Wikipedia narrative remains for the trained model; traces record the actual candidate retrieval |

The checkpoint's `generation_config.json` stores temperature 0.6 / top-p 0.9.
The executable inference script and API client override those values, so the
adapter follows the evaluation code's 0.0 / 0.95. Inference advances one learned
decision path. The binary tree search described in the original paper constructs
training data.

Serve the pinned weights under the configured model name on a completion-capable
endpoint. A serving alias can be supplied through the runner's model option;
that alias must still resolve to the intended DeepRAG weights. The model is not
automatically downloaded or deployed by this adapter. Tokenizer revision is
separately recorded in the config. Use the completion endpoint to preserve the
prompt's literal control strings and avoid applying a second chat template.

## R-Search

The public [training script](https://github.com/wentao0429/ReasoningSearch/blob/07f619128223872219af1f14c395038963393df1/train.sh)
initializes `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. The repository publishes no trained checkpoint, so the adapter runs on
the shared served model.

| Setting | Adapter value | Provenance / rationale |
| --- | --- | --- |
| Default model | Served `Qwen2.5-32B-Instruct`; tokenizer `Qwen/Qwen2.5-32B-Instruct` | Qwen adaptation of the public workflow; uses no GRPO/LoRA checkpoint |
| Output and execution | NL-DAG plan; topology validation; topological execution; per-node evidence; answer synthesis | [paper Sec. 3.3](https://arxiv.org/abs/2506.08352) and [search_manager.py](https://github.com/wentao0429/ReasoningSearch/blob/07f619128223872219af1f14c395038963393df1/RL/search_manager.py) |
| Maximum nodes | 6 | `DAGParser`, `GenerationManager`, and `check_dag` defaults. Original training prompt says 8; the adaptation prompt is aligned to the executable limit of 6 |
| Query length | At most 10 whitespace-separated words | Original training prompt; explicit parser enforcement is a local implementation choice |
| Planning temperature / top-p | `0.6` / `0.9` | Released `GenerationManager.generate_with_search` planning settings; DepPlan planning uses temperature 0.7 |
| Final-answer temperature | `0` | Shared evaluation protocol; replaces the released source's temperature 0.6 for this phase |
| Generation length | Plan 1,000; answer 2,000 tokens | `train.sh` sets 2,000; the generator halves this budget for phase one. We use these rollout budgets for inference |
| Retrieved passages per node | 3 | Public `SEARCH_TOPK=3`; paper Sec. 4.1 reports top 2 during training. We use the code's inference default |
| Tool routing | General, News, Academy all search the common dense candidate collection | Candidate-set adaptation replacing Google/GNews/arXiv. Tool labels remain in the plan and trace |
| Invalid adaptation plan | Explicit failure without an LLM repair call | Shared evaluation protocol; invalid graphs terminate the question |
| Invalid trained plan | Explicit failure | No extra chat prompting in trained-checkpoint mode |
| Independent nodes | Recorded as topological frontiers; local synchronous retrieval executes them in source order | Preserves dependency ordering; runtime reflects sequential candidate retrieval |

Node queries execute as written. The public executor sorts a fixed NL-DAG and
does not rewrite downstream queries with predecessor answers. The adapter keeps
that behavior. It rejects cycles, self-loops, undeclared endpoints, duplicate
labels, invalid tool names, and node/query limit violations. It discards any
model-generated tail after the search plan and fills `<result>` with passages
returned by the retriever.

To use separately obtained trained weights, select a config with
`inference_mode="trained_checkpoint"`, an explicit `trained_checkpoint` artifact
identity, the actual served `model`, its matching `tokenizer`, and
`endpoint_api="completion"`. The adapter then uses the released training prompt and
`safe_chat_template` message arrangement: stage one stops at
`</search>`, execution supplies `<result>`, and stage two stops at `</answer>`.
The model identity field does not load weights; record the
served checkpoint/revision in the run configuration. The candidate-set change
still applies to this optional mode.

