# Agent baselines: implementations and settings

We implement these workflows with Qwen2.5-32B-Instruct, BGE-base-en-v1.5 and
the supplied candidate passages, retaining the control mechanisms described below. Each configured value is tagged `paper`, `code`, `choice`, or
`adaptation` in `baselines/configs/methods/{coa,gear,belle,per}.json`.

## Sources

| Method | Primary source | Implementation |
|---|---|---|
| CoA | [NeurIPS 2024 paper](https://papers.nips.cc/paper/2024/file/ee71a4b14ec26710b39ee6be113d7750-Paper-Conference.pdf): Algorithm 1, Algorithm 2, Appendix B, Table 11, window-size analysis | [Google Research project explanation](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/) confirms worker chain and final-unit manager. |
| GeAR | [ACL Findings 2025 paper](https://aclanthology.org/2025.findings-acl.624.pdf): sections 3–5, Algorithm 1, section 6.2, Appendix B.1–B.2, Appendix K | [Official project page](https://gear-rag.github.io/) publishes the diverse beam pseudocode. |
| BELLE | [ACL 2025 paper](https://aclanthology.org/2025.acl-long.211.pdf): sections 4.1–4.3, Appendix A.1, Appendix C, Figure 9, Table 10 | — |
| PER-QA | [ACL 2025 paper](https://aclanthology.org/2025.acl-long.1142.pdf): section 3.1, Appendix D.1, Tables 21–23 | [Official PER-PSE repository](https://github.com/GenIRAG/PER-PSE), especially [method.py](https://github.com/GenIRAG/PER-PSE/blob/main/method.py), [module.py](https://github.com/GenIRAG/PER-PSE/blob/main/module.py), [prompt.py](https://github.com/GenIRAG/PER-PSE/blob/main/prompt.py), and `pipeline.py`. |

The PER paper names GPT-4o as its planner, while its released code uses
GPT-4o-mini. We use Qwen2.5-32B-Instruct for the shared-model comparison.

## CoA

`methods/agents.py:coa` packs all candidate passages into sentence-based chunks
in their input order. Worker i receives its source chunk and only the preceding communication
unit. The manager receives the final unit and the question. All supplied passages
are processed in the worker chain.

| Parameter | Value | Provenance |
|---|---|---|
| Worker window | 8192 tokens | Paper's default 8k agent window; interpreted as 8192. |
| Temperature | 0 | Appendix B, all non-self-consistency experiments. |
| Worker/manager maximum output | 1024 tokens | Appendix B's non-Gemini-Ultra setting; Ultra used 2048. |
| Chunking | Sentence accumulation, source order | Algorithm 2. |
| Carry/output reservations | 1024 + 1024 tokens | API-context adaptation: reserve previous worker output and current completion, then subtract question/instruction tokens and 64 chat-framing tokens. |
| Oversized sentence | Split by tokenizer token boundaries | Chosen completion rule for sentences exceeding a whole chunk; no source text is discarded. |

The tokenizer is the shared Qwen tokenizer. A sample with fewer than one source
chunk's tokens uses a single worker.

## GeAR

`methods/gear.py` builds a question-independent triple index from every candidate
passage. Every triple retains its passage identity. SyncGE retrieves initial
passages, extracts proximal triples, links them to the triple index, and expands
paths along shared subject/object entities. Sequence scores are embedding cosine
similarities. Beam candidates receive the Algorithm 1 sibling-rank multiplier
`exp(-min(rank, gamma) / gamma)`, and triples already in the previous beam set are
excluded. Beams are flattened breadth-first, mapped back to passages, and fused
with initial retrieval using RRF. Each round updates gist memory; the reasoner
checks answerability, and the rewriter targets missing information. After stopping,
each gist triple queries both indexes; these rankings and all retrieval rounds
are fused before final QA.

| Parameter | Value | Provenance |
|---|---|---|
| Initial passages | 10 per iteration | Appendix B.2, multistep configuration. |
| Beam width / length | 10 / 2 | Appendix B.2. |
| Neighbors per beam | At most 100 | Appendix B.2; candidates above this bound are ranked by shared dense query similarity, an implementation choice because neighbor ordering is unspecified. |
| Diversity gamma | 20 | Appendix B.2: twice beam width. |
| Iterations | HotpotQA 2, 2Wiki 2, MuSiQue 4 | Appendix B.1. |
| Temperature | 0 | Section 6.2. |
| Base/triple retrieval | Hybrid BM25 + dense RRF | Published hybrid mechanism; BGE replaces the original all-mpnet-base-v2. |
| RRF constant | 60 | Chosen default; the paper specifies RRF without its constant. |
| Final QA passages | 5 | Table 3 caption; distinct from the ten passages retrieved initially in each iteration. |
| Maximum completion | 1024 tokens | Chosen default; the paper specifies no generation cap. |

Final QA follows Appendix K.3's one-example format and six-word answer
instruction. We provide a short illustrative example of the required format.

When graph expansion reaches no new triples, we retain the last nonempty beam.
Exact repeated gist triples are deduplicated; gist memory has no size cap. These
choices adapt the workflow to a fixed candidate collection. Graph extraction is logged as `gear_index_*` with
`stage=preprocessing`; its model calls remain in traces and token accounting.
We construct the index from the supplied candidate passages.

## BELLE

`methods/belle.py` retains all five operators: CoT, single-step, iterative-step,
sub-step, and adaptive-step. A four-type classifier precedes debate. Every round
executes **affirmative → negative → fast → slow → judge**. Opposing debaters
receive their own history and preceding second-level views. The fast debater
assesses the current sides with prior fast views; the slow debater receives full
discussion history and previous slow conclusions. Hard judging can continue the
debate; the final allowed round uses soft judging to extract a plan from history.

Plans execute in order. Sub-step creates explicit answer-tag dependencies. A
following solver runs on each instantiated subquestion, preserving intermediate
answer flow; a final sub-step uses single retrieval per subquestion. Iterative-step
alternates retrieval and reasoning until an answer or its configured bound.
Adaptive-step chooses no retrieval, one retrieval, or iterative retrieval.

| Parameter | Value | Provenance |
|---|---|---|
| Debate rounds | At most 3 | Table 10. |
| Speaking order and five roles | As above | Appendix C.2 and Table 10. |
| Question types | Inference, Comparison, Temporal, Null | Section 4.1 / Figure 9. |
| Retrieved passages | 5 | Chosen within Appendix A.1's 3–10 range; the paper gives a range rather than per-dataset counts. |
| Retriever | BM25 | Appendix A.1; applied to the supplied candidate collection. |
| Temperature / output cap | 0 / 1024 tokens | Chosen defaults; the paper does not report these values. |
| Iterative operator cap | 6 retrieval rounds | Chosen finite execution bound. |
| Plan/subquestion caps | 8 operators / 12 nodes | Chosen validation bounds; oversized plans return an error. |

The classifier includes four type examples (Figure 9), and the CoT operator has
four demonstrations, one per type (Appendix A.1). Their compact symbolic content
is newly authored and is recorded as a prompt adaptation.

We implement the adaptive operator as a prompted A/B/C complexity decision with
the shared LLM. BELLE describes operators as prompting interfaces and cites
Adaptive-RAG; our implementation uses a prompted router. The
paper does not specify the full operator subprompts, so the concrete subprompts and JSON
plan schema are implementation choices. Composed sub-step planning uses a tagged
DAG; the schema is this implementation's serialization.

## PER-QA-RAG

`methods/agents.py:per` first classifies Bridge versus Comparison and uses a
type-specific planner. Its `<A#>` references form a validated DAG. Independent
nodes may appear in either order; execution is topological. Each dependent query
is instantiated with the actual parent answer **before** retrieval. HyDE is used
only for query construction; hypothetical text never becomes answer evidence.
Executor outputs form a context graph with the plan's edges. Only a complete
supported graph reaches the reasoner, as in official `aggregate` behavior.

| Parameter | Value | Provenance |
|---|---|---|
| Passages per atomic query | 5 | Appendix D.1. |
| Temperature / maximum output | 0 / 1024 | Official `method.py:per_qa_rag`; planner's live `module.py` configuration also uses these values. |
| HyDE | Enabled | Official `method.py:per_qa_rag`. |
| Query repetitions before HyDE passage | 5 | Official `module.py:PERQARAGExecutor.get_hyde_query`. |
| Answers retained per node | 1 | Official `PERQARAGExecutor.execute(max_answer_num=1)`; list responses use their first answer. |
| Rewriter / relevance expert | Disabled | Official `method.py:per_qa_rag`; those optional branches are not invoked. |
| Maximum plan nodes | 16 | Chosen validation bound, allowing more nodes than the 2–4-hop datasets require. |

PER-DP's 5/15 retry counts and its gold-answer checks belong to dataset
preprocessing and are excluded from inference. The shared model
replaces the paper's GPT-4o planner and Llama3.1-8B executor/reasoner. The prompt
module retains the official released few-shot classifier, both planner prompts,
retrieval executor, and type-specific reasoner prompts. Doubled template braces
are rendered as ordinary JSON braces; the GenIRAG MIT notice is included in
`prompts/per.py`. The official executor's string filter is retained (including its
broad rejection of words such as `unknown` and `evidence`); an explicit false
support flag is also honored if a model returns one. An incomplete context graph
returns a failure.

## Shared choices

Seeds are 42, 123, and 2026. All methods have a 256-execution-step failure guard;
method-specific loops and their source settings remain separate from this guard.
Step units are CoA workers, GeAR retrieval rounds, PER plan nodes, and BELLE
primitive solver iterations (CoT, single retrieval, or one iterative round).
Model-call counts are recorded separately, including planning, debate, final
generation, and GeAR graph preprocessing.
Prompts use the common JSON/plain-text interfaces and concise answer format.
CoA, GeAR, and BELLE prompt serializations are adaptations. PER preserves the
rendered official prompts while adapting the user payload to this runner's API.

