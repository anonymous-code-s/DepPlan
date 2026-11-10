# Classic baselines: implementations and settings

We implement these
workflows in `baselines/methods/classic.py` and the eight corresponding JSON
files in `baselines/configs/methods/`. Each configuration has a
`parameter_sources` entry for every execution field, identifying `paper`,
`code`, `choice`, or `adaptation`.

## Shared execution configuration

We use `Qwen2.5-32B-Instruct`; the endpoint must serve that model.
The shared seeds are 42, 123, and 2026. Retrieval operates on the collection
provided by the runner, with BGE dense embeddings where required. IRCoT applies
BM25 to this collection; other retrieval baselines use its dense ranking. These
are corpus/backbone adaptations, distinct from the original systems below.
Direct prompting, CoT, and Plan-and-Solve make no retrieval call and do not
include candidate passages in the prompt. Their retrieval configuration fields
are unused.

Default temperature is zero. The code records the method's actual prompts,
retrieval queries, selected passage identifiers, intermediate outputs, and stop
condition through the runner. `max_steps` counts controller/hop turns, not LLM
calls. Final readers, keyword extraction, and batch
grounding are additional calls. Prompt strings are visible in
`baselines/prompts/`; the [prompt guide](../prompts/README.md) lists the active
functions. Outputs are written to the selected run directory.

## Direct prompting

**Source:** Brown et al., *Language Models are Few-Shot Learners*, Sections 2
and 3.3 ([paper](https://arxiv.org/abs/2005.14165)). The method conditions a
generation on the task and input/output examples, without a retrieval loop.

**Our implementation:** one closed-book answer call using the six existing
QA demonstrations in `original_prompt`. Temperature 0 and 128 output tokens
are defaults we select for this QA implementation; Brown et al. do not specify
these settings for Qwen. The demonstration set is fixed during inference.

## Chain-of-thought prompting

**Source:** Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models*, Sections 2 and 3.1
([paper](https://arxiv.org/abs/2201.11903)). The paper uses few-shot reasoning
examples and greedy decoding; its eight-example arithmetic prompt is
task-specific, rather than a multi-hop QA default.

**Our implementation:** one closed-book call generates reasoning followed
by an answer. The six existing QA reasoning examples are retained. The active
prompt requests a reasoning chain followed by a final answer. Temperature 0 follows greedy
decoding; the 512-token output cap is a chosen QA default. Answer extraction
reads the explicit final marker and does not launch a retrieval or voting stage.

## IRCoT

**Paper:** Trivedi et al., Sections 3.1–3.2 and 4.1
([paper](https://aclanthology.org/2023.acl-long.557/)): retrieve with the original
question, generate one next reasoning sentence, retrieve with that sentence,
and accumulate evidence. Footnotes 4 and 5 give eight reasoning steps and
fifteen collected paragraphs. A separate QA reader generates a new complete
reasoning chain from the retrieved collection.

**Code:** commit `3c1820f698eea5eeddb4fba3c56b64c961e063e4`;
[`ircot_codex_hotpotqa.jsonnet`](https://github.com/StonyBrookNLP/ircot/blob/3c1820f698eea5eeddb4fba3c56b64c961e063e4/base_configs/ircot_codex_hotpotqa.jsonnet)
sets BM25 retrieval count 6 and paragraph cap 15.
[`GPT3Generator`](https://github.com/StonyBrookNLP/ircot/blob/3c1820f698eea5eeddb4fba3c56b64c961e063e4/commaqa/models/gpt3generator.py)
defaults to temperature 0 and 300 output tokens. The generic code exit
controller defaults to ten sentences; we use the paper's eight-step cap
across datasets.

**Our implementation:** BM25 is applied within the supplied collection, and
Qwen replaces Codex/Flan-T5. Five existing QA reasoning examples replace the
authors' dataset-specific paragraph/distractor demonstration construction.
Both next-sentence generation and the independent QA reader use these examples.
A lightweight punctuation splitter replaces the original spaCy sentence
segmenter; this can differ on abbreviations. Extra generated sentences are
discarded, and only the first is used as the next query. Duplicate passage IDs
are removed before the fifteen-passage cap. Stop conditions are the answer
marker, eight turns, fifteen passages, or empty generation. The final reader
receives passages without the retrieval-stage answer guess.

## ReAct

**Source:** Yao et al., Section 3 and Appendix A.1
([paper](https://arxiv.org/abs/2210.03629)); official commit
`6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`.
The [`HotpotQA notebook`](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb)
uses six `webthink_simple6` demonstrations, temperature 0, 100 output tokens,
and seven action turns, then `finish[]` if unfinished. These notebook settings
belong to text-davinci-002; the original paper also reports PaLM-540B results.
[`WikiEnv`](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/wikienv.py)
shows the first five sentences after Search and advances a keyword cursor on
repeated Lookup calls.

**Our implementation:** the six original
[`webthink_simple6` demonstrations](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/prompts/prompts_naive.json)
are included in `react.py` with their MIT copyright and license notice.
Search opens one ranked local passage instead of an exact Wikipedia page.
Lookup examines only that current passage, returning one matching sentence per
call; another Search resets its cursor. The implementation preserves the seven
turns, 100-token cap, and empty answer on exhaustion. The shared evaluation
protocol rejects an invalid action without a formatting-repair call, recording
the failed question and all usage from the attempted generation.
Any model-generated Observation is discarded before another action is executed.

## Self-Ask

**Source:** Press et al., Sections 3.3–3.5
([paper](https://aclanthology.org/2023.findings-emnlp.378/)); official
[`search-engine notebook`](https://github.com/ofirpress/self-ask/blob/559e56ec1ba8dc93cf63c4148ec1f4717caf4b60/self-ask_plus_search-engine_demo.ipynb)
at commit `559e56ec1ba8dc93cf63c4148ec1f4717caf4b60`. The controller emits
follow-up questions; Google answer boxes/snippets supply intermediate answers.
The notebook uses temperature 0 and 256 output tokens, and permits model
fallback when the search engine supplies no answer. Its loop has no numerical
iteration bound.

**Our implementation:** retain four existing follow-up examples and replace
Google with local retrieval plus a Qwen passage reader. Three documents per
follow-up follow the separate Self-Ask reproduction in DEC Appendix A
([source](https://aclanthology.org/2025.findings-acl.846/)). The original Google-backed
notebook leaves the retrieval count unspecified. Eight controller turns are
a chosen bound. The passage reader's answer is appended before generating the
next follow-up; intermediate answers generated by the controller are discarded. Missing
retrieval answers trigger the original-style internal-knowledge fallback.

## Plan-and-Solve

**Source:** Wang et al., Section 3
([paper](https://aclanthology.org/2023.acl-long.147/)); official commit
`9d5a176f5b4de1011d426009ab6398dc76046e88`.
[`config.py`](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting/blob/9d5a176f5b4de1011d426009ab6398dc76046e88/config.py)
sets temperature 0, `max_length_cot=256`, `max_length_direct=32`, and
self-consistency disabled. The paper's PS instruction first requests a plan
and then execution of that plan.

**Our implementation:** the zero-shot PS instruction drives one plan/solve
generation, followed by a separate concise-answer extraction call. The
256/32-token limits are retained. This is the PS variant; no PS+ arithmetic
augmentation, self-consistency voting, or retrieval is added. The module also
contains a few-shot QA template; this configuration uses the zero-shot prompt.

## Generate-then-Ground

**Source:** Shi et al., Sections 3.1–3.3 and 5.4
([paper](https://aclanthology.org/2024.acl-long.397/)): each iteration generates
one sub-question and provisional answer, retrieves ten documents, and revises
the answer by citing evidence. Grounding examines ranked batches of three and
stops at the first supporting batch. If all batches lack evidence, the original
guess is retained. The paper uses ColBERTv2 and gpt-3.5-turbo, temperature 0.

**Our implementation:** Qwen and the supplied dense candidates replace those
components. JSON encodes the deduction/grounding outputs. Grounding requires a
nonempty evidence quote that matches a passage after whitespace/case
normalization.
The final one-document batch is included when all ten candidates are available.
Only the revised answer enters the next deduction. Three explicitly fictional
formatting examples demonstrate sequential deduction and revision. Eight
deduction turns and 512/128 generation/final-output tokens are chosen limits;
exhaustion invokes a final synthesis of the available history.

**Source availability:** the
[`official repository`](https://github.com/shizhl/Generate-then-Ground/tree/4a40dde76833d5a0ef7f1d4f663ec7a787b8e1b2)
at commit `4a40dde76833d5a0ef7f1d4f663ec7a787b8e1b2` contains only README.
No runnable official implementation or original deduction-example set was
available there. The separate Mistral-7B grounding-distillation variant is not
used by this shared-Qwen inference configuration.

## DEC

**Paper:** Ji et al., Sections 3.1–3.2, Figure 5, and Appendix A
([paper](https://aclanthology.org/2025.findings-acl.846/)): create one ordered
decomposition; rewrite each sub-question using earlier answers; retrieve ten
E5 candidates; retain the best two plus keyword-matching documents. Figure 5
asks for one or two single-word discriminative keywords. The original extractor
is a LoRA-tuned Llama-3.2-3B-Instruct model.

**Code:** commit `f46b99bfd1cc1839e75c7b4f54e72478606ebb78`,
[`DEC_LOCAL_EK_ALL_base2.py`](https://github.com/neukg/DEC/blob/f46b99bfd1cc1839e75c7b4f54e72478606ebb78/DEC_LOCAL_EK_ALL_base2.py),
functions `process_question`, `process_sub_question`, and `EK_recaller`.
The filter preserves the first two documents and appends a remaining document
only if **every** extracted keyword occurs literally in its contents. An empty
keyword list retains just the first two. Matching is case-sensitive in the
released code. `utils.py` does not set generation temperature or token limits.

**Our implementation:** We use Qwen for keyword extraction and BGE for dense
retrieval. Extracted keywords filter the combined title and text of each passage. Default keyword count is two; the extractor has a
chosen 128-token output cap. Decomposition and rewrites use JSON; each answer
updates the history before the next rewrite. Temperature 0, 512-token controller
outputs, 128-token answers, and at most eight sub-questions are explicit chosen
limits. Final synthesis combines the answered sub-questions. The original code
has no numerical sub-question cap.

## Source versions

Code references link to the commits used for these implementations. ReAct's
demonstrations retain their MIT notice. Settings derived from papers are
identified in each configuration's `parameter_sources` entries.
