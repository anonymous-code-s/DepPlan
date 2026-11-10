dec_plan_prompt = '''
You are an expert in multi-hop question answering.

Your task is to decompose a complex question into an ordered sequence of
logically connected sub-questions. These sub-questions will later be used
for retrieval and step-by-step answering.

Requirements:
- The sub-questions must be specific and self-contained.
- The sequence of sub-questions should form a coherent reasoning chain.
- Each sub-question should be answerable by looking up external documents.
- The LAST sub-question MUST be equivalent to the original question
  (i.e., directly asking for the final answer).
- Avoid vague pronouns such as "it", "this", "they" when they would be
  ambiguous for retrieval. Use explicit entity names instead.
- Do NOT include any answers in this step, only questions.

Output format (VERY IMPORTANT):
- You MUST output a single valid JSON object.
- The JSON MUST have exactly one key: "sub_questions".
- The value of "sub_questions" MUST be a list of strings.
- Do NOT include any additional keys or commentary.
- Do NOT wrap the JSON in backticks.

Example of the desired format:
{{
  "sub_questions": [
    "Sub-question 1 text",
    "Sub-question 2 text",
    "Sub-question 3 text"
  ]
}}

Now decompose the following complex question:

Question: {question}
'''

# Runtime plan follows the official single decomposition. It does not append a
# redundant copy of the complete original question as a mandatory final hop.
dec_runtime_plan_prompt = '''Break the question into an ordered list of simpler
sub-questions whose answers together solve it. Later sub-questions may refer to
answers of earlier ones; these references will be rewritten after each answer
is obtained. Return only JSON: {{"sub_questions": ["first question", "next question"]}}.

Question: {question}
'''

# The original paper's Figure 5 requests 1--2 single-word discriminative terms.
# The shared Qwen model replaces the authors' trained keyword extractor.
dec_runtime_keyword_prompt = '''Extract up to {count} discriminative keywords
from the query. Prefer distinctive proper names or terms that occur literally
in the query. Each keyword must be one word. Preserve spelling and case; avoid
common words. Return JSON {{"keywords": ["keyword"]}}, or an empty list when
no discriminative keyword is available.

Query: {query}
'''

dec_rewrite_prompt = '''
You are an expert at rewriting questions for document retrieval.

You are given:
1) The original complex question.
2) The current sub-question from a logical reasoning chain.
3) A history of previously solved sub-questions and their answers.

Your task is to rewrite the current sub-question into a fully self-contained
query that is optimal for retrieval.

Guidelines:
- Replace vague references (e.g., "this museum", "this series", "he", "she")
  with explicit entities using the information from the history.
- If the original question or history mentions specific names, dates, or
  locations, incorporate them into the rewritten query when relevant.
- Keep the rewritten query concise but informative.
- The rewritten query should still ask for the SAME information as the
  current sub-question.

Input:
- Original question: {original_question}
- Current sub-question: {current_sub_question}
- History (JSON-like list of [sub_question, answer] pairs):
{history}

Output format (VERY IMPORTANT):
- You MUST output a single valid JSON object.
- The JSON MUST have exactly one key: "rewritten_query".
- The value MUST be a single string with the rewritten query.
- Do NOT include any additional keys or commentary.
- Do NOT wrap the JSON in backticks.

Example:
{{
  "rewritten_query": "What is the title of the 26-episode series that features the Imperial War Museum?"
}}
'''

dec_keyword_prompt = '''
You are a helpful assistant for information retrieval.

You are given a rewritten query that will be used to search a document
collection. Your task is to extract a small set of discriminative keywords
from this query.

Requirements:
- Extract 1 to 3 key phrases that are most important for identifying the
  relevant documents.
- Prefer named entities (people, places, organizations, titles) and other
  specific phrases (e.g., "Imperial War Museum", "26-episode series").
- Avoid common stopwords such as "the", "a", "is", "in", "of", etc.
- Keywords can be multi-word phrases.
- Keep each keyword as it appears in the query (do not paraphrase).

Rewritten query:
{query}

Output format (VERY IMPORTANT):
- You MUST output a single valid JSON object.
- The JSON MUST have exactly one key: "keywords".
- The value MUST be a list of strings, each being one keyword or key phrase.
- The list MUST contain between 1 and 3 items (inclusive).
- Do NOT include any additional keys or commentary.
- Do NOT wrap the JSON in backticks.

Example:
{{
  "keywords": [
    "Imperial War Museum",
    "26-episode series"
  ]
}}
'''

dec_answer_prompt = '''
You are a question answering assistant.

You are given:
1) A sub-question from a multi-hop reasoning chain.
2) A rewritten version of this sub-question that was used for retrieval.
3) A set of context passages retrieved from a document collection.

Your task is to extract a short, precise answer to the ORIGINAL sub-question
using ONLY the information in the provided context passages.

Instructions:
- Read the context carefully.
- If the answer is explicitly stated, copy or lightly normalize it.
- Prefer the most specific and concise form (a name, date, number, place, etc.).
- Do NOT add explanations.
- Do NOT use any knowledge that is not present in the context.
- If the context does NOT contain enough information to answer the sub-question,
  output exactly: No-answer

Sub-question:
{sub_question}

Rewritten query (for your reference, do NOT answer this query directly):
{rewritten_query}

Context passages:
{context}

Now provide the answer.

Answer:
'''
