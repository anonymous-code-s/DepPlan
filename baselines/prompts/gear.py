"""GeAR stages from ACL Findings 2025 sections 3–5 and Appendix K.
JSON schemas are an executable serialization adaptation of the paper's prompts.
"""
gear_extract_prompt = """Build a knowledge graph from the supplied passage.
Extract its factual (subject, relation, object) triples. Use explicit entity names
instead of pronouns; retain dates and quantities; do not add outside facts.
Return JSON: {"triples": [["subject", "relation", "object"], ...]}.
Extract all factual relations, independently of any downstream question."""
gear_reader_prompt = """Read the passages and extract proximal knowledge triples
that help answer the current query or connect it to the original question.
Existing gist memory is provided only to identify useful new connections.
Return JSON: {"triples": [["subject", "relation", "object"], ...]}.
Use explicit entity names; every new triple must be supported by the passages."""
gear_reasoner_prompt = """Decide whether the gist memory contains enough facts to
answer the original question. If enough, derive a concise answer; otherwise state
the missing relation. Return JSON: {"answerable": true, "answer": "short answer",
"reason": "brief evidence assessment"}. Set answerable to false and answer to an
empty string when evidence is incomplete."""
gear_rewriter_prompt = """Use the original question, gist memory, and missing
information identified by the reasoner to formulate the next retrieval query.
Resolve intermediate entities using memory. Return JSON: {"query": "one query"}.
Do not merely repeat a previous query when a missing relation is known."""
gear_answer_prompt = """Answer the question from the retrieved passages. Connect
the relevant evidence across passages and provide only the concise final answer
in at most six words. If the evidence is insufficient, return Unknown.
Illustrative example: Passage 1: The film Amber was directed by Nera.
Passage 2: Nera died on May 2, 1988. Passage 3: Sol was a musician.
Question: When did the director of Amber die? Answer: May 2, 1988."""
