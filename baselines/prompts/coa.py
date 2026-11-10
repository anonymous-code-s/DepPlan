"""CoA query-task prompts adapted from NeurIPS 2024 Table 11."""
prompt_worker = """Read the current source text and the summary of the preceding
source text. Write an updated summary that preserves relevant evidence from both.
The next worker receives this summary; a manager ultimately uses it to answer the
question. Include entities, relations, dates, and intermediate connections needed
for the question. Preserve useful earlier evidence even when the current source
adds nothing. Return the updated summary as plain text."""
prompt_manager = """Answer the question using the final summary of the source
text. Give only the concise answer, such as an entity, date, number, or yes/no.
If the summary does not support an answer, return Unknown."""
