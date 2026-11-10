"""DeepRAG's released inference instruction and completion boundaries.

Source: gxy-gxy/DeepRAG, commit 1665780b7b6317375d56e71c3ddc52d6f706c380,
ICL-prompt/instruction.txt and src/algorithms/sragsftv2.py.
"""

INSTRUCTION = (
    'Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. '
    'Your task is to answer questions by logically decomposing them into clear '
    'sub-questions and iteratively addressing each one. Use "Follow up:" to '
    'introduce each sub-question and "Intermediate answer:" to provide answers. '
    'For each sub-question, decide whether you can provide a direct answer or if '
    'additional information is required. If additional information is needed, '
    'state, "Let\'s search the question in Wikipedia." and then use the retrieved '
    'information to respond comprehensively. If a direct answer is possible, '
    'provide it immediately without searching.\n'
)

EOS = ("<|im_end|>", "<|eot_id|>")
FOLLOW_UP_STOP = (*EOS, "Intermediate answer:", "So the final", "Let")
INTERMEDIATE_STOP = (*EOS, "Follow up:", "So the final answer")
FINAL_STOP = (*EOS, "Follow up:")


def completion_prompt(tokenizer, question: str, assistant: str) -> str:
    """Apply the checkpoint template once, retaining its assistant prefill."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": INSTRUCTION + "Question: " + question},
            {"role": "assistant", "content": assistant},
        ],
        add_generation_prompt=False,
        tokenize=False,
    ).strip("\n")
    return prompt.removesuffix("<|eot_id|>").removesuffix("<|im_end|>")


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    parts = []
    for step in history:
        item = "\nFollow up: " + step["follow_up"].strip()
        if "docs" in step:
            item += "\nLet's search the question in Wikipedia.\nContext:\n" + step["docs"]
        item += "\nIntermediate answer: " + step["answer"].strip()
        parts.append(item)
    return "".join(parts) + "\n"
