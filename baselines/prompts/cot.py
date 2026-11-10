cot_prompt = '''
Instruction: You are an assistant that thinks step by step but only outputs the final answer. You answer based only on your internal world knowledge. For each question, you must output only the final answer after "Answer:" with no explanation or extra words.

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Answer: 1,800 to 7,000 ft

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Answer: Richard Nixon

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Answer: The Saimaa Gesture

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Answer: director, screenwriter, actor

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's agazine was started first.
Answer: Arthur's Magazine

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Answer: Yes

Question: {Question}
Thought: {Thought}
Answer:
'''

# Active shared-backbone adaptation: retain the archived six QA demonstrations,
# but allow the model to generate the reasoning before extracting its answer.
cot_runtime_prompt = cot_prompt.replace(
    "Instruction: You are an assistant that thinks step by step but only outputs the final answer. You answer based only on your internal world knowledge. For each question, you must output only the final answer after \"Answer:\" with no explanation or extra words.",
    "Use your internal knowledge to reason through the question step by step, as in the examples. Write the reasoning, then put the concise final answer on its own line after Answer:.",
).replace("Thought: {Thought}\nAnswer:", "Thought:")
