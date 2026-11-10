prompt_selfask = '''Answer the following Question using the format below. 

Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Answer: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Answer: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Answer: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
Answer: No

Question: {Question}
'''

self_ask_runtime_prompt = prompt_selfask.replace(
    "Answer the following Question using the format below.",
    "Answer the following Question using the format below. For each turn, either ask ONE Follow up: question and stop before Intermediate answer:, or give Answer: when the prior answers suffice. The external search component supplies intermediate answers; do not invent its output.",
)

retrieve_answer_prompt = '''
Instruction: You are an NLP assistant. Your task is to extract a concise and accurate answer to the given sub-question strictly based on the provided context. The answer must be short and factual. If the context does not contain sufficient information to answer the question, respond with "No-answer".

question: Who was the first President of Namibia?
context: Sam Nujoma was sworn in as the first President of Namibia.
answer: Sam Nujoma

question: Is The Square a documentary?
context: The Square is a 1994 Chinese documentary film directed by Zhang Yuan. It is Zhang's first true documentary film, after two documentary-influenced fiction films: "Mama" and "Beijing Bastards".
answer: Yes

question: What is Philipsburg the capital of?
context: Philipsburg is the main town and capital of the country of Sint Maarten. The town is situated on a narrow stretch of land between Great Bay and the Great Salt Pond. It functions as the commercial center of Saint Martin island, whereof Sint Maarten encompasses the southern half.
answer: Sint Maarten

question: When did the French come to the Caribbean?
context: Pierre Belain d'Esnambuc was a French trader and adventurer in the Caribbean, who established the first permanent French colony, Saint-Pierre, on the island of Martinique in 1635. Belain sailed to the Caribbean in 1625, hoping to establish a French settlement on the island of St. Christopher (St. Kitts).
answer: 1625

question: How many Germans live in Brazil?
context: People of German origin are found in various places around the globe. United States is home to approximately 50 million German Americans or one third of the German diaspora, making it the largest centre of German-descended people outside Germany. Brazil is the second largest with 5 million people claiming German ancestry.
answer: 5 million

question: Did Marie Curie win a Nobel Prize in Physics?
context: Marie Curie conducted pioneering research on radioactivity.
answer: No-answer

question: {question}
context: {context}
answer:
'''
