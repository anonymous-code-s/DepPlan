prompt_plan_solve = '''Answer the following Question using the format below. Provide only the concise final answer; do not include explanations or any extra words.

Question: Who lived longer, Muhammad Ali or Alan Turing?
Instruction: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
Plan:
Step 1: How old was Muhammad Ali when he died?  
Step 2: How old was Alan Turing when he died?  
Solution:
Step 1: Muhammad Ali was 74 years old when he died.  
Step 2: Alan Turing was 41 years old when he died.  
Answer: Muhammad Ali

Question: When was the founder of craigslist born?
Instruction: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
Plan:
Step 1: Who was the founder of Craigslist?  
Step 2: When was that person born?
Solution:
Step 1: Craigslist was founded by Craig Newmark.  
Step 2: Craig Newmark was born on December 6, 1952.
Answer: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Instruction: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
Plan:
Step 1: Who was the mother of George Washington?  
Step 2: Who was the father of his mother?
Solution:
Step 1: The mother of George Washington was Mary Ball Washington.  
Step 2: The father of Mary Ball Washington was Joseph Ball.
Answer: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Instruction: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
Plan:
Step 1: Who is the director of Jaws?  
Step 2: Where is the director of Jaws from?  
Step 3: Who is the director of Casino Royale?  
Step 4: Where is the director of Casino Royale from? 
Solution:
Step 1: The director of Jaws is Steven Spielberg.  
Step 2: Steven Spielberg is from the United States.  
Step 3: The director of Casino Royale is Martin Campbell.  
Step 4: Martin Campbell is from New Zealand. 
Answer: No

Question: {Question}
Instruction: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
'''

# Published PS zero-shot instruction (Wang et al., Section 3). The archived
# few-shot QA adaptation above is not the prompt executed by the new runner.
plan_solve_runtime_prompt = '''Question: {question}
Let's first understand the problem and devise a plan to solve the problem.
Then, let's carry out the plan to solve the problem step by step.
'''

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
