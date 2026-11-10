prompt_react = '''
Instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples.

Input: Who succeeded the first President of Namibia?
Thought 1: I need to search for the first President of Namibia.
Action 1: Search[First President of Namibia]
Observation 1: The first President of Namibia was Sam Nujoma.
Thought 2: Now I need to find out who succeeded Sam Nujoma as the President of Namibia.
Action 2: Search[Successor of Sam Nujoma]
Observation 2: The successor of Sam Nujoma was Hifikepunye Pohamba, who took office in 2005.
Thought 3: I now know the answer.
Action 3: Finish[Hifikepunye Pohamba]

Input: What currency is used where Billy Giles died?
Thought 1: I need to find the location where Billy Giles died.
Action 1: Search[Billy Giles death location]
Observation 1: Billy Giles died in Belfast, Northern Ireland.
Thought 2: I need to determine the currency used in Northern Ireland.
Action 2: Search[Currency used in Northern Ireland]
Observation 2: The currency used in Northern Ireland is the pound sterling.
Thought 3: The currency used in the location where Billy Giles died is pound sterling.
Action 3: Finish[pound sterling]

Input: When did the first establishment McDonaldization is named after open in the country Horndean is located?
Thought 1: I need to determine what McDonaldization is named after.
Action 1: Search[McDonaldization named after]
Observation 1: McDonaldization is named after McDonald's.
Thought 2: I need to find the country where Horndean is located.
Action 2: Search[Horndean location]
Observation 2: Horndean is located in England.
Thought 3: I need to find out when the first McDonald's opened in England.
Action 3: Finish[1974]

Input: When did Napoleon occupy the city where the mother of the woman who brought Louis XVI style to the court died?
Thought 1: I need to determine who brought Louis XVI style to the court and identify her mother.
Action 1: Search[woman who brought Louis XVI style to the court]
Observation 1: Marie Antoinette brought the Louis XVI style to the court. Her mother was Empress Maria Theresa.
Thought 2: I need to find the city where Maria Theresa died.
Action 2: Search[city where Maria Theresa died]
Observation 2: Maria Theresa died in Vienna.
Thought 3: I need to find when Napoleon occupied Vienna.
Action 3: Finish[1805]

Input: How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?
Thought 1: I need to find the continent where Aruba is located.
Action 1: Search[Aruba >> continent]
Observation 1: Aruba is located in South America.
Thought 2: I need to find which country governed Prazeres.
Action 2: Search[Prazeres >> country]
Observation 2: Prazeres was governed by Portugal.
Thought 3: I need to determine the colonial holding in South America governed by Portugal and find out how many Germans live there.
Action 3: Finish[5 million]

Input: {input}
'''

# Six original HotpotQA demonstrations from ysymyth/ReAct, commit
# 6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9, prompts/prompts_naive.json.
# The upstream MIT notice accompanies this redistributed prompt text.
REACT_UPSTREAM_LICENSE = 'MIT License\n\nCopyright (c) 2023 Shunyu Yao\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

REACT_EXAMPLES = '\nQuestion: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\nThought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\nAction 1: Search[Colorado orogeny]\nObservation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nThought 2: It does not mention the eastern sector. So I need to look up eastern sector.\nAction 2: Lookup[eastern sector]\nObservation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nThought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\nAction 3: Search[High Plains]\nObservation 3: High Plains refers to one of two distinct land regions:\nThought 4: I need to instead search High Plains (United States).\nAction 4: Search[High Plains (United States)]\nObservation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nThought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAction 5: Finish[1,800 to 7,000 ft]\nQuestion: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?\nThought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.\nAction 1: Search[Milhouse]\nObservation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\nThought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".\nAction 2: Lookup[named after]\nObservation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \nThought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\nAction 3: Finish[Richard Nixon]\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: [\'Adam Clayton Powell III\', \'Seventh Avenue (Manhattan)\', \'Adam Clayton Powell Jr. State Office Building\', \'Isabel Washington Powell\', \'Adam Powell\', \'Adam Clayton Powell (film)\', \'Giancarlo Esposito\'].\nThought 2: To find the documentary, I can search Adam Clayton Powell (film).\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nThought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAction 3: Finish[The Saimaa Gesture]\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nThought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\nAction 1: Search[Nicholas Ray]\nObservation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nThought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\nAction 2: Search[Elia Kazan]\nObservation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nThought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction 3: Finish[director, screenwriter, actor]\nQuestion: Which magazine was started first Arthur\'s Magazine or First for Women?\nThought 1: I need to search Arthur\'s Magazine and First for Women, and find which was started first.\nAction 1: Search[Arthur\'s Magazine]\nObservation 1: Arthur\'s Magazine (1844-\x80\x931846) was an American literary periodical published in Philadelphia in the 19th century. \nThought 2: Arthur\'s Magazine was started in 1844. I need to search First for Women next.\nAction 2: Search[First for Women]\nObservation 2: First for Women is a woman\'s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \nThought 3: First for Women was started in 1989. 1844 (Arthur\'s Magazine) < 1989 (First for Women), so Arthur\'s Magazine was started first.\nAction 3: Finish[Arthur\'s Magazine]\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]\n'

react_runtime_prompt = 'Solve the question with interleaved Thought, Action, and Observation turns.\nUse exactly one action per turn and stop before Observation; the environment\nsupplies observations. Actions:\nSearch[entity]: retrieve one relevant passage from the supplied collection and\nshow its first sentences; that passage becomes the current passage.\nLookup[keyword]: return the next sentence containing the keyword in the current\npassage. Repeated Lookup advances through the matches.\nFinish[answer]: provide a concise final answer and finish.\nThe demonstrations show the original Wikipedia environment; the current search\nuses the supplied passage collection and may return different observations.\n' + REACT_EXAMPLES + '\nQuestion: {input}\n'
