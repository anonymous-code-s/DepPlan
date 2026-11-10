"""Convert an official dataset file to per-question passages for run-qa."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',choices=['hotpot','musique','2wiki'],required=True)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
text=a.input.read_text()
rows=[json.loads(s) for s in text.splitlines() if s.strip()] if a.input.suffix=='.jsonl' else json.loads(text)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
    for row in rows:
        if a.dataset=='musique':
            passages=[str(x['title'])+'\n'+str(x['paragraph_text']) for x in row['paragraphs']]
        else:
            passages=[str(title)+'\n'+' '.join(sentences) for title,sentences in row['context']]
        answer=row['answer']
        if isinstance(answer,dict):
            answers=[answer['text'],*answer.get('aliases',[])]
        else:
            answers=[str(answer),*row.get('answer_aliases',[])]
        item={'dataset':a.dataset,'example_id':str(row.get('_id',row.get('id'))),
              'question':row['question'],'passages':passages,'answers':answers}
        if a.dataset == 'musique':
            item['hops'] = len(row['question_decomposition'])
            item['question_decomposition'] = row['question_decomposition']
        elif 'type' in row:
            item['question_type'] = row['type']
        f.write(json.dumps(item,ensure_ascii=False)+'\n')
