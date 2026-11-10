"""Read official development files and prepared question/passage JSONL."""
import hashlib
import json
from pathlib import Path


DATASETS = {'HotpotQA': 'hotpot', 'hotpotqa': 'hotpot', 'hotpot': 'hotpot',
            'MuSiQue': 'musique', 'musique': 'musique',
            '2WikiQA': '2wiki', '2wiki': '2wiki', '2wikimultihopqa': '2wiki'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def question(row, dataset, index):
    dataset = DATASETS.get(row.get('dataset', dataset), row.get('dataset', dataset))
    if dataset not in {'hotpot', 'musique', '2wiki'}:
        raise ValueError('Specify the dataset for each input file')
    question_text = row.get('question') or row.get('query')
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError(f'Input row {index} has no question')
    values = row.get('passages', row.get('flat_contexts'))
    if values is None:
        if 'paragraphs' in row:
            values = [dict(id=str(p.get('idx', i)), title=p.get('title', ''),
                           text=p['paragraph_text']) for i, p in enumerate(row['paragraphs'])]
        else:
            values = [dict(id=str(i), title=title, text=' '.join(sentences))
                      for i, (title, sentences) in enumerate(row.get('context', []))]
    passages = []
    for i, item in enumerate(values or []):
        if isinstance(item, str):
            title, sep, text = item.partition('\n')
            item = dict(id=str(i), title=title if sep else '', text=text if sep else title)
        if not isinstance(item, dict):
            raise ValueError(f'Invalid passage at input row {index}')
        text = item.get('text', item.get('paragraph_text', item.get('contents', '')))
        passages.append(dict(id=str(item.get('id', i)), title=str(item.get('title', '')),
                             text=str(text)))
    if len({p['id'] for p in passages}) != len(passages):
        raise ValueError(f'Duplicate passage IDs at input row {index}')
    answers = row.get('answers')
    if answers is None:
        answer = row.get('answer')
        if isinstance(answer, dict):
            answers = [answer.get('text', ''), *answer.get('aliases', [])]
        elif answer is not None:
            answers = [answer, *row.get('answer_aliases', [])]
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f'Input row {index} has no reference answers')
    return dict(dataset=dataset, example_id=str(row.get('example_id', row.get('_id', row.get('id', index)))),
                question=question_text, passages=passages, answers=[str(a) for a in answers])


def load_questions(path, dataset=None):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    raw = (json.loads(text) if text.lstrip().startswith('[') else
           [json.loads(line) for line in text.splitlines() if line.strip()])
    if not isinstance(raw, list) or not raw:
        raise ValueError('Input must contain a nonempty JSON array or JSONL question list')
    rows = [question(row, dataset, i) for i, row in enumerate(raw)]
    keys = [(row['dataset'], row['example_id']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate dataset/question IDs in input')
    return rows
