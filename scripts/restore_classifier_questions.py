"""Restore question texts from the dataset IDs in a classifier manifest."""
import argparse
import json
from pathlib import Path


def iter_records(path):
    with path.open(encoding='utf-8') as stream:
        if path.suffix == '.jsonl':
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            yield from json.load(stream)


def restore(manifest, sources):
    rows = list(iter_records(manifest))
    questions = {}
    for dataset, path in sources.items():
        wanted = {r['example_id'] for r in rows if r['dataset'] == dataset}
        for row in iter_records(path):
            example_id = str(row.get('_id', row.get('id')))
            if example_id in wanted:
                key = dataset, example_id
                question = row['question']
                if key in questions and questions[key] != question:
                    raise ValueError(f'Conflicting question texts for {dataset}/{example_id}')
                questions[key] = question
    missing = [(r['dataset'], r['example_id']) for r in rows
               if (r['dataset'], r['example_id']) not in questions]
    if missing:
        raise ValueError(f'{len(missing)} manifest IDs are absent from the supplied training files')
    return [dict(row, question=questions[(row['dataset'], row['example_id'])]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--hotpot-train', type=Path, required=True)
    parser.add_argument('--musique-train', type=Path, required=True)
    parser.add_argument('--wiki-train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path to preserve existing data')
    rows = restore(args.manifest, {'hotpot': args.hotpot_train,
                                  'musique': args.musique_train, '2wiki': args.wiki_train})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'restored_questions': len(rows)}))


if __name__ == '__main__':
    main()
