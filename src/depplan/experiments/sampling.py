"""Create a fixed, proportionally stratified question manifest."""
import argparse
from collections import defaultdict
import math
import random

from .common import (new_output, question_key, read_jsonl, source_files, write_json,
                     write_jsonl)


def stratum(row):
    if row['dataset'] == 'musique':
        hops = row.get('hops', len(row.get('question_decomposition', [])))
        if hops not in (2, 3, 4):
            raise ValueError('MuSiQue questions need hops in {2, 3, 4}')
        return str(hops) + '-hop'
    label = row.get('question_type', row.get('type'))
    if not isinstance(label, str) or not label.strip():
        raise ValueError('HotpotQA and 2WikiQA questions need their official question_type')
    return label


def select(rows, count, seed):
    if count < 1 or len(rows) < count:
        raise ValueError('Sample size must be positive and no larger than the input')
    if len({question_key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    groups = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    exact = {k: count * len(v) / len(rows) for k, v in groups.items()}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    for key in sorted(groups, key=lambda k: (-(exact[k] - quotas[k]), k))[:count-sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = [r for key in sorted(groups)
                for r in rng.sample(sorted(groups[key], key=question_key), quotas[key])]
    return sorted(selected, key=question_key), quotas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-dataset', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    rows = [r for p in args.input for r in read_jsonl(p)]
    selected, counts = [], {}
    for dataset in sorted({r['dataset'] for r in rows}):
        sample, quotas = select([r for r in rows if r['dataset'] == dataset], args.per_dataset, args.seed)
        selected.extend(sample)
        counts[dataset] = quotas
    if not selected:
        raise ValueError('No questions supplied')
    output = new_output(args.output)
    write_jsonl(output / 'questions.jsonl', selected)
    write_json(output / 'manifest.json', {'sampling_seed': args.seed, 'allocation': 'proportional',
        'sources': source_files(args.input), 'stratum_counts': counts,
        'questions': [{'dataset': r['dataset'], 'example_id': r['example_id'], 'stratum': stratum(r)}
                      for r in selected]})


if __name__ == '__main__':
    main()
