"""Input validation, manifests and seed aggregation for supplementary experiments."""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .ablations.reporting import digest, read_jsonl, write_json
from .ablations.runner import file_digest


def service_arguments(parser):
    parser.add_argument('--model', default='Qwen2.5-32B-Instruct')
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://localhost:8000/v1'))
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    parser.add_argument('--device', default='cpu')


def check_grid(values, name, allowed=None):
    if not values or len(set(values)) != len(values):
        raise ValueError(name + ' must be nonempty and unique')
    if any(type(v) is not int or v < 0 or v >= 2**32 for v in values):
        raise ValueError(name + ' must contain nonnegative integers below 2**32')
    if allowed is not None and set(values) - set(allowed):
        raise ValueError('Unsupported ' + name)


def parse_config(parser, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    known, _ = preliminary.parse_known_args(argv)
    parser.add_argument('--config', help='Experiment settings in JSON; CLI options take precedence')
    if known.config:
        settings = json.loads(Path(known.config).read_text())
        allowed = {action.dest for action in parser._actions} - {'config', 'help'}
        if not isinstance(settings, dict) or set(settings) - allowed:
            parser.error('Unknown experiment configuration fields')
        parser.set_defaults(**settings)
    return parser.parse_args(argv)


def source_files(paths):
    return [{'file': Path(p).name, 'sha256': file_digest(p)} for p in paths]


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError('Use an empty output directory: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty summary')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def question_key(row):
    dataset = row['dataset']
    qid = str(row['example_id'])
    prefix = dataset + ':'
    return dataset, qid[len(prefix):] if qid.startswith(prefix) else qid


def load_runs(paths, method='depplan'):
    rows, seen = [], set()
    for path in paths:
        for original in read_jsonl(path):
            row = dict(original)
            row['method'] = row.get('method', row.get('variant', method))
            if row['method'] == 'full':
                row['method'] = 'depplan'
            if type(row.get('seed')) is not int:
                raise ValueError('Every prediction needs its decoding seed')
            if row.get('dataset') not in {'hotpot', 'musique', '2wiki'}:
                raise ValueError('Every prediction needs a supported dataset key')
            identity = (row['method'], row['seed'], *question_key(row))
            if identity in seen:
                raise ValueError('Duplicate prediction: ' + str(identity))
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError('No predictions supplied')
    return rows


def validate_seed_cohorts(rows):
    groups = defaultdict(lambda: defaultdict(set))
    for row in rows:
        groups[row['method'], row['dataset']][row['seed']].add(question_key(row))
    for group, seeds in groups.items():
        cohorts = list(seeds.values())
        if any(ids != cohorts[0] for ids in cohorts[1:]):
            raise ValueError(f'Question IDs differ across seeds for {group}')


def seed_summary(rows, group_fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_fields)].append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        if len({r['seed'] for r in values}) != len(values):
            raise ValueError('Duplicate seed in summary group')
        item = dict(zip(group_fields, key))
        item['seeds'] = len(values)
        for metric in metrics:
            numbers = [r.get(metric) for r in values]
            complete = all(v is not None and math.isfinite(v) for v in numbers)
            item[metric + '_mean'] = mean(numbers) if complete else None
            item[metric + '_sample_sd'] = stdev(numbers) if complete and len(numbers) > 1 else None
        summaries.append(item)
    return summaries


def load_gold(path, dataset='musique'):
    text = Path(path).read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else read_jsonl(path)
    result = {}
    for row in rows:
        record = dict(row, dataset=row.get('dataset', dataset),
                      example_id=str(row.get('example_id', row.get('id', row.get('_id', '')))))
        key = question_key(record)
        if not key[1] or key in result:
            raise ValueError('Missing or duplicate gold question ID')
        result[key] = record
    return result


def gold_subquestions(row):
    return [step['question'] for step in row.get('question_decomposition', [])]


def plan_subquestions(row):
    explicit = row.get('subquestions')
    if explicit is not None:
        if not isinstance(explicit, list) or not all(isinstance(s, str) for s in explicit):
            raise ValueError('subquestions must be a list of strings')
        return [s.strip() for s in explicit if s.strip()]
    plan = row.get('plan')
    if isinstance(plan, dict):
        return [step['sub_question'] for step in plan.get('steps', [])]
    return []
