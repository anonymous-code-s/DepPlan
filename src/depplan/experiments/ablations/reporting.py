"""Derive integer EM counts and seed summaries from saved predictions."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from ...evaluation import semantic_summary


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize_records(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty run')
    keys = [(r['dataset'], r['example_id']) for r in rows]
    if len(set(keys)) != len(rows):
        raise ValueError('Duplicate prediction IDs')
    for r in rows:
        if r['em'] not in (0, 1) or not math.isfinite(r['f1']) or not 0 <= r['f1'] <= 1:
            raise ValueError('EM must be binary and F1 must be in [0, 1]')
        if r['status'] != 'success' and (r['em'] or r['f1']):
            raise ValueError('Failed questions must have zero EM and F1')
    n = len(rows)
    total = [r['usage']['total_tokens'] for r in rows]
    def measured_average(field):
        values = [r.get(field) for r in rows]
        return mean(values) if all(v is not None and math.isfinite(v) for v in values) else None
    judge_tokens = [r.get('judge_usage', {}).get('total_tokens') for r in rows]
    return {
        **semantic_summary(rows),
        'questions': n,
        'successful_questions': sum(r['status'] == 'success' for r in rows),
        'failed_questions': sum(r['status'] != 'success' for r in rows),
        'em_correct': sum(int(r['em']) for r in rows),
        'em_percent': 100 * sum(int(r['em']) for r in rows) / n,
        'f1_percent': 100 * mean(r['f1'] for r in rows),
        'mean_online_seconds': mean(r['online_seconds'] for r in rows),
        'mean_amr_seconds': measured_average('amr_seconds'),
        'mean_total_seconds': measured_average('total_seconds'),
        'questions_with_missing_amr_timing': sum(r.get('amr_seconds') is None for r in rows),
        'mean_strategy_selection_seconds': mean(r.get('strategy_selection_seconds', 0.0) for r in rows),
        'mean_plan_construction_seconds': mean(r.get('plan_construction_seconds', 0.0) for r in rows),
        'mean_plan_execution_seconds': mean(r.get('plan_execution_seconds', 0.0) for r in rows),
        'mean_online_overhead_seconds': measured_average('online_overhead_seconds'),
        'mean_scoring_seconds': measured_average('scoring_seconds'),
        'mean_judge_tokens': mean(judge_tokens) if all(v is not None for v in judge_tokens) else None,
        'mean_steps': mean(len(r.get('steps', [])) for r in rows),
        'mean_llm_calls': mean(r['usage']['call_count'] for r in rows),
        'mean_total_tokens': mean(total) if all(t is not None for t in total) else None,
        'mean_known_total_tokens': mean(r['usage']['known_total_tokens'] for r in rows),
        'questions_with_missing_usage': sum(t is None for t in total),
    }


def _write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(root, output):
    root, output = Path(root), Path(output)
    per_seed, cells, identities = [], {}, {}
    protocols = {}
    for file in sorted(root.rglob('predictions.jsonl')):
        config_file = file.parent / 'configuration.json'
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text())
        if config.get('experiment') != 'component_ablations':
            continue
        rows = read_jsonl(file)
        if len(rows) != config['questions']:
            raise ValueError(f'Incomplete run: {file.parent}')
        expected_keys = [tuple(key) for key in config['question_keys']]
        actual_keys = [(r['dataset'], r['example_id']) for r in rows]
        if actual_keys != expected_keys:
            raise ValueError(f'Prediction order or IDs differ from configuration: {file.parent}')
        for row in rows:
            if row['seed'] != config['seed'] or row['variant'] != config['variant']['name']:
                raise ValueError('Prediction seed or variant differs from configuration')
        datasets = sorted({r['dataset'] for r in rows})
        protocol = {k: config[k] for k in ['settings', 'checkpoint_sha256', 'encoder',
                                          'model', 'deployment', 'code_sha256', 'device',
                                          'classifier_config_sha256', 'structure_source']}
        protocol['orchestration'] = config.get('orchestration')
        protocol['semantic_evaluator'] = config.get('semantic_evaluator')
        for dataset in datasets:
            selected = [r for r in rows if r['dataset'] == dataset]
            identity = [(r['example_id'], r['input_sha256']) for r in selected]
            if dataset in identities and identities[dataset] != identity:
                raise ValueError(f'Question inputs differ across conditions for {dataset}')
            identities[dataset] = identity
            if dataset in protocols and protocols[dataset] != protocol:
                raise ValueError(f'Shared run settings differ for {dataset}')
            protocols[dataset] = protocol
            item = dict(variant=config['variant']['name'], dataset=dataset, seed=config['seed'],
                        **summarize_records(selected))
            cell = item['variant'], dataset
            if any(r['seed'] == item['seed'] for r in cells.get(cell, [])):
                raise ValueError(f'Duplicate seed for {cell}')
            cells.setdefault(cell, []).append(item)
            per_seed.append(item)
    if not per_seed:
        raise ValueError('No completed component-ablation runs found')
    seed_sets = {tuple(sorted(r['seed'] for r in values)) for values in cells.values()}
    if len(seed_sets) != 1:
        raise ValueError('Conditions have different completed seed sets')
    metric_names = ['em_percent', 'f1_percent', 'acc_percent', 'mean_online_seconds', 'mean_steps',
                    'mean_llm_calls', 'mean_total_tokens', 'mean_known_total_tokens',
                    'mean_total_seconds', 'mean_amr_seconds', 'mean_strategy_selection_seconds',
                    'mean_plan_construction_seconds', 'mean_plan_execution_seconds',
                    'mean_online_overhead_seconds', 'mean_scoring_seconds', 'mean_judge_tokens']
    aggregate = []
    for (variant, dataset), values in sorted(cells.items()):
        item = dict(variant=variant, dataset=dataset, seeds=len(values),
                    questions_per_seed=values[0]['questions'])
        for metric in metric_names:
            numbers = [v[metric] for v in values]
            complete = all(x is not None for x in numbers)
            item[metric + '_mean'] = mean(numbers) if complete else None
            item[metric + '_sample_sd'] = stdev(numbers) if complete and len(numbers) > 1 else None
        aggregate.append(item)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'per_seed.csv', sorted(per_seed, key=lambda r: (r['variant'], r['dataset'], r['seed'])))
    _write_csv(output / 'mean_sample_sd.csv', aggregate)
    write_json(output / 'summary.json', {'decoding_seeds': list(next(iter(seed_sets))),
                                        'per_seed': per_seed, 'aggregates': aggregate})
    return aggregate
