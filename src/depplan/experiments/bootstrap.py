"""Nested paired bootstrap of saved question-level observations.

Each invocation evaluates one complete Bonferroni family. Aggregate per-seed
CSV files are not accepted as substitutes for paired question observations.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import (new_output, question_key, read_jsonl, source_files,
                     write_csv, write_json)


SEEDS = (42, 123, 2026)
COMPARISON_FIELDS = {'name', 'method_a', 'method_b', 'dataset', 'metric',
                     'metric_a', 'metric_b', 'scale', 'where_a', 'where_b',
                     'weight_a', 'weight_b'}


def _number(value, label):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(label + ' must be a finite number; missing observations cannot be imputed')
    return float(value)


def _field(row, path):
    if not isinstance(path, str) or not path:
        raise ValueError('Metric/weight field paths must be nonempty strings')
    value = row
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            raise ValueError('Missing field: ' + path)
        value = value[part]
    return value


def _validate_seeds(seeds):
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Nested paired bootstrap requires exactly three distinct decoding seeds')


def load_observations(paths):
    """Allow different representations, but reject duplicates after selection."""
    rows = []
    for path in paths:
        for original in read_jsonl(path):
            if not isinstance(original, dict):
                raise ValueError('Each JSONL observation must be an object')
            row = dict(original)
            row['method'] = row.get('method', row.get('variant'))
            if row['method'] == 'full':
                row['method'] = 'depplan'
            if (not isinstance(row['method'], str) or not row['method']
                    or row.get('dataset') not in {'hotpot', 'musique', '2wiki'}
                    or not row.get('example_id') or type(row.get('seed')) is not int):
                raise ValueError('Question-level method, dataset, example_id and integer seed are required')
            rows.append(row)
    if not rows:
        raise ValueError('No question-level observations were supplied')
    return rows


def paired_arrays(rows, comparison, seeds=SEEDS):
    """Align both methods by (dataset, question ID) within each matched seed."""
    _validate_seeds(seeds)
    if not isinstance(comparison, dict) or set(comparison) - COMPARISON_FIELDS:
        raise ValueError('Unknown comparison configuration fields')
    for name in ('name', 'method_a', 'method_b'):
        if not isinstance(comparison.get(name), str) or not comparison[name]:
            raise ValueError('A comparison requires a nonempty ' + name)
    dataset = comparison.get('dataset')
    if dataset not in {'hotpot', 'musique', '2wiki', 'all'}:
        raise ValueError('Comparison dataset must be hotpot, musique, 2wiki or all')
    scale = _number(comparison.get('scale', 1.0), 'scale')
    if scale <= 0:
        raise ValueError('scale must be positive')
    sides, weight_sides, cohort, question_texts = [], [], None, {}
    for suffix in ('a', 'b'):
        method = comparison['method_' + suffix]
        metric = comparison.get('metric_' + suffix, comparison.get('metric'))
        if not isinstance(metric, str) or not metric:
            raise ValueError('Specify metric or both metric_a and metric_b')
        weight_path = comparison.get('weight_' + suffix)
        where = comparison.get('where_' + suffix, {})
        if not isinstance(where, dict):
            raise ValueError('where_a/where_b must be objects of exact field-value filters')
        index = defaultdict(dict)
        for row in rows:
            if row.get('method', row.get('variant')) != method:
                continue
            if dataset != 'all' and row.get('dataset') != dataset:
                continue
            if not all(_field(row, field) == expected for field, expected in where.items()):
                continue
            seed = row.get('seed')
            if type(seed) is not int or seed not in seeds:
                raise ValueError('Unexpected or missing decoding seed for ' + method)
            if row.get('dataset') not in {'hotpot', 'musique', '2wiki'} or not row.get('example_id'):
                raise ValueError('Each observation needs a dataset and nonempty question ID')
            key = question_key(row)
            if row.get('question') is not None:
                if not isinstance(row['question'], str) or not row['question'].strip():
                    raise ValueError('A supplied question must be nonempty text')
                text = row['question'].strip()
                if key in question_texts and text != question_texts[key]:
                    raise ValueError('Question text differs across paired observations')
                question_texts[key] = text
            if key in index[seed]:
                raise ValueError('Duplicate question observation; select one representation with where filters')
            weight = _number(_field(row, weight_path), weight_path) if weight_path else 1.0
            if weight < 0:
                raise ValueError('Weights must be nonnegative')
            value = _field(row, metric)
            # A question with no generated sub-questions has an explicit zero
            # weight and an unavailable quality mean. Keep its sampling slot.
            value = 0.0 if value is None and weight == 0 else _number(value, metric)
            index[seed][key] = (value, weight)
        if set(index) != set(seeds):
            raise ValueError('Missing one or more complete runs for ' + method)
        for seed in seeds:
            ids = set(index[seed])
            if cohort is None:
                cohort = ids
            if not ids or ids != cohort:
                raise ValueError('Question IDs differ between methods or decoding seeds')
        keys = sorted(cohort)
        sides.append(np.asarray([[index[s][key][0] for key in keys] for s in seeds], dtype=float))
        weight_sides.append(np.asarray([[index[s][key][1] for key in keys] for s in seeds], dtype=float))
    return sides[0], sides[1], weight_sides[0], weight_sides[1], sorted(cohort)


def _weighted_mean(values, weights):
    denom = weights.sum()
    if denom <= 0:
        raise ValueError('A run or bootstrap sample has zero total weight; the statistic is undefined')
    return float(np.dot(values, weights) / denom)


def nested_paired_bootstrap(a, b, *, replicates=10000, seed=2026,
                           weights_a=None, weights_b=None):
    """Return observed difference, percentile interval, +1 p, and replicates.

    a and b are [3 runs, N matched questions]. The same question indices are
    used for both methods for each sampled run pair. Repeated selections of a
    run pair receive independent inner resamples.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != 3 or a.shape[1] == 0 or b.shape != a.shape:
        raise ValueError('Expected two equally shaped [3, N>0] paired arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing or non-finite paired observations')
    if type(replicates) is not int or replicates < 1:
        raise ValueError('replicates must be a positive integer')
    wa = np.ones_like(a) if weights_a is None else np.asarray(weights_a, dtype=float)
    wb = np.ones_like(b) if weights_b is None else np.asarray(weights_b, dtype=float)
    for weights in (wa, wb):
        if weights.shape != a.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Weights must have the same shape and be finite nonnegative numbers')
        if np.any(weights.sum(axis=1) <= 0):
            raise ValueError('Each run requires positive total weight')
    point = float(np.mean([_weighted_mean(a[i], wa[i]) - _weighted_mean(b[i], wb[i]) for i in range(3)]))
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    n = a.shape[1]
    unit_weights = np.all(wa == 1) and np.all(wb == 1)
    paired_difference = a - b
    for replicate in range(replicates):
        run_pairs = rng.integers(0, 3, size=3)
        estimates = []
        for pair in run_pairs:
            indices = rng.integers(0, n, size=n)
            if unit_weights:
                estimates.append(float(paired_difference[pair, indices].mean()))
            else:
                estimates.append(_weighted_mean(a[pair, indices], wa[pair, indices])
                                 - _weighted_mean(b[pair, indices], wb[pair, indices]))
        differences[replicate] = np.mean(estimates)
    at_or_below_zero = int(np.count_nonzero(differences <= 0))
    at_or_above_zero = int(np.count_nonzero(differences >= 0))
    smaller_tail = min(at_or_below_zero, at_or_above_zero)
    interval = np.percentile(differences, [2.5, 97.5], method='linear')
    return dict(delta=point, ci_low=float(interval[0]), ci_high=float(interval[1]),
                p_raw=min(1.0, 2 * (smaller_tail + 1) / (replicates + 1)),
                at_or_below_zero=at_or_below_zero, at_or_above_zero=at_or_above_zero,
                smaller_tail=smaller_tail, replicates=replicates), differences


def evaluate_family(rows, comparisons, *, seeds=SEEDS, replicates=10000, bootstrap_seed=2026):
    _validate_seeds(seeds)
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError('bootstrap_seed must be an integer in [0, 2**32)')
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError('Supply the complete nonempty comparison family')
    names = [c.get('name') for c in comparisons if isinstance(c, dict)]
    if len(names) != len(comparisons) or len(set(names)) != len(names):
        raise ValueError('Comparison names must be unique')
    # Fail before computing any tests if another family member has missing data.
    paired = [paired_arrays(rows, comparison, seeds) for comparison in comparisons]
    results, samples = [], {}
    for i, (comparison, (a, b, wa, wb, keys)) in enumerate(zip(comparisons, paired)):
        random_seed = int(np.random.SeedSequence([bootstrap_seed, i]).generate_state(1)[0])
        result, differences = nested_paired_bootstrap(a, b, replicates=replicates, seed=random_seed,
                                                     weights_a=wa, weights_b=wb)
        scale = float(comparison.get('scale', 1))
        for field in ('delta', 'ci_low', 'ci_high'):
            result[field] *= scale
        result.update(name=comparison['name'], method_a=comparison['method_a'],
                      method_b=comparison['method_b'], dataset=comparison['dataset'],
                      metric_a=comparison.get('metric_a', comparison.get('metric')),
                      metric_b=comparison.get('metric_b', comparison.get('metric')),
                      scale=scale, questions_per_seed=len(keys), seeds=list(seeds),
                      bootstrap_seed=random_seed, family_size=len(comparisons),
                      p_adjusted=min(1.0, len(comparisons) * result['p_raw']))
        result['significant_0_05'] = result['p_adjusted'] < .05
        results.append(result)
        samples['comparison_' + str(i)] = differences * scale
    return results, samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='Saved per-question JSONL observations')
    parser.add_argument('--family', required=True, help='JSON object with name and comparisons list')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs=3, type=int, default=list(SEEDS))
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    args = parser.parse_args(argv)
    family = json.loads(Path(args.family).read_text(encoding='utf-8'))
    if (not isinstance(family, dict) or set(family) != {'name', 'comparisons'}
            or not isinstance(family['name'], str) or not family['name']):
        parser.error('Family JSON must contain exactly a nonempty name and comparisons')
    rows = load_observations(args.input)
    results, samples = evaluate_family(rows, family['comparisons'], seeds=args.seeds,
        replicates=args.replicates, bootstrap_seed=args.bootstrap_seed)
    out = new_output(args.output)
    write_json(out / 'configuration.json', dict(family=family, decoding_seeds=args.seeds,
        replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
        sources=source_files([*args.input, args.family]),
        resampling='three matched run pairs with replacement; shared question indices within each pair',
        intervals='marginal 95% percentile; NumPy linear quantiles',
        p_value='min(1, 2 * (min(count<=0, count>=0) + 1) / (B + 1))',
        adjustment='Bonferroni over every comparison in the supplied family'))
    write_json(out / 'comparisons.json', results)
    write_csv(out / 'comparisons.csv', results)
    np.savez_compressed(out / 'bootstrap_differences.npz', **samples)


if __name__ == '__main__':
    main()
