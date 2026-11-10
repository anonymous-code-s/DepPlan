"""Rescore fixed MuSiQue plans over semantic-matching thresholds."""
import argparse
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import (gold_subquestions, load_gold, load_runs, new_output,
                     parse_config, plan_subquestions, question_key, seed_summary,
                     source_files, validate_seed_cohorts, write_csv, write_json,
                     write_jsonl)


def coverage_from_similarity(matrix, thresholds):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite predicted-by-gold similarity matrix')
    pred, gold = linear_sum_assignment(-matrix)
    matches = matrix[pred, gold]
    return ({str(t): float(np.count_nonzero(matches >= t) / matrix.shape[1]) for t in thresholds},
            [{'predicted_index': int(p), 'gold_index': int(g), 'cosine': float(matrix[p, g])}
             for p, g in zip(pred, gold)])


def score(rows, gold, thresholds, encoder):
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError('Thresholds must be distinct numbers in [0, 1]')
    records = []
    for row in rows:
        reference = gold[question_key(row)]
        if row.get('question') and row['question'].strip() != reference['question'].strip():
            raise ValueError('Prediction and gold question texts differ')
        expected = gold_subquestions(reference)
        predicted = plan_subquestions(row)
        if not expected:
            raise ValueError('MuSiQue gold decomposition is required')
        if predicted:
            vectors = encoder.encode(predicted + expected, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
            matrix = vectors[:len(predicted)] @ vectors[len(predicted):].T
        else:
            matrix = np.empty((0, len(expected)))
        values, assignments = coverage_from_similarity(matrix, thresholds)
        records.append(dict(method=row['method'], dataset=row['dataset'], seed=row['seed'],
                            example_id=row['example_id'], hops=len(expected),
                            predicted_steps=len(predicted), assignments=assignments, coverage=values))
    return records


def summarize(records, thresholds):
    groups = defaultdict(list)
    for row in records:
        for group in ['all', f"{row['hops']}-hop"]:
            groups[row['method'], row['dataset'], row['seed'], group].append(row)
    return [dict(method=m, dataset=d, seed=s, group=g, threshold=t, questions=len(values),
                 plans=sum(r['predicted_steps'] > 0 for r in values),
                 coverage_percent=100 * mean(r['coverage'][str(t)] for r in values))
            for (m, d, s, g), values in sorted(groups.items()) for t in thresholds]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', nargs='+', required=True)
    p.add_argument('--gold', required=True, help='Official MuSiQue JSONL, or the exact selected subset')
    p.add_argument('--thresholds', nargs='+', type=float, default=[.5, .6, .7, .8, .9])
    p.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = parse_config(p, argv)
    rows, gold = load_runs(args.predictions), load_gold(args.gold)
    validate_seed_cohorts(rows)
    groups = defaultdict(set)
    for row in rows:
        if row['dataset'] != 'musique':
            raise ValueError('Coverage scoring requires MuSiQue decompositions')
        groups[row['method'], row['seed']].add(question_key(row))
    if any(ids != set(gold) for ids in groups.values()):
        raise ValueError('Each run must cover exactly the gold IDs, including missing-plan records')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder, device=args.device)
    out = new_output(args.output)
    write_json(out / 'configuration.json', dict(encoder=args.encoder, thresholds=args.thresholds,
        sources=source_files([*args.predictions, args.gold]),
        matching='maximum-total-cosine assignment followed by thresholding',
        aggregation='mean per-question gold-step recall; missing plans score zero'))
    records = score(rows, gold, args.thresholds, encoder)
    per_seed = summarize(records, args.thresholds)
    write_jsonl(out / 'per_question.jsonl', records)
    write_csv(out / 'per_seed.csv', per_seed)
    write_csv(out / 'mean_sample_sd.csv', seed_summary(per_seed,
        ['method', 'dataset', 'group', 'threshold'], ['coverage_percent']))


if __name__ == '__main__':
    main()
