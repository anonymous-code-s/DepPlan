"""Evaluate saved MuSiQue plans, dependency order and intermediate answers.

No generation or retrieval is performed. See docs/plan_statistics.md for the
normalized baseline contract and the definitions used for unmatched steps.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import re
from statistics import mean

import numpy as np

from ..metrics import answer_scores, references_from_original
from .common import (load_gold, load_runs, new_output, question_key, seed_summary,
                     source_files, write_csv, write_json, write_jsonl)
from .coverage import coverage_from_similarity


SEEDS = (42, 123, 2026)
METRICS = ('coverage_percent', 'dependency_f1_percent',
           'ordering_accuracy_percent', 'intermediate_answer_f1_percent')
GOLD_REFERENCE = re.compile(r'(?<!\w)#(\d+)\b')


@dataclass
class PlanView:
    ids: list
    questions: list
    edges: set
    order: list
    answers: list
    native_available: bool = True


def _id(value):
    if type(value) is not int or value < 0:
        raise ValueError('Step IDs must be nonnegative integers')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty string')
    return value.strip()


def transitive_pairs(ids, edges):
    """Return all gold-constrained pairs, rejecting missing IDs and cycles."""
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate step IDs')
    children = {i: set() for i in ids}
    for parent, child in edges:
        if parent not in children or child not in children:
            raise ValueError('Dependency refers to an unknown step')
        children[parent].add(child)
    pairs = set()
    for origin in ids:
        pending = list(children[origin])
        reached = set()
        while pending:
            node = pending.pop()
            if node == origin:
                raise ValueError('Dependency graph contains a cycle')
            if node not in reached:
                reached.add(node)
                pending.extend(children[node])
        pairs.update((origin, target) for target in reached)
    return pairs


def gold_view(reference):
    steps = reference.get('question_decomposition')
    if not isinstance(steps, list) or not steps:
        raise ValueError('A nonempty MuSiQue question_decomposition is required')
    ids, questions, answers, edges = [], [], [], set()
    for step in steps:
        sid = _id(step.get('id'))
        question = _text(step.get('question'), 'Gold sub-question')
        if 'answer' not in step:
            raise ValueError('Every gold decomposition step requires its answer')
        answer = step['answer']
        if isinstance(answer, dict):
            _text(answer.get('text'), 'Gold step answer')
            aliases = answer.get('aliases', [])
        else:
            _text(answer, 'Gold step answer')
            aliases = step.get('answer_aliases', [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError('Gold answer aliases must be a list of strings')
        refs = references_from_original(step)
        if not any(s.strip() for s in refs):
            raise ValueError('Gold step answers must not be empty')
        ids.append(sid)
        questions.append(question)
        answers.append(refs)
        edges.update((int(parent), sid) for parent in GOLD_REFERENCE.findall(question))
    transitive_pairs(ids, edges)
    return PlanView(ids, questions, edges, list(ids), answers)


def prediction_view(row):
    """Read a DepPlan record or explicitly normalized baseline evaluation_plan.

    The Self-Ask follow_up events have an unambiguous question/answer sequence
    and are also accepted. Unstructured reasoning, passage chunks, and graph
    gist triples are never silently relabeled as sub-questions or answers.
    """
    if 'evaluation_plan' in row:
        payload = row['evaluation_plan']
    elif 'plan' in row:
        payload = row['plan']
    elif row.get('method') == 'self_ask' and isinstance(row.get('trace'), list):
        events = [e for e in row['trace'] if e.get('kind') == 'follow_up']
        if any(e.get('answer') is not None and not isinstance(e['answer'], str) for e in events):
            raise ValueError('A saved follow-up answer must be a string or null')
        return PlanView(list(range(len(events))),
                        [_text(e.get('question'), 'Follow-up question') for e in events],
                        set(), list(range(len(events))), [e.get('answer') for e in events], False)
    else:
        raise ValueError('Supply a DepPlan plan or a normalized evaluation_plan; '
                         'raw baseline reasoning cannot be inferred as sub-questions')
    if payload is None:
        return PlanView([], [], set(), [], [])
    if not isinstance(payload, dict) or not isinstance(payload.get('steps'), list):
        raise ValueError('evaluation_plan/plan must contain a steps list, or be null')
    steps = payload['steps']
    ids, questions, answers, edges = [], [], [], set()
    native = all('depends_on' in step for step in steps)
    for step in steps:
        sid = _id(step.get('id'))
        ids.append(sid)
        questions.append(_text(step.get('sub_question', step.get('question')), 'Sub-question'))
        answer = step.get('answer')
        if answer is not None and not isinstance(answer, str):
            raise ValueError('A step answer must be a string or null')
        answers.append(answer)
        dependencies = step.get('depends_on', [])
        if not isinstance(dependencies, list):
            raise ValueError('depends_on must be a list of integer IDs')
        edges.update((_id(parent), sid) for parent in dependencies)
    transitive_pairs(ids, edges)
    order = payload.get('execution_order')
    if not isinstance(order, list):
        raise ValueError('execution_order must contain every step ID exactly once')
    for sid in order:
        _id(sid)
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('execution_order must contain every step ID exactly once')
    # DepPlan stores answers separately from its frozen plan, keyed by step_id.
    trace = row.get('steps', []) if 'evaluation_plan' not in row else []
    if isinstance(trace, list):
        observed = set()
        for step in trace:
            sid = _id(step.get('step_id'))
            if sid not in ids or sid in observed:
                raise ValueError('Execution answers contain unknown or duplicate step IDs')
            observed.add(sid)
            answer = step.get('answer')
            if answer is not None and not isinstance(answer, str):
                raise ValueError('An execution answer must be a string or null')
            answers[ids.index(sid)] = answer
    return PlanView(ids, questions, edges, order, answers, native)


def evaluate_alignment(predicted, gold, matrix, *, threshold=.70, mode='sequence', failed=False):
    """Score one fixed alignment; all reported metrics are percentages."""
    if mode not in {'sequence', 'native'} or not 0 <= threshold <= 1:
        raise ValueError('Invalid dependency mode or similarity threshold')
    if mode == 'native' and not predicted.native_available:
        raise ValueError('Native evaluation requires explicitly recorded depends_on fields')
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(predicted.ids), len(gold.ids)):
        raise ValueError('Similarity dimensions do not match the predicted and gold steps')
    coverage, assignments = coverage_from_similarity(matrix, [threshold])
    accepted = [a for a in assignments if a['cosine'] >= threshold]
    mapping = {predicted.ids[a['predicted_index']]: gold.ids[a['gold_index']] for a in accepted}
    inverse = {g: p for p, g in mapping.items()}
    predicted_edges = (set(zip(predicted.order, predicted.order[1:]))
                       if mode == 'sequence' else predicted.edges)
    # Unaligned predicted endpoints are false positives, not discarded edges.
    true_edges = sum((mapping.get(u), mapping.get(v)) in gold.edges for u, v in predicted_edges)
    denom = len(predicted_edges) + len(gold.edges)
    dep_f1 = 2 * true_edges / denom if denom else 1.0
    constraints = transitive_pairs(gold.ids, gold.edges)
    positions = {sid: i for i, sid in enumerate(predicted.order)}
    consistent = sum(u in inverse and v in inverse and positions[inverse[u]] < positions[inverse[v]]
                     for u, v in constraints)
    order_accuracy = consistent / len(constraints) if constraints else 1.0
    step_f1 = [0.0] * len(gold.ids)
    if not failed:
        for assignment in accepted:
            pi, gi = assignment['predicted_index'], assignment['gold_index']
            answer = predicted.answers[pi]
            if isinstance(answer, str) and answer.strip():
                step_f1[gi] = answer_scores(gold.answers[gi], answer, dataset='musique')['f1']
    if not predicted.ids:
        dep_f1 = order_accuracy = 0.0
    return dict(dependency_mode=mode, coverage_percent=100 * coverage[str(threshold)],
                dependency_f1_percent=100 * dep_f1, ordering_accuracy_percent=100 * order_accuracy,
                intermediate_answer_f1_percent=100 * mean(step_f1),
                assignments=assignments, accepted_matches=len(accepted),
                predicted_edges=[list(e) for e in sorted(predicted_edges)],
                gold_edges=[list(e) for e in sorted(gold.edges)],
                gold_order_pairs=len(constraints), correct_order_pairs=consistent,
                gold_step_answer_f1=step_f1)


def validate_cohorts(rows, gold, seeds=SEEDS):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Expected seeds must be distinct nonnegative integers below 2**32')
    groups = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row.get('dataset') != 'musique' or not row.get('example_id'):
            raise ValueError('MuSiQue dataset and a nonempty example_id are required')
        if not isinstance(row.get('method'), str) or not row['method']:
            raise ValueError('Every record requires its method')
        if type(row.get('seed')) is not int or row['seed'] not in seeds:
            raise ValueError('Unexpected or missing seed')
        if row.get('status') not in {'success', 'error', 'failed', 'graph_failed'}:
            raise ValueError('Record status must identify success or failure')
        key = question_key(row)
        group = groups[row['method']][row['seed']]
        if key in group:
            raise ValueError('Duplicate method/seed/question record')
        group.add(key)
        if key not in gold:
            raise ValueError('Prediction question is absent from the supplied gold cohort')
        if row.get('question') is not None and row['question'].strip() != gold[key]['question'].strip():
            raise ValueError('Prediction and gold question texts differ')
    if not groups or not gold:
        raise ValueError('Predictions and gold cohort must be nonempty')
    for method, runs in groups.items():
        if set(runs) != set(seeds) or any(ids != set(gold) for ids in runs.values()):
            raise ValueError('Every method/seed must contain exactly the gold IDs, including failures: ' + method)


def score(rows, gold, encoder, *, threshold=.70, modes=('sequence',), seeds=SEEDS):
    validate_cohorts(rows, gold, seeds)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {'sequence', 'native'}:
        raise ValueError('Choose sequence, native, or both dependency modes')
    records = []
    for row in rows:
        reference = gold_view(gold[question_key(row)])
        predicted = prediction_view(row)
        if predicted.questions:
            vectors = np.asarray(encoder.encode(predicted.questions + reference.questions,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
            matrix = vectors[:len(predicted.ids)] @ vectors[len(predicted.ids):].T
        else:
            matrix = np.empty((0, len(reference.ids)))
        for mode in modes:
            result = evaluate_alignment(predicted, reference, matrix, threshold=threshold, mode=mode,
                                        failed=row['status'] != 'success')
            records.append(dict(method=row['method'], dataset=row['dataset'], example_id=row['example_id'],
                seed=row['seed'], status=row['status'], hops=len(reference.ids),
                predicted_steps=len(predicted.ids), threshold=threshold, **result))
    return records


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        for group in ('all', str(row['hops']) + '-hop'):
            groups[row['method'], row['dataset'], row['seed'], row['dependency_mode'], group].append(row)
    return [dict(method=m, dataset=d, seed=s, dependency_mode=mode, group=group,
                 questions=len(values), failed_questions=sum(r['status'] != 'success' for r in values),
                 **{metric: mean(r[metric] for r in values) for metric in METRICS})
            for (m, d, s, mode, group), values in sorted(groups.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dependency-mode', choices=['sequence', 'native', 'both'], default='sequence')
    parser.add_argument('--threshold', type=float, default=.70)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be in [0, 1]')
    rows, gold = load_runs(args.predictions), load_gold(args.gold)
    validate_cohorts(rows, gold, args.seeds)
    modes = ('sequence', 'native') if args.dependency_mode == 'both' else (args.dependency_mode,)
    # Validate structures before loading a local scoring model.
    for reference in gold.values():
        gold_view(reference)
    for row in rows:
        view = prediction_view(row)
        if 'native' in modes and not view.native_available:
            raise ValueError('Native evaluation requires an explicit dependency graph')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder, device=args.device)
    records = score(rows, gold, encoder, threshold=args.threshold, modes=modes, seeds=args.seeds)
    out = new_output(args.output)
    write_json(out / 'configuration.json', dict(encoder=args.encoder, threshold=args.threshold,
        seeds=args.seeds, dependency_modes=modes, sources=source_files([*args.predictions, args.gold]),
        alignment='maximum-total-cosine one-to-one assignment, followed by thresholding',
        ordering='all transitively gold-constrained pairs; unmatched endpoints count incorrect',
        dependency_f1='2 * aligned correct edges / (all predicted edges + all gold edges)',
        aggregation='question mean, then mean/sample SD across seeds; intermediate failures score zero'))
    write_jsonl(out / 'per_question.jsonl', records)
    per_seed = summarize(records)
    write_csv(out / 'per_seed.csv', per_seed)
    write_csv(out / 'mean_sample_sd.csv', seed_summary(per_seed,
        ['method', 'dataset', 'dependency_mode', 'group'], METRICS))


if __name__ == '__main__':
    main()
