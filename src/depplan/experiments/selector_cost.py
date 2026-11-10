"""Measure CPU GCN, direct-Qwen and majority selection on a fixed cohort."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import time
from urllib.parse import urlsplit

import numpy as np
import torch

from ..classifier.runtime import Classifier
from ..llm import CallUsage, LLMClient, sum_usage
from ..plan import extract_json_object
from .ablations.runner import code_digest, file_digest
from .ablations.variants import DIRECT_STRATEGY_SYSTEM, LABELS, training_majority
from .common import (check_grid, digest, new_output, parse_config, question_key,
                     read_jsonl, seed_summary, service_arguments, source_files,
                     write_csv, write_json, write_jsonl)


DATASETS = ('hotpot', 'musique', '2wiki')
SEEDS = (42, 123, 2026)
SELECTORS = ('gcn', 'direct_qwen', 'majority')
METRICS = ('mean_selection_seconds', 'mean_llm_calls', 'mean_logical_requests',
           'mean_prompt_tokens', 'mean_completion_tokens', 'mean_total_tokens',
           'mean_known_prompt_tokens', 'mean_known_completion_tokens',
           'mean_known_total_tokens', 'failed_questions', 'questions_with_missing_usage')


def input_view(row):
    """Retain the selector input, excluding answer/evidence/scoring labels."""
    dataset, qid = question_key(row)
    return {'dataset': dataset, 'example_id': qid, 'question': row['question'],
            'structure_source': row['structure_source'],
            'status': row.get('status', 'success'), 'amr_triples': row.get('amr_triples'),
            'source_split': row.get('source_split')}


def validate_cohort(rows, structure_source, questions_per_dataset=200, training_keys=()):
    if type(questions_per_dataset) is not int or questions_per_dataset < 1:
        raise ValueError('questions_per_dataset must be a positive integer')
    keys, counts = set(), Counter()
    normalized_training = {question_key({'dataset': d, 'example_id': q})
                           for d, q in training_keys}
    for row in rows:
        if row.get('dataset') not in DATASETS:
            raise ValueError('Every cohort question needs a supported dataset')
        if not isinstance(row.get('example_id'), str) or not row['example_id'].strip():
            raise ValueError('Every cohort question needs a nonempty string example_id')
        key = question_key(row)
        if not key[1] or key in keys:
            raise ValueError('Missing or duplicate cohort question ID: ' + str(key))
        if key in normalized_training:
            raise ValueError('Development cohort overlaps the training manifest: ' + str(key))
        keys.add(key)
        counts[key[0]] += 1
        if row.get('source_split') not in (None, 'dev', 'development', 'validation'):
            raise ValueError('Selector costs require a development-set cohort')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError('Every cohort question needs question text')
        if row.get('structure_source') != structure_source:
            raise ValueError('Question graph source differs from the classifier checkpoint')
        status = row.get('status', 'success')
        if status not in ('success', 'graph_failed'):
            raise ValueError('Graph status must be success or graph_failed')
        if status == 'success':
            triples = row.get('amr_triples')
            if (not isinstance(triples, list) or not triples
                    or any(not isinstance(t, (tuple, list)) or len(t) != 3
                           or not all(isinstance(v, str) and v for v in t) for t in triples)
                    or not any(t[1] == ':instance' for t in triples)):
                raise ValueError('A successful question needs cached AMR triples with concepts')
    expected = {d: questions_per_dataset for d in DATASETS}
    if dict(counts) != expected:
        raise ValueError(f'Expected {questions_per_dataset} questions from each dataset; got {dict(counts)}')
    return [input_view(row) for row in rows]


def configuration(args, rows, majority_info, checkpoint_config):
    weights = Path(args.checkpoint) / 'model.safetensors'
    weights_hash = file_digest(weights)
    if (checkpoint_config.get('weights_sha256') is not None
            and checkpoint_config['weights_sha256'] != weights_hash):
        raise ValueError('Checkpoint weights differ from their configured SHA-256')
    if tuple(checkpoint_config.get('labels', ())) != LABELS:
        raise ValueError('Checkpoint label order differs from the strategy labels')
    endpoint = urlsplit(args.base_url)
    inputs = list(args.input) + [args.training_manifest]
    if getattr(args, 'config', None):
        inputs.append(args.config)
    result = {
        'experiment': 'online_selector_cost', 'schema_version': 1,
        'selectors': list(SELECTORS), 'seeds': list(args.seeds),
        'questions_per_dataset': args.questions_per_dataset, 'questions': len(rows),
        'question_keys': [list(question_key(r)) for r in rows],
        'input_sha256': digest(rows), 'input_files': source_files(inputs),
        'checkpoint_sha256': weights_hash,
        'classifier_config_sha256': file_digest(Path(args.checkpoint) / 'config.json'),
        'code_sha256': code_digest(),
        'structure_source': checkpoint_config['structure_source'],
        'encoder': args.encoder or checkpoint_config['encoder'],
        'classifier_device': 'cpu', 'cpu_threads': torch.get_num_threads(),
        'model': args.model,
        'deployment': endpoint.scheme + '://' + endpoint.netloc.split('@')[-1] + endpoint.path,
        'selector_temperature': 0.0, 'selector_max_tokens': args.max_tokens,
        'top_p': 1.0, 'top_k': -1, 'api_attempts': 3, 'timeout': args.timeout,
        'direct_prompt_sha256': hashlib.sha256(DIRECT_STRATEGY_SYSTEM.encode('utf-8')).hexdigest(),
        'majority': majority_info,
        'time_scope': ('per-question selection and output validation, including node encoding, '
                       'GCN inference or API retries; excludes model loading, AMR parsing, '
                       'answer scoring and result-file serialization'),
        'gcn_features': 'encoded afresh for each question in every seed; no feature cache',
        'direct_llm_requests_per_question': 1,
        'llm_call_count': 'physical API attempts, including failed attempts',
        'runs': len(SELECTORS) * len(args.seeds),
        'question_runs': len(rows) * len(SELECTORS) * len(args.seeds),
    }
    result['configuration_sha256'] = digest(result)
    return result


async def measure_question(row, selector, seed, config, *, classifier=None,
                           client=None, majority=None):
    if selector not in SELECTORS:
        raise ValueError('Unknown selector')
    result = {
        'dataset': row['dataset'], 'example_id': row['example_id'],
        'question': row['question'], 'selector': selector, 'method': selector, 'seed': seed,
        'input_sha256': digest(input_view(row)), 'configuration_sha256': config['configuration_sha256'],
        'checkpoint_sha256': config['checkpoint_sha256'], 'code_sha256': config['code_sha256'],
        'strategy': None, 'status': 'error', 'logical_requests': 0,
    }
    graph = {k: row[k] for k in ('dataset', 'example_id', 'structure_source', 'status', 'amr_triples')}
    if selector == 'direct_qwen':
        if client is None:
            raise ValueError('The direct selector requires its logged LLM client')
        client.example_id = row['dataset'] + ':' + row['example_id']
        client.records.clear()
    started = time.perf_counter()
    try:
        if selector == 'gcn':
            if row['status'] != 'success':
                raise ValueError('Cached AMR preprocessing failed for this question')
            prediction = classifier.predict([graph], batch_size=1)[0]
            strategy = prediction['predicted_strategy']
        elif selector == 'direct_qwen':
            result['logical_requests'] = 1
            response = await client.complete(role='strategy_selection', system=DIRECT_STRATEGY_SYSTEM,
                user='Question:\n' + row['question'], temperature=0.0,
                max_tokens=config['selector_max_tokens'])
            strategy = extract_json_object(response.text).get('strategy')
        else:
            strategy = majority
        if strategy not in LABELS:
            raise ValueError('Selector returned an invalid strategy')
        result.update(strategy=strategy, status='success')
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    result['selection_seconds'] = time.perf_counter() - started
    attempts = ([CallUsage(**r['usage']) for r in client.records]
                if selector == 'direct_qwen' else [])
    result['usage'] = sum_usage(attempts)
    return result


def summarize(rows, selector, seed):
    summaries = []
    for dataset in ('all', *DATASETS):
        selected = rows if dataset == 'all' else [r for r in rows if r['dataset'] == dataset]
        if not selected:
            continue
        item = {'selector': selector, 'dataset': dataset, 'seed': seed,
                'questions': len(selected),
                'failed_questions': sum(r['status'] != 'success' for r in selected),
                'mean_selection_seconds': mean(r['selection_seconds'] for r in selected),
                'mean_llm_calls': mean(r['usage']['call_count'] for r in selected),
                'mean_logical_requests': mean(r['logical_requests'] for r in selected),
                'questions_with_missing_usage': sum(any(r['usage'][field] is None
                    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')) for r in selected)}
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            values = [r['usage'][field] for r in selected]
            item['mean_' + field] = mean(values) if all(v is not None for v in values) else None
            item['mean_known_' + field] = mean(r['usage']['known_' + field] for r in selected)
        summaries.append(item)
    return summaries


async def run(args):
    check_grid(args.seeds, 'seeds', SEEDS)
    if set(args.seeds) != set(SEEDS):
        raise ValueError('This comparison requires all three seeds: 42, 123 and 2026')
    if args.device != 'cpu':
        raise ValueError('Online selector costs use CPU node encoding and GCN inference')
    if (type(args.max_tokens) is not int or args.max_tokens < 1
            or isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float))
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError('Positive token budget and finite positive timeout are required')
    checkpoint_config = json.loads((Path(args.checkpoint) / 'config.json').read_text())
    majority, counts, training_keys = training_majority(args.training_manifest)
    rows = validate_cohort([r for path in args.input for r in read_jsonl(path)],
                          checkpoint_config['structure_source'], args.questions_per_dataset, training_keys)
    majority_info = {'strategy': majority, 'training_counts': counts,
                     'training_manifest_sha256': file_digest(args.training_manifest)}
    config = configuration(args, rows, majority_info, checkpoint_config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return config
    output = new_output(args.output)
    write_json(output / 'configuration.json', config)
    write_jsonl(output / 'cohort.jsonl', rows)
    # Both the GCN and the encoder are loaded before any per-question timer.
    classifier = Classifier(args.checkpoint, config['encoder'], device='cpu')
    _ = classifier.encoder
    reports = []
    for selector in SELECTORS:
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cell = output / selector / f'seed_{seed}'
            cell.mkdir(parents=True)
            cell_config = dict(config, selector=selector, seed=seed,
                               experiment_configuration_sha256=config['configuration_sha256'])
            del cell_config['configuration_sha256']
            cell_config['configuration_sha256'] = digest(cell_config)
            write_json(cell / 'configuration.json', cell_config)
            client = (LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key,
                               seed=seed, max_attempts=3, timeout=args.timeout,
                               log_path=cell / 'calls.jsonl') if selector == 'direct_qwen' else None)
            results = []
            try:
                with (cell / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    for row in rows:
                        result = await measure_question(row, selector, seed, cell_config,
                            classifier=classifier, client=client, majority=majority)
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        results.append(result)
            finally:
                if client is not None:
                    await client.close()
            summaries = summarize(results, selector, seed)
            reports.extend(summaries)
            write_json(cell / 'summary.json', summaries)
    aggregate = seed_summary(reports, ['selector', 'dataset'], METRICS)
    write_csv(output / 'per_seed.csv', reports)
    write_csv(output / 'mean_sample_sd.csv', aggregate)
    write_json(output / 'summary.json', {'per_seed': reports, 'mean_sample_sd': aggregate})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True,
                        help='Fixed development cohort with questions and source-matched cached AMRs')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-manifest', default='data/classifier/train_manifest.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--questions-per-dataset', type=int, default=200,
                        help='Expected cohort size per dataset; override explicitly for smoke checks')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--encoder', help='BGE model ID or prepared local encoder directory')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--dry-run', action='store_true')
    service_arguments(parser)
    return asyncio.run(run(parse_config(parser, argv)))


if __name__ == '__main__':
    main()
