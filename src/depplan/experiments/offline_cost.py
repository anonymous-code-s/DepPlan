"""Measure silver-label generation and retraining on a frozen training pool."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from ..llm import CallUsage, LLMClient, sum_usage
from ..plan import extract_json_object
from .common import (digest, new_output, parse_config, question_key, read_jsonl,
                     service_arguments, source_files, write_json, write_jsonl)


LABELS = {'sequential', 'parallel', 'hierarchical'}


def validate_training(rows, manifest):
    if len(rows) != len(manifest) or not rows:
        raise ValueError('Training graphs must cover the entire supplied manifest')
    expected = {question_key(r): r['silver_strategy'] for r in manifest}
    actual = {question_key(r): r.get('silver_strategy') for r in rows}
    if len(expected) != len(manifest) or len(actual) != len(rows) or expected != actual:
        raise ValueError('Training IDs or original silver labels differ from the manifest')
    if set(actual.values()) - LABELS:
        raise ValueError('Unknown silver strategy label')
    if any(r.get('status', 'success') != 'success' or not r.get('amr_triples') for r in rows):
        raise ValueError('Cost measurement requires valid cached graphs for the whole pool')
    if len({r['structure_source'] for r in rows}) != 1:
        raise ValueError('Train one graph-source classifier at a time')


def amr_text(row):
    if isinstance(row.get('raw_amr'), str) and row['raw_amr'].strip():
        return row['raw_amr']
    import penman
    if not row.get('amr_root'):
        raise ValueError('Cached graphs need raw_amr or amr_root for PENMAN rendering')
    return penman.encode(penman.Graph([tuple(t) for t in row['amr_triples']], top=row['amr_root']))


async def measure_labels(rows, amrs, prompt, args, output):
    clients = [LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key,
                         seed=args.seed, max_attempts=3, timeout=args.timeout,
                         log_path=output / f'calls_worker_{i}.jsonl') for i in range(args.workers)]
    results = [None] * len(rows)
    queue = asyncio.Queue()
    for i in range(len(rows)):
        queue.put_nowait(i)

    async def worker(client):
        while not queue.empty():
            i = queue.get_nowait()
            row = rows[i]
            client.example_id = row['dataset'] + ':' + row['example_id']
            client.records.clear()
            started = time.perf_counter()
            result = dict(dataset=row['dataset'], example_id=row['example_id'],
                          original_silver_strategy=row['silver_strategy'], strategy=None, status='error')
            try:
                response = await client.complete(role='silver_strategy', system=prompt,
                    user=f"Question:\n{row['question']}\n\nAMR:\n{amrs[i]}",
                    temperature=0.0, max_tokens=args.max_tokens)
                strategy = extract_json_object(response.text).get('strategy')
                if strategy not in LABELS:
                    raise ValueError('Teacher returned an unknown strategy')
                result.update(strategy=strategy, status='success')
            except Exception as exc:
                result['error'] = type(exc).__name__ + ': ' + str(exc)
            result['seconds'] = time.perf_counter() - started
            result['usage'] = sum_usage([CallUsage(**r['usage']) for r in client.records])
            results[i] = result
            with (output / 'labeling_progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')

    started = time.perf_counter()
    try:
        await asyncio.gather(*(worker(client) for client in clients))
        wall = time.perf_counter() - started
    finally:
        await asyncio.gather(*(client.close() for client in clients))
    calls = [CallUsage(**call) for row in results for call in row['usage']['calls']]
    summary = dict(questions=len(results), failed_questions=sum(r['status'] != 'success' for r in results),
                   wall_seconds=wall, original_label_matches=sum(r['strategy'] == r['original_silver_strategy'] for r in results),
                   **sum_usage(calls))
    summary.pop('calls')
    write_jsonl(output / 'labels.jsonl', results)
    write_json(output / 'labeling_cost.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Cached graphs with original silver_strategy labels')
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--encoder', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--dry-run', action='store_true')
    service_arguments(parser)
    args = parse_config(parser, argv)
    if args.workers < 1 or args.max_tokens < 1 or args.timeout <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Invalid worker count, token budget, timeout or seed')
    rows, manifest = read_jsonl(args.input), read_jsonl(args.training_manifest)
    validate_training(rows, manifest)
    amrs = [amr_text(row) for row in rows]
    prompt = Path(args.prompt).read_text().strip()
    if not prompt:
        parser.error('Empty labeling prompt')
    endpoint = urlsplit(args.base_url)
    config = dict(experiment='offline_preparation_cost', questions=len(rows), seed=args.seed,
                  model=args.model, encoder=args.encoder, device=args.device, workers=args.workers,
                  max_tokens=args.max_tokens, temperature=0, api_attempts=3, timeout=args.timeout,
                  deployment=endpoint.scheme + '://' + endpoint.netloc.split('@')[-1] + endpoint.path,
                  sources=source_files([args.input, args.training_manifest, args.prompt]),
                  amr_inputs_sha256=digest(amrs),
                  training_labels='original frozen silver labels', epochs=30,
                  labeling_time='wall-clock workload including service requests and retries',
                  training_time='30 epochs, excluding frozen BGE feature preparation and serialization')
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    output = new_output(args.output)
    write_json(output / 'configuration.json', config)
    write_jsonl(output / 'training_input.jsonl', rows)
    labeling = asyncio.run(measure_labels(rows, amrs, prompt, args, output))
    from ..classifier.train import train
    train(SimpleNamespace(input=str(output / 'training_input.jsonl'), output=str(output / 'retrained_classifier'),
                          encoder=args.encoder, device=args.device, seed=args.seed))
    training = json.loads((output / 'retrained_classifier' / 'timing.json').read_text())
    write_json(output / 'summary.json', {'silver_label_generation': labeling, 'classifier_training': training})


if __name__ == '__main__':
    main()
