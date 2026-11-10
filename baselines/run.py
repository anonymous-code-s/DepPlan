"""Run baseline workflows on official or prepared development questions."""
import argparse
import asyncio
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from depplan.metrics import answer_scores
from depplan import metrics as shared_metrics, prompts as shared_prompts

from .configuration import ORDER, ROOT, load_config, methods
from .core import MethodContext, ModelClient, Tokenizer
from .data import DATASETS, file_hash, load_questions
from .retrieval import Encoder


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()


def source_fingerprint():
    paths = sorted(ROOT.rglob('*.py'))
    paths += sorted((ROOT / 'configs').rglob('*.json'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    # The runner imports the shared answer scorer and semantic judge prompt.
    # Changes to either must invalidate resume, including installed-package use.
    for module in (shared_metrics, shared_prompts):
        digest.update(module.__name__.encode())
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def call_totals(records):
    out = {'llm_calls': len(records)}
    for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
        values = [r.get(key) for r in records]
        out[key] = sum(values) if all(v is not None for v in values) else None
        out['known_' + key] = sum(v for v in values if v is not None)
    out['calls_without_usage'] = sum(r.get('total_tokens') is None for r in records)
    return out


def summarize(rows):
    count = len(rows)
    if not count:
        raise ValueError('Cannot summarize an empty run')
    out = dict(questions=count, failures=sum(r['status'] != 'success' for r in rows),
               em_correct=sum(r['em'] for r in rows),
               em_percent=100 * statistics.mean(r['em'] for r in rows),
               f1_percent=100 * statistics.mean(r['f1'] for r in rows))
    for key in ['time_seconds', 'steps', 'llm_calls', 'prompt_tokens', 'completion_tokens',
                'total_tokens', 'known_total_tokens']:
        values = [r[key] for r in rows]
        out['mean_' + key] = statistics.mean(values) if all(v is not None for v in values) else None
    out['calls_without_usage'] = sum(r['calls_without_usage'] for r in rows)
    judgments = [r.get('semantic_correct') for r in rows]
    out['semantic_accuracy_percent'] = (100 * statistics.mean(judgments)
                                        if all(v is not None for v in judgments) else None)
    out['acc_correct'] = sum(judgments) if all(v is not None for v in judgments) else None
    out['acc_percent'] = out['semantic_accuracy_percent']
    out['semantic_unscored_questions'] = sum(v is None for v in judgments)
    out['judge_failures'] = sum(r.get('judge_status') == 'error' for r in rows)
    for key in ['scoring_time_seconds', 'judge_time_seconds', 'judge_llm_calls',
                'judge_prompt_tokens', 'judge_completion_tokens', 'judge_total_tokens',
                'judge_known_total_tokens']:
        values = [r.get(key) for r in rows]
        out['mean_' + key] = statistics.mean(values) if all(v is not None for v in values) else None
    out['judge_calls_without_usage'] = sum(r.get('judge_calls_without_usage', 0) for r in rows)
    return out


def summary_row(row):
    fields = ['example_id', 'status', 'em', 'f1', 'semantic_correct', 'judge_status',
              'time_seconds', 'steps', 'llm_calls', 'prompt_tokens', 'completion_tokens',
              'total_tokens', 'known_total_tokens', 'calls_without_usage']
    fields += ['scoring_time_seconds', 'judge_time_seconds', 'judge_llm_calls',
               'judge_prompt_tokens', 'judge_completion_tokens', 'judge_total_tokens',
               'judge_known_total_tokens', 'judge_calls_without_usage']
    return {key: row.get(key) for key in fields}


async def evaluate_one(row, method, config, seed, client, encoder, tokenizer, judge=None):
    client.example_id = row['example_id']
    before = len(client.records)
    started = time.perf_counter()
    ctx = MethodContext(row['question'], row['passages'], row['dataset'], config,
                        seed, client, encoder, tokenizer)
    prediction, error, status = '', None, 'success'
    try:
        prediction = await asyncio.wait_for(method(ctx), timeout=config['question_timeout_seconds'])
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError('Method produced no final answer')
    except Exception as exc:
        status, error = 'error', type(exc).__name__
        ctx.event('failure', error_type=error)
    elapsed = time.perf_counter() - started
    scoring_started = time.perf_counter()
    scores = (answer_scores(row['answers'], prediction, dataset=row['dataset'])
              if status == 'success' else {'em': 0, 'f1': 0.0})
    result = dict(example_id=row['example_id'], dataset=row['dataset'], seed=seed,
                  method=config['method'], model=client.model, question=row['question'],
                  references=row['answers'], prediction=prediction, status=status,
                  error_type=error, em=int(scores['em']), f1=scores['f1'],
                  time_seconds=elapsed, steps=ctx.steps, stop_reason=ctx.stop,
                  trace=ctx.trace, **call_totals(client.records[before:]))
    result.update(semantic_correct=None, judge_status='disabled', judge_time_seconds=0.0,
                  **{'judge_' + key: value for key, value in call_totals([]).items()})
    if judge:
        judge_before = len(judge.records)
        judge_started = time.perf_counter()
        result['semantic_correct'], result['judge_status'] = 0, 'not_needed'
        if status == 'success':
            judge.example_id = row['example_id']
            try:
                value = await judge.generate(
                    role='semantic_judge', system=shared_prompts.SEMANTIC_JUDGE_SYSTEM,
                    user=json.dumps(dict(question=row['question'], references=row['answers'],
                                         prediction=prediction), ensure_ascii=False),
                    temperature=0, max_tokens=64)
                obj = ctx.json(value)
                flag = obj.get('correct')
                if not (type(flag) is bool or type(flag) is int and flag in {0, 1}):
                    raise ValueError('Judge must return a binary correct field')
                result.update(semantic_correct=int(flag), judge_status='success')
            except Exception as exc:
                result.update(semantic_correct=None, judge_status='error',
                              judge_error_type=type(exc).__name__)
        result.update(judge_time_seconds=time.perf_counter() - judge_started,
                      **{'judge_' + key: value
                         for key, value in call_totals(judge.records[judge_before:]).items()})
    result['scoring_time_seconds'] = time.perf_counter() - scoring_started
    return result


async def run_cell(rows, config, seed, method, args, endpoint, encoder):
    if args.judge_model and not args.judge_base_url:
        raise ValueError('Semantic accuracy requires an independent --judge-base-url or '
                         'JUDGE_BASE_URL when --judge-model or JUDGE_MODEL is configured')
    dataset = rows[0]['dataset']
    output = args.output / config['method'] / dataset / f'seed_{seed}'
    identity = dict(method=config['method'], dataset=dataset, seed=seed, config=config,
                    input_sha256=file_hash(args.input), code_sha256=source_fingerprint(),
                    question_ids=[r['example_id'] for r in rows],
                    endpoint=endpoint['base_url'], judge_model=args.judge_model,
                    judge_endpoint=args.judge_base_url if args.judge_model else None,
                    judge_temperature=0 if args.judge_model else None,
                    judge_max_tokens=64 if args.judge_model else None,
                    declared_serving_metadata={key: endpoint[key] for key in (
                        'model_revision', 'model_sha256', 'vllm_version', 'dtype',
                        'tensor_parallel_size', 'hardware') if key in endpoint})
    manifest_path, predictions = output / 'manifest.json', output / 'predictions.jsonl'
    old = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not args.resume:
            raise FileExistsError(f'Run exists: {output}. Use --resume for identical settings.')
        if manifest['identity'] != identity:
            raise ValueError('Resume requires identical input, configuration, endpoint, and source code')
        if predictions.exists():
            with predictions.open(encoding='utf-8') as stream:
                old = [summary_row(json.loads(line)) for line in stream if line.strip()]
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output directory is nonempty without a run manifest: {output}')
        dump(manifest_path, dict(identity=identity, started_utc=datetime.now(timezone.utc).isoformat()))
    done = {r['example_id'] for r in old}
    if len(done) != len(old) or not done.issubset(identity['question_ids']):
        raise ValueError('Resume predictions contain duplicate or unknown question IDs')
    tokenizer = Tokenizer(endpoint.get('tokenizer', config['tokenizer']),
                          revision=config.get('tokenizer_revision'))
    tokenizer._get()
    if config['retriever'] in {'dense', 'hybrid'}:
        encoder.load()
    client = ModelClient(model=config['model'], base_url=endpoint['base_url'],
                         api_key=endpoint['api_key'], seed=seed, config=config,
                         tokenizer=tokenizer, log_path=output / 'calls.jsonl')
    judge = (ModelClient(model=args.judge_model, base_url=args.judge_base_url,
                          api_key=os.environ.get('JUDGE_API_KEY', 'EMPTY'), seed=seed,
                          config=dict(api_attempts=config['api_attempts'],
                                      request_timeout_seconds=config['request_timeout_seconds']),
                          tokenizer=None, log_path=output / 'judge_calls.jsonl')
             if args.judge_model else None)
    results = list(old)
    try:
        for row in rows:
            if row['example_id'] in done:
                continue
            result = await evaluate_one(row, method, config, seed, client, encoder, tokenizer, judge)
            append(predictions, result)
            results.append(summary_row(result))
            client.records.clear()
            if judge:
                judge.records.clear()
            print(f"{config['method']} {dataset} seed={seed} {len(results)}/{len(rows)} {result['status']}", flush=True)
    finally:
        await client.close()
        if judge:
            await judge.close()
    summary = dict(method=config['method'], dataset=dataset, seed=seed, **summarize(results))
    dump(output / 'summary.json', summary)
    return summary


def aggregate(summaries, output):
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row['method'], row['dataset'])].append(row)
    rows = []
    for (method, dataset), seeds in grouped.items():
        seeds.sort(key=lambda row: row['seed'])
        for metric in ['em_percent', 'f1_percent', 'semantic_accuracy_percent',
                       'mean_time_seconds', 'mean_total_tokens', 'mean_llm_calls', 'mean_steps',
                       'mean_scoring_time_seconds', 'mean_judge_time_seconds',
                       'mean_judge_total_tokens', 'mean_judge_llm_calls']:
            values = [s[metric] for s in seeds]
            usable = all(v is not None for v in values)
            rows.append(dict(method=method, dataset=dataset, metric=metric, runs=len(seeds),
                             seeds=';'.join(str(s['seed']) for s in seeds),
                             mean=statistics.mean(values) if usable else '',
                             sample_sd=statistics.stdev(values) if usable and len(values) > 1 else ''))
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'mean_sample_sd.csv'
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            retained = [r for r in csv.DictReader(stream)
                        if (r['method'], r['dataset']) not in grouped]
        rows = retained + rows
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=ORDER, default=list(ORDER))
    p.add_argument('--method', choices=ORDER, help='Select one method')
    p.add_argument('--input', type=Path)
    p.add_argument('--dataset', choices=sorted(DATASETS))
    p.add_argument('--output', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2026])
    p.add_argument('--seed', type=int)
    p.add_argument('--config', type=Path, help='JSON overrides for one selected method')
    p.add_argument('--endpoints', type=Path, help='JSON mapping method names to service settings')
    p.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://localhost:8000/v1'))
    p.add_argument('--model', help='Served model name for one selected method')
    p.add_argument('--tokenizer', help='Tokenizer identifier or local directory')
    p.add_argument('--encoder', help='BGE encoder identifier or local directory')
    p.add_argument('--device')
    p.add_argument('--limit', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--list', action='store_true', help='Print effective configs without loading models')
    p.add_argument('--validate-only', action='store_true', help='Check configs and input without API calls')
    semantic = p.add_mutually_exclusive_group()
    semantic.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'),
                          help='Independent semantic evaluator (or JUDGE_MODEL; unset disables scoring)')
    semantic.add_argument('--no-semantic-accuracy', dest='judge_model',
                          action='store_const', const=None,
                          help='Explicitly disable semantic-accuracy evaluation')
    p.add_argument('--judge-base-url', default=os.environ.get('JUDGE_BASE_URL'),
                   help='Independent evaluator endpoint (or JUDGE_BASE_URL; required with a judge model)')
    return p


async def run(args):
    selected = [args.method] if args.method else args.methods
    if len(set(selected)) != len(selected):
        raise ValueError('Select each method once')
    if len(selected) != 1 and (args.model or args.config):
        raise ValueError('--model and --config apply to one selected method; use --endpoints for several')
    overrides = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    overrides.update({k: v for k, v in [('model', args.model), ('tokenizer', args.tokenizer),
                                       ('encoder', args.encoder)] if v is not None})
    endpoints = json.loads(args.endpoints.read_text(encoding='utf-8')) if args.endpoints else {}
    configs = {}
    for name in selected:
        per_method = {k: v for k, v in endpoints.get(name, {}).items() if k in {'model', 'tokenizer'}}
        configs[name] = load_config(name, {**overrides, **per_method})
    if args.list:
        print(json.dumps(configs, ensure_ascii=False, indent=2))
        return
    if not args.input:
        raise ValueError('--input is required')
    rows = load_questions(args.input, args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        rows = rows[:args.limit]
    registry = methods()
    if args.validate_only:
        print(json.dumps(dict(methods=selected, questions=len(rows),
                              datasets=sorted({r['dataset'] for r in rows}),
                              status='configuration_and_input_valid'), indent=2))
        return
    if not args.output:
        raise ValueError('--output is required for execution')
    seeds = [args.seed] if args.seed is not None else args.seeds
    if len(set(seeds)) != len(seeds):
        raise ValueError('Select each decoding seed once')
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row['dataset']].append(row)
    summaries = []
    encoders = {}
    for name, config in configs.items():
        endpoint = dict(base_url=args.base_url, api_key_env='OPENAI_API_KEY')
        endpoint.update(endpoints.get(name, {}))
        endpoint['api_key'] = os.environ.get(endpoint['api_key_env'], 'EMPTY')
        encoder = encoders.setdefault(config['encoder'], Encoder(config['encoder'], args.device))
        for dataset_rows in by_dataset.values():
            for seed in seeds:
                summaries.append(await run_cell(dataset_rows, config, seed, registry[name],
                                                 args, endpoint, encoder))
    aggregate(summaries, args.output)


def main(argv=None):
    asyncio.run(run(parser().parse_args(argv)))


if __name__ == '__main__':
    main()
