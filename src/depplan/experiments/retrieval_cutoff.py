"""Sweep retrieval cutoffs while reusing each seed's saved plans."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from ..executor import PlanExecutor
from ..llm import CallUsage, LLMClient, sum_usage
from ..metrics import answer_scores
from ..plan import parse_plan
from .ablations.runner import (Settings, _synchronize, code_digest, input_view,
                               load_questions, resume_records)
from .ablations.reporting import summarize_records
from .common import (check_grid, digest, load_runs, parse_config, question_key,
                     seed_summary, service_arguments, source_files,
                     validate_seed_cohorts, write_csv, write_json)


def align_plans(questions, plans):
    expected = {question_key(r) for r in questions}
    validate_seed_cohorts(plans)
    if {r['method'] for r in plans} != {'depplan'}:
        raise ValueError('Retrieval sensitivity requires DepPlan plans only')
    index = {}
    for row in plans:
        key = row['seed'], question_key(row)
        if key in index:
            raise ValueError('Duplicate saved plan')
        index[key] = row
    seeds = sorted({r['seed'] for r in plans})
    for seed in seeds:
        if {key[1] for key in index if key[0] == seed} != expected:
            raise ValueError('Every seed must have a record for every input question, including failures')
        for q in questions:
            saved = index[seed, question_key(q)]
            if saved.get('question', q['question']) != q['question']:
                raise ValueError('Saved plan and question text differ')
    return seeds, index


async def execute_question(row, saved, embedding, client, retriever, settings, cutoff, seed, device):
    result = dict(dataset=row['dataset'], example_id=row['example_id'], seed=seed, top_k=cutoff,
                  question=row['question'], gold_answers=row['answers'], plan=saved.get('plan'),
                  prediction='', status='error', em=0, f1=0.0, steps=[], online_seconds=0.0)
    result['plan_sha256'] = digest(result['plan'])
    client.example_id = row['example_id']
    client.records.clear()
    executor = PlanExecutor(client, retriever, top_k=cutoff,
                            max_tokens=settings.answer_max_tokens, compact_prompts=settings.compact_prompts)
    if result['plan'] is None:
        result['error'] = 'No saved executable plan'
    else:
        started = None
        try:
            plan = parse_plan(result['plan'], require_placeholder_edges=True)
            _synchronize(device)
            started = time.perf_counter()
            answer = await executor.execute(row['question'], embedding, plan)
            result.update(prediction=answer.prediction, steps=answer.steps, status='success')
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
            result['steps'] = executor.last_trace
        finally:
            if started is not None:
                _synchronize(device)
                result['online_seconds'] = time.perf_counter() - started
        if result['status'] == 'success':
            scores = answer_scores(row['answers'], result['prediction'], dataset=row['dataset'])
            result.update(em=int(scores['em']), f1=scores['f1'])
    result['qa_seconds'] = result['online_seconds']
    result['usage'] = sum_usage([CallUsage(**r['usage']) for r in client.records])
    return result


async def run(args):
    check_grid(args.cutoffs, 'cutoffs')
    if 0 in args.cutoffs:
        raise ValueError('Retrieval cutoffs must be positive')
    questions = load_questions(args.input)
    plans = load_runs(args.plans)
    seeds, index = align_plans(questions, plans)
    settings = Settings.from_args(SimpleNamespace(config=args.qa_config))
    endpoint = urlsplit(args.base_url)
    manifest = {'experiment': 'retrieval_cutoff', 'cutoffs': args.cutoffs, 'seeds': seeds,
                'questions': len(questions), 'sources': source_files(args.input + args.plans),
                'settings': asdict(settings), 'encoder': args.encoder, 'device': args.device,
                'model': args.model, 'deployment': endpoint.scheme + '://' + endpoint.netloc.split('@')[-1] + endpoint.path,
                'code_sha256': code_digest(),
                'time_scope': 'QA retrieval, intermediate answers and final answer; excludes passage encoding, AMR, planning and scoring'}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    existing = root / 'experiment.json'
    if existing.exists() and (not args.resume or json.loads(existing.read_text()) != manifest):
        raise ValueError('Existing experiment differs; use a new output directory')
    write_json(existing, manifest)
    from ..retrieval import DenseRetriever
    retriever = DenseRetriever(args.encoder, device=args.device)
    reports = []
    for dataset in sorted({r['dataset'] for r in questions}):
        rows = [r for r in questions if r['dataset'] == dataset]
        embeddings = {}
        for seed in seeds:
            for cutoff in args.cutoffs:
                out = root / dataset / f'seed_{seed}' / f'k_{cutoff}'
                config = dict(manifest, dataset=dataset, seed=seed, top_k=cutoff,
                              question_keys=[[r['dataset'], r['example_id']] for r in rows],
                              input_sha256=digest([input_view(r) for r in rows]),
                              plans_sha256=digest([index[seed, question_key(r)].get('plan') for r in rows]))
                saved = resume_records(out, config, args.resume)
                client = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key,
                                   seed=seed, max_attempts=settings.api_attempts,
                                   timeout=settings.request_timeout_seconds, log_path=out / 'calls.jsonl')
                try:
                    with (out / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
                        for row in rows[len(saved):]:
                            key = question_key(row)
                            source = index[seed, key]
                            if source.get('plan') is not None and key not in embeddings:
                                embeddings[key] = {'flat_contexts': row['passages'], 'context_embeddings':
                                    retriever.encoder.encode(row['passages'], normalize_embeddings=True,
                                                             show_progress_bar=False).tolist()}
                            result = await execute_question(row, source, embeddings.get(key), client,
                                                            retriever, settings, cutoff, seed, args.device)
                            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            saved.append(result)
                finally:
                    await client.close()
                summary = summarize_records(saved)
                summary['mean_qa_seconds'] = summary.pop('mean_online_seconds')
                write_json(out / 'summary.json', summary)
                reports.append(dict(dataset=dataset, top_k=cutoff, seed=seed, **summary))
    write_csv(root / 'per_seed.csv', reports)
    write_csv(root / 'mean_sample_sd.csv', seed_summary(reports, ['dataset', 'top_k'],
              ['em_percent', 'f1_percent', 'mean_qa_seconds', 'mean_total_tokens',
               'mean_known_total_tokens', 'mean_steps', 'mean_llm_calls', 'failed_questions']))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--plans', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--qa-config')
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[1, 3, 5, 10, 15, 20])
    parser.add_argument('--encoder', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    service_arguments(parser)
    asyncio.run(run(parse_config(parser, argv)))


if __name__ == '__main__':
    main()
