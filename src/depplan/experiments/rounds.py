"""Compare one, two and three interaction rounds on fixed question sets."""
import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from .ablations.runner import Resources, Settings, load_questions, run_cell
from .ablations.variants import VARIANTS
from .common import (check_grid, parse_config, question_key, read_jsonl, seed_summary,
                     service_arguments, source_files, write_csv, write_json)


def timing_summary(rows, timing):
    values = [timing.get(question_key(row)) for row in rows]
    complete = all(v is not None for v in values)
    return {'mean_amr_seconds': mean(values) if complete else None,
            'mean_total_seconds': mean(row['online_seconds'] + value for row, value in zip(rows, values))
            if complete else None}


async def run(args):
    check_grid(args.rounds, 'rounds', (1, 2, 3))
    check_grid(args.seeds, 'seeds')
    questions = load_questions(args.input)
    settings = Settings.from_args(SimpleNamespace(config=args.qa_config,
        semantic_accuracy=getattr(args, 'semantic_accuracy', False),
        judge_model=getattr(args, 'judge_model', None)))
    resources = Resources(args)
    timing_rows = read_jsonl(args.amr_timings) if args.amr_timings else questions
    timing = {}
    for row in timing_rows:
        if row.get('amr_seconds') is not None:
            value = row['amr_seconds']
            if not isinstance(value, (int, float)) or not 0 <= value < float('inf'):
                raise ValueError('AMR times must be finite nonnegative seconds')
            if question_key(row) in timing:
                raise ValueError('Duplicate AMR timing ID')
            timing[question_key(row)] = value
    # Feed the same cached preprocessing measurements into per-question totals.
    for row in questions:
        if question_key(row) in timing:
            row['amr_seconds'] = timing[question_key(row)]
    groups = {d: [r for r in questions if r['dataset'] == d]
              for d in sorted({r['dataset'] for r in questions})}
    manifest = {'experiment': 'interaction_rounds', 'rounds': args.rounds, 'seeds': args.seeds,
                'questions': {d: len(rows) for d, rows in groups.items()},
                'sources': source_files(args.input + ([args.amr_timings] if args.amr_timings else [])),
                'settings': asdict(settings),
                'time_scope': 'online: selection through answer generation; total: online plus measured AMR'}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    existing = root / 'experiment.json'
    if existing.exists() and (not args.resume or json.loads(existing.read_text()) != manifest):
        raise ValueError('Existing experiment differs; use a new output directory')
    write_json(existing, manifest)
    reports = []
    for dataset, rows in groups.items():
        for rounds in args.rounds:
            for seed in args.seeds:
                out = root / dataset / f'rounds_{rounds}' / f'seed_{seed}'
                summary = await run_cell(rows, VARIANTS['full'], seed,
                                         replace(settings, interaction_rounds=rounds), resources, out,
                                         resume=args.resume, experiment='interaction_rounds')
                summary.update(timing_summary(read_jsonl(out / 'predictions.jsonl'), timing))
                reports.append(dict(dataset=dataset, rounds=rounds, seed=seed, **summary))
                write_json(out / 'summary.json', summary)
    metrics = ['em_percent', 'f1_percent', 'acc_percent', 'mean_online_seconds', 'mean_total_seconds',
               'mean_amr_seconds', 'mean_steps', 'mean_llm_calls', 'mean_total_tokens',
               'mean_known_total_tokens', 'failed_questions']
    write_csv(root / 'per_seed.csv', reports)
    write_csv(root / 'mean_sample_sd.csv', seed_summary(reports, ['dataset', 'rounds'], metrics))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--qa-config')
    parser.add_argument('--amr-timings', help='Optional per-question dataset, example_id, amr_seconds JSONL')
    parser.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument('--encoder')
    parser.add_argument('--tokenizer')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--semantic-accuracy', action='store_true',
                        help='Also score Acc; the paper rounds table uses EM/F1 only')
    parser.add_argument('--judge-model')
    parser.add_argument('--judge-base-url')
    parser.add_argument('--judge-api-key')
    service_arguments(parser)
    asyncio.run(run(parse_config(parser, argv)))


if __name__ == '__main__':
    main()
