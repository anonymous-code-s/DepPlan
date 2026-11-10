"""Score saved DepPlan answers with a separately configured evaluator."""
from collections import Counter, defaultdict
from pathlib import Path
import json
from urllib.parse import urlsplit

from .evaluation import score_prediction
from .llm import CallUsage, sum_usage
from .experiments.ablations.reporting import read_jsonl, summarize_records, write_json
from .experiments.ablations.runner import Settings, code_digest, file_digest, judge_endpoint, make_judge


def save_rows(path, rows):
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows),
                         encoding='utf-8')
    temporary.replace(path)


async def score_saved(args):
    if not args.judge_model or not judge_endpoint(args):
        raise ValueError('score-qa requires --judge-model and --judge-base-url (or JUDGE_* environment variables)')
    source, output = Path(args.input), Path(args.output)
    if source.resolve().parent == output.resolve():
        raise ValueError('Use a separate output directory to preserve the original QA records')
    rows = read_jsonl(source)
    if not rows or len({r['seed'] for r in rows}) != 1:
        raise ValueError('score-qa expects the complete predictions from one decoding run')
    keys = [(r['dataset'], r['example_id']) for r in rows]
    if len(set(keys)) != len(rows):
        raise ValueError('Duplicate QA question IDs')
    settings = Settings(judge_model=args.judge_model)
    endpoint = urlsplit(judge_endpoint(args))
    config = dict(experiment='semantic_answer_evaluation', input_sha256=file_digest(source),
                  code_sha256=code_digest(), judge_model=settings.judge_model,
                  judge_base_url=endpoint.scheme + '://' + endpoint.netloc.split('@')[-1] + endpoint.path,
                  seed=rows[0]['seed'], temperature=0,
                  max_tokens=64, question_keys=keys, questions=len(rows))
    # JSON round-trip makes tuple/list equality independent of serialization.
    config = json.loads(json.dumps(config))
    saved = []
    if output.exists() and any(output.iterdir()):
        if not args.resume or not (output / 'configuration.json').exists():
            raise ValueError('Use --resume or an empty evaluation output directory')
        if json.loads((output / 'configuration.json').read_text()) != config:
            raise ValueError('Evaluation input or configuration changed')
        saved = read_jsonl(output / 'predictions.jsonl') if (output / 'predictions.jsonl').exists() else []
        if [(r['dataset'], r['example_id']) for r in saved] != keys[:len(saved)]:
            raise ValueError('Saved evaluations must be an ordered prefix of the input')
        logged = Counter(r['example_id'] for r in read_jsonl(output / 'judge_calls.jsonl')) \
            if (output / 'judge_calls.jsonl').exists() else Counter()
        expected = Counter()
        for record in saved:
            expected[record['example_id']] += record['judge_usage']['call_count']
        if logged != expected:
            raise ValueError('Interrupted evaluator calls lack matching saved costs; preserve this output and use a new directory')
    else:
        if args.retry_failed:
            raise ValueError('--retry-failed requires an existing evaluation with --resume')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'configuration.json', config)
    pending = [i for i in range(len(rows)) if i >= len(saved)
               or args.retry_failed and saved[i].get('judge_status') == 'error']
    judge = make_judge(args, settings, rows[0]['seed'], output) if pending else None
    try:
        for i in pending:
            record = dict(rows[i])
            evaluation = await score_prediction(record, judge)
            previous = saved[i] if i < len(saved) else None
            if previous:
                calls = previous.get('judge_usage', {}).get('calls', []) + evaluation['judge_usage']['calls']
                evaluation['judge_usage'] = sum_usage([CallUsage(**call) for call in calls])
                evaluation['judge_seconds'] += previous.get('judge_seconds', 0.0)
            record.update(evaluation, judge_evaluation_attempts=(previous or {}).get('judge_evaluation_attempts', 0) + 1)
            record['scoring_seconds'] = record['judge_seconds']
            if i < len(saved):
                saved[i] = record
            else:
                saved.append(record)
            save_rows(output / 'predictions.jsonl', saved)
    finally:
        if judge is not None:
            await judge.close()
    groups = defaultdict(list)
    for record in saved:
        groups[record['dataset']].append(record)
    summary = {dataset: summarize_records(records) for dataset, records in groups.items()}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
