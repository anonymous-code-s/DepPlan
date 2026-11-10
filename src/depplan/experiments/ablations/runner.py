"""Run component controls on shared question inputs and decoding seeds."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from ...agents import orchestration_info
from ...executor import PlanExecutor
from ...evaluation import score_prediction
from ...llm import CallUsage, LLMClient, sum_usage
from ...metrics import answer_scores
from ...plan import extract_json_object
from ...planner import MultiAgentPlanner
from ...prompt_compaction import indexed_semantic_units
from ...semantic_units.linearization import build_semantic_units
from .reporting import aggregate_runs, digest, read_jsonl, summarize_records, write_json
from .surrogate import build_text_surrogate
from .variants import DIRECT_STRATEGY_SYSTEM, LABELS, VARIANTS, training_majority


@dataclass(frozen=True)
class Settings:
    planning_temperature: float = 0.7
    answer_temperature: float = 0.0
    interaction_rounds: int = 2
    retrieval_top_k: int = 5
    planning_max_tokens: int = 1024
    answer_max_tokens: int = 512
    compact_prompts: bool = True
    api_attempts: int = 3
    plan_repair_attempts: int = 0
    request_timeout_seconds: float = 180.0
    selector_temperature: float = 0.0
    selector_max_tokens: int = 128
    semantic_accuracy: bool = True
    judge_model: str | None = None

    def __post_init__(self):
        if type(self.semantic_accuracy) is not bool or (self.judge_model is not None
                and (not isinstance(self.judge_model, str) or not self.judge_model.strip())):
            raise ValueError('Semantic evaluation needs a boolean switch and an optional nonempty judge model')
        if type(self.interaction_rounds) is not int or self.interaction_rounds not in (1, 2, 3):
            raise ValueError("Interaction rounds must be 1, 2, or 3")
        if type(self.api_attempts) is not int or not 1 <= self.api_attempts <= 3:
            raise ValueError('API attempts must be an integer from 1 to 3')
        if type(self.plan_repair_attempts) is not int or self.plan_repair_attempts != 0:
            raise ValueError('Additional plan-repair calls are disabled for every condition')

    @classmethod
    def from_args(cls, args):
        values = json.loads(Path(args.config).read_text()) if getattr(args, 'config', None) else {}
        allowed = set(cls.__dataclass_fields__)
        auxiliary = {'language_model', 'encoder', 'decoding_seeds', 'amr_input'}
        if set(values) - allowed - auxiliary:
            raise ValueError('Unknown QA configuration options: ' + ', '.join(sorted(set(values)-allowed-auxiliary)))
        result = {k: v for k, v in values.items() if k in allowed}
        for field in ['semantic_accuracy', 'judge_model']:
            if getattr(args, field, None) is not None:
                result[field] = getattr(args, field)
        if getattr(args, 'timeout', None) is not None:
            result['request_timeout_seconds'] = args.timeout
        settings = cls(**result)
        if settings.interaction_rounds != 2:
            raise ValueError('Component ablations use two proposal-verification rounds')
        if settings.answer_temperature != 0:
            raise ValueError('Answer decoding must remain greedy across component ablations')
        if (settings.retrieval_top_k < 1 or settings.planning_max_tokens < 1
                or settings.answer_max_tokens < 1 or settings.request_timeout_seconds <= 0):
            raise ValueError('Retrieval, token budgets and timeout must be positive')
        return settings


def load_questions(paths):
    rows = [row for path in paths for row in read_jsonl(path)]
    if not rows:
        raise ValueError('The input contains no questions')
    keys = set()
    for row in rows:
        key = row.get('dataset'), row.get('example_id')
        if key[0] not in {'hotpot', 'musique', '2wiki'} or not isinstance(key[1], str) or not key[1]:
            raise ValueError('Each question needs a dataset and nonempty string example_id')
        if key in keys:
            raise ValueError(f'Duplicate question: {key}')
        keys.add(key)
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'Missing question text: {key}')
        for field in ['passages', 'answers']:
            if not isinstance(row.get(field), list) or not row[field] or not all(isinstance(t, str) for t in row[field]):
                raise ValueError(f'{field} must be a nonempty string list: {key}')
        if row.get('status', 'success') == 'success' and not isinstance(row.get('amr_triples'), list):
            raise ValueError(f'Missing AMR triples: {key}')
        if row.get('amr_seconds') is not None:
            value = row['amr_seconds']
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'AMR time must be finite nonnegative seconds: {key}')
    return rows


def code_digest():
    root = Path(__file__).resolve().parents[2]
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.json'}})


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_view(row):
    return {key: row.get(key) for key in ['dataset', 'example_id', 'question', 'passages',
                                         'answers', 'amr_triples', 'structure_source', 'status',
                                         'amr_seconds', 'amr_batch_size', 'amr_configured_batch_size']}


class Resources:
    def __init__(self, args):
        self.args = args
        self.checkpoint = Path(args.checkpoint)
        self.classifier_config = json.loads((self.checkpoint / 'config.json').read_text())
        self.encoder_name = args.encoder or self.classifier_config['encoder']
        self.classifier = None
        self.retriever = None
        self.tokenizer = None

    def prepare(self, variant):
        if self.retriever is None:
            from ...retrieval import DenseRetriever
            self.retriever = DenseRetriever(self.encoder_name, device=self.args.device)
        if variant.selector == 'gcn' and self.classifier is None:
            from ...classifier.runtime import Classifier
            self.classifier = Classifier(self.checkpoint, self.encoder_name, self.args.device)
            self.classifier._encoder = self.retriever.encoder
        if variant.units == 'text' and self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer or self.args.model)


def _synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


@contextmanager
def _measure(result, name, device):
    _synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        _synchronize(device)
        result[name] = time.perf_counter() - start


async def evaluate_question(row, variant, settings, resources, client, seed, majority=None, *, judge=None):
    result = dict(dataset=row['dataset'], example_id=row['example_id'], variant=variant.name,
                  seed=seed, input_sha256=digest(input_view(row)), question=row['question'],
                  gold_answers=row['answers'], status='error', prediction='', em=0, f1=0.0,
                  strategy=None, steps=[])
    client.example_id = row['example_id']
    client.records.clear()
    planner = MultiAgentPlanner(client, temperature=settings.planning_temperature,
                               max_tokens=settings.planning_max_tokens,
                               compact_prompts=settings.compact_prompts,
                               use_verifier=variant.verifier, use_judge=variant.judge,
                               rounds=settings.interaction_rounds)
    executor = PlanExecutor(client, resources.retriever, top_k=settings.retrieval_top_k,
                            max_tokens=settings.answer_max_tokens,
                            compact_prompts=settings.compact_prompts)
    _synchronize(resources.args.device)
    started = time.perf_counter()
    try:
        if row.get('status', 'success') != 'success':
            raise ValueError('The common AMR preprocessing failed for this question')
        with _measure(result, 'strategy_selection_seconds', resources.args.device):
            if variant.selector == 'gcn':
                graph = {k: row[k] for k in ['dataset', 'example_id', 'amr_triples', 'structure_source'] if k in row}
                strategy = resources.classifier.predict([graph])[0]['predicted_strategy']
                if strategy not in LABELS:
                    raise ValueError('Classifier returned no valid strategy')
            elif variant.selector == 'llm':
                response = await client.complete(role='strategy_selection', system=DIRECT_STRATEGY_SYSTEM,
                    user='Question:\n' + row['question'], temperature=settings.selector_temperature,
                    max_tokens=settings.selector_max_tokens)
                strategy = extract_json_object(response.text).get('strategy')
                if strategy not in LABELS:
                    raise ValueError('The direct selector returned an invalid strategy')
            elif variant.selector == 'majority':
                if majority not in LABELS:
                    raise ValueError('A training-derived majority strategy is required')
                strategy = majority
            else:
                strategy = None
        result['strategy'] = strategy
        with _measure(result, 'semantic_units_seconds', resources.args.device):
            units = None
            if variant.units != 'none':
                linearized = build_semantic_units(row['amr_triples'])
                units = indexed_semantic_units(linearized) if settings.compact_prompts else linearized['semantic_units']
                if variant.units == 'text':
                    units, surrogate_report = build_text_surrogate(row['question'], units, resources.tokenizer,
                        strategy=strategy, indexed_units=settings.compact_prompts)
                    result['surrogate'] = surrogate_report
            result['semantic_units'] = units
        with _measure(result, 'passage_encoding_seconds', resources.args.device):
            matrix = resources.retriever.encoder.encode(row['passages'], convert_to_numpy=True,
                                                        normalize_embeddings=True)
            embedding = {'flat_contexts': row['passages'], 'context_embeddings': matrix.tolist()}
        with _measure(result, 'plan_construction_seconds', resources.args.device):
            planning = await planner.plan(row['question'], units=units, strategy=strategy)
            result['plan'] = planning.plan.to_dict()
        with _measure(result, 'plan_execution_seconds', resources.args.device):
            execution = await executor.execute(row['question'], embedding, planning.plan)
            result.update(prediction=execution.prediction, steps=execution.steps, status='success')
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['steps'] = executor.last_trace
    _synchronize(resources.args.device)
    result['online_seconds'] = time.perf_counter() - started
    result['usage'] = sum_usage([CallUsage(**r['usage']) for r in client.records])
    result['amr_seconds'] = row.get('amr_seconds')
    result['total_seconds'] = (result['online_seconds'] + row['amr_seconds']
                               if row.get('amr_seconds') is not None else None)
    result['time_seconds'] = result['total_seconds']
    result['amr_configured_batch_size'] = row.get('amr_configured_batch_size')
    result['agent_planning_seconds'] = result.get('plan_construction_seconds', 0.0)
    result['retrieval_answer_seconds'] = result.get('plan_execution_seconds', 0.0)
    result['plan_construction_seconds'] = (result['agent_planning_seconds']
                                           + result.get('semantic_units_seconds', 0.0))
    result['plan_execution_seconds'] = (result['retrieval_answer_seconds']
                                        + result.get('passage_encoding_seconds', 0.0))
    result['online_overhead_seconds'] = max(0.0, result['online_seconds'] - sum(
        result.get(field, 0.0) for field in ['strategy_selection_seconds',
                                          'plan_construction_seconds', 'plan_execution_seconds']))
    scoring_started = time.perf_counter()
    if result['status'] == 'success':
        scores = answer_scores(row['answers'], result['prediction'], dataset=row['dataset'])
        result.update(em=int(scores['em']), f1=scores['f1'])
    result.update(await score_prediction(result, judge if settings.semantic_accuracy else None))
    if settings.semantic_accuracy and judge is None:
        result['judge_status'] = 'not_configured'
    result['scoring_seconds'] = time.perf_counter() - scoring_started
    return result


def judge_endpoint(args):
    return getattr(args, 'judge_base_url', None) or os.environ.get('JUDGE_BASE_URL')


def make_judge(args, settings, seed, output):
    if not settings.semantic_accuracy or not settings.judge_model:
        return None
    if not judge_endpoint(args):
        raise ValueError('Set --judge-base-url or JUDGE_BASE_URL for the independent evaluator')
    return LLMClient(model=settings.judge_model, base_url=judge_endpoint(args),
        api_key=getattr(args, 'judge_api_key', None) or os.environ.get('JUDGE_API_KEY')
                or 'EMPTY',
        seed=seed, max_attempts=settings.api_attempts, timeout=settings.request_timeout_seconds,
        log_path=Path(output) / 'judge_calls.jsonl', vllm_parameters=False)


def configuration(rows, variant, seed, settings, resources, majority_info, experiment="component_ablations"):
    endpoint = urlsplit(resources.args.base_url)
    if settings.semantic_accuracy and settings.judge_model and not judge_endpoint(resources.args):
        raise ValueError('Set --judge-base-url or JUDGE_BASE_URL for the independent evaluator')
    evaluator = urlsplit(judge_endpoint(resources.args) or '')
    return {
        'experiment': experiment, 'schema_version': 2,
        'variant': variant.to_dict(), 'seed': seed, 'settings': asdict(settings),
        'model': resources.args.model, 'encoder': resources.encoder_name,
        'orchestration': orchestration_info(),
        'deployment': endpoint.scheme + '://' + endpoint.netloc.split('@')[-1] + endpoint.path,
        'semantic_evaluator': ({'model': settings.judge_model, 'temperature': 0, 'max_tokens': 64,
            'deployment': evaluator.scheme + '://' + evaluator.netloc.split('@')[-1] + evaluator.path,
            'time_and_tokens': 'separate from QA inference'}
            if settings.semantic_accuracy and settings.judge_model else None),
        'device': resources.args.device,
        'checkpoint_sha256': file_digest(resources.checkpoint / 'model.safetensors'),
        'classifier_config_sha256': file_digest(resources.checkpoint / 'config.json'),
        'structure_source': resources.classifier_config['structure_source'],
        'code_sha256': code_digest(),
        'question_keys': [[r['dataset'], r['example_id']] for r in rows],
        'input_sha256': digest([input_view(r) for r in rows]), 'questions': len(rows),
        'majority': majority_info if variant.selector == 'majority' else None,
        'tokenizer': (resources.args.tokenizer or resources.args.model) if variant.units == 'text' else None,
        'time_scope': 'total = measured cached AMR + online selection, units, passage encoding, planning and execution; scoring excluded',
        'component_scope': {'plan_construction': 'semantic units + agent planning',
                            'plan_execution': 'passage encoding + retrieval + answer generation',
                            'overhead': 'remaining measured online coordination time'},
    }


def resume_records(out, config, resume):
    if not out.exists() or not any(out.iterdir()):
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'configuration.json', config)
        return []
    if not resume:
        raise ValueError(f'Output is not empty; use --resume or a new directory: {out}')
    old = json.loads((out / 'configuration.json').read_text())
    if old != config:
        raise ValueError(f'Resume configuration or inputs changed: {out}')
    saved = read_jsonl(out / 'predictions.jsonl') if (out / 'predictions.jsonl').exists() else []
    keys = [(r['dataset'], r['example_id']) for r in saved]
    if keys != [tuple(k) for k in config['question_keys'][:len(saved)]]:
        raise ValueError('Resume predictions must be an ordered prefix of the fixed question list')
    completed = {r['example_id'] for r in saved}
    for filename in ['calls.jsonl', 'judge_calls.jsonl']:
        if not (out / filename).exists():
            continue
        with (out / filename).open(encoding='utf-8') as log:
            if any(json.loads(line)['example_id'] not in completed for line in log if line.strip()):
                raise ValueError('Interrupted requests lack a final question record; retain this run and start a new output directory')
    return saved


async def run_cell(rows, variant, seed, settings, resources, out, *, majority_info=None, resume=False,
                   experiment="component_ablations"):
    sources = {r.get('structure_source') for r in rows if r.get('status', 'success') == 'success'}
    if sources - {None, resources.classifier_config['structure_source']}:
        raise ValueError('Question graph source differs from the configured checkpoint')
    out = Path(out)
    config = configuration(rows, variant, seed, settings, resources, majority_info, experiment)
    saved = resume_records(out, config, resume)
    if len(saved) < len(rows):
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        resources.prepare(variant)
        client = LLMClient(model=resources.args.model, base_url=resources.args.base_url,
                           api_key=resources.args.api_key, seed=seed,
                           max_attempts=settings.api_attempts,
                           timeout=settings.request_timeout_seconds, log_path=out / 'calls.jsonl')
        judge = None
        try:
            judge = make_judge(resources.args, settings, seed, out)
            with (out / 'predictions.jsonl').open('a', encoding='utf-8') as f:
                for row in rows[len(saved):]:
                    result = await evaluate_question(row, variant, settings, resources, client, seed,
                                                     majority_info['strategy'] if majority_info else None,
                                                     judge=judge)
                    f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    saved.append(result)
        finally:
            await client.close()
            if judge is not None:
                await judge.close()
    summary = summarize_records(saved)
    write_json(out / 'summary.json', summary)
    return summary


async def run_suite(args):
    rows = load_questions(args.input)
    if not args.variants or set(args.variants) - set(VARIANTS):
        raise ValueError('Select known ablation variants')
    if not args.seeds or any(type(seed) is not int or seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError('Seeds must be integers in [0, 2**32)')
    variants = [VARIANTS[name] for name in args.variants]
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Variants and seeds must not repeat')
    settings, resources = Settings.from_args(args), Resources(args)
    majority_info = None
    if any(v.selector == 'majority' for v in variants):
        strategy, counts, training_keys = training_majority(args.training_manifest)
        if training_keys & {(r['dataset'], r['example_id']) for r in rows}:
            raise ValueError('QA inputs overlap the majority-selector training manifest')
        majority_info = {'strategy': strategy, 'training_counts': counts,
                         'manifest_sha256': file_digest(args.training_manifest)}
    groups = {name: [r for r in rows if r['dataset'] == name]
              for name in ['hotpot', 'musique', '2wiki'] if any(r['dataset'] == name for r in rows)}
    grid = [(name, variant, seed) for name in groups for variant in variants for seed in args.seeds]
    plan = {'datasets': {name: len(group) for name, group in groups.items()},
            'variants': [v.to_dict() for v in variants], 'seeds': args.seeds,
            'runs': len(grid), 'question_runs': sum(len(groups[name]) for name, _, _ in grid),
            'majority': majority_info, 'settings': asdict(settings)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'experiment.json', plan)
    for name, variant, seed in grid:
        summary = await run_cell(groups[name], variant, seed, settings, resources,
                                 root / name / variant.name / f'seed_{seed}',
                                 majority_info=majority_info, resume=args.resume)
        print(json.dumps(dict(dataset=name, variant=variant.name, seed=seed, **summary)))
    aggregate_runs(root, root / 'reports')
    return plan
