"""Command-line tools for DepPlan training, inference and evaluation."""
import argparse
import asyncio
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ['classify','evaluate-classifier']:
        q = commands.add_parser(name)
        q.add_argument('--input', required=True)
        q.add_argument('--checkpoint', required=True)
        q.add_argument('--encoder', default=None, help='BGE model ID or local directory')
        q.add_argument('--device', default='cpu')
        q.add_argument('--output', required=True)
    q = commands.add_parser('score-classification')
    q.add_argument('--input',required=True)
    q.add_argument('--output')
    q = commands.add_parser('train-classifier')
    q.add_argument('--input',required=True)
    q.add_argument('--output',required=True)
    q.add_argument('--encoder',default='BAAI/bge-base-en-v1.5')
    q.add_argument('--encoder-revision', help='Optional revision of the frozen encoder')
    q.add_argument('--device',default='cpu')
    q.add_argument('--seed',type=int,default=2026,
                   help='Random seed for a new classifier training run (default: 2026)')
    q = commands.add_parser('linearize')
    q.add_argument('--input',required=True)
    q.add_argument('--output',required=True)
    q = commands.add_parser('parse-amr')
    q.add_argument('--input',required=True)
    q.add_argument('--output',required=True)
    q.add_argument('--model-dir',required=True)
    q.add_argument('--device',default='cpu')
    q.add_argument('--structure-source',required=True)
    q.add_argument('--batch-size', type=int, default=4,
                   help='Questions per AMR parsing batch (default: 4)')
    q.add_argument('--parser-revision', help='Revision of the externally supplied parser weights')
    from .experiments.ablations.variants import VARIANTS
    for command in ['run-qa', 'run-ablations']:
        q = commands.add_parser(command)
        q.add_argument('--input', required=True, **({'nargs':'+'} if command=='run-ablations' else {}))
        q.add_argument('--checkpoint', required=True)
        q.add_argument('--output', required=True)
        q.add_argument('--encoder', default=None)
        q.add_argument('--device', default='cpu')
        q.add_argument('--config', help='Shared QA JSON settings')
        q.add_argument('--model', default=None)
        q.add_argument('--tokenizer', default=None,
                       help='Generation-model tokenizer ID or local directory for text controls')
        q.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL','http://localhost:8000/v1'))
        q.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY','EMPTY'))
        q.add_argument('--timeout', type=float, default=None)
        q.add_argument('--resume', action='store_true')
        semantic = q.add_mutually_exclusive_group()
        semantic.add_argument('--semantic-accuracy', dest='semantic_accuracy', action='store_true')
        semantic.add_argument('--no-semantic-accuracy', dest='semantic_accuracy', action='store_false',
                              help='Disable semantic-accuracy evaluation')
        q.set_defaults(semantic_accuracy=None)
        q.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'), help='Independent semantic evaluator model')
        q.add_argument('--judge-base-url', default=os.environ.get('JUDGE_BASE_URL'))
        q.add_argument('--judge-api-key', default=os.environ.get('JUDGE_API_KEY'))
        if command == 'run-qa':
            q.add_argument('--seed', type=int, default=2026)
        else:
            q.add_argument('--suite-config', help='Ablation suite JSON settings')
            q.add_argument('--seeds', type=int, nargs='+', default=None)
            q.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=None)
            q.add_argument('--training-manifest', default=None)
            q.add_argument('--dry-run', action='store_true', help='Validate inputs and print the run grid without API calls')
    q = commands.add_parser('aggregate-ablations')
    q.add_argument('--input', required=True)
    q.add_argument('--output', required=True)
    q = commands.add_parser('score-qa', help='Evaluate saved QA answers without repeating generation')
    q.add_argument('--input', required=True, help='DepPlan predictions.jsonl from one run')
    q.add_argument('--output', required=True, help='A separate evaluation output directory')
    q.add_argument('--judge-model', default=os.environ.get('JUDGE_MODEL'))
    q.add_argument('--judge-base-url', default=os.environ.get('JUDGE_BASE_URL'))
    q.add_argument('--judge-api-key', default=os.environ.get('JUDGE_API_KEY'))
    q.add_argument('--resume', action='store_true')
    q.add_argument('--retry-failed', action='store_true', help='Retry only previously failed evaluator decisions')
    args = p.parse_args()
    if args.command in {'run-qa', 'run-ablations'}:
        qa_config = json.loads(Path(args.config).read_text()) if args.config else {}
        args.model = args.model or qa_config.get('language_model', 'Qwen2.5-32B-Instruct')
        args.encoder = args.encoder or qa_config.get('encoder')
        if args.command == 'run-ablations':
            suite = json.loads(Path(args.suite_config).read_text()) if args.suite_config else {}
            if set(suite) - {'seeds','variants','training_manifest','tokenizer'}:
                p.error('Unknown ablation suite options')
            args.seeds = args.seeds or suite.get('seeds', qa_config.get('decoding_seeds', [42,123,2026]))
            args.variants = args.variants or suite.get('variants', list(VARIANTS))
            args.training_manifest = args.training_manifest or suite.get('training_manifest', 'data/classifier/train_manifest.jsonl')
            args.tokenizer = args.tokenizer or suite.get('tokenizer')
        args.tokenizer = args.tokenizer or 'Qwen/Qwen2.5-32B-Instruct'
    from .classifier.runtime import read_jsonl, write_jsonl, classification_metrics
    if args.command in ['classify','evaluate-classifier']:
        from .classifier.runtime import Classifier
        rows = read_jsonl(args.input)
        result = Classifier(args.checkpoint,args.encoder,args.device).predict(rows)
        write_jsonl(args.output,result)
        if args.command=='evaluate-classifier':
            metrics = classification_metrics(result)
            Path(args.output).with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            print(json.dumps(metrics,indent=2))
    elif args.command=='score-classification':
        metrics = classification_metrics(read_jsonl(args.input))
        rendered = json.dumps(metrics,indent=2)+'\n'
        if args.output:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(rendered)
        print(rendered)
    elif args.command=='train-classifier':
        from .classifier.train import train
        train(args)
    elif args.command=='linearize':
        from .semantic_units.linearization import build_semantic_units
        write_jsonl(args.output,[dict(row,**build_semantic_units(row['amr_triples']))
                                for row in read_jsonl(args.input) if row.get('status','success')=='success'])
    elif args.command=='parse-amr':
        from amrlib.models.model_factory import load_inference_model
        from .experiments.ablations.runner import _synchronize
        from .parsing import parse_questions
        if args.batch_size < 1:
            p.error('--batch-size must be positive')
        # The parser is selected entirely by --model-dir; pass a SPRING model
        # directory for the reference graph source.
        parser = load_inference_model(args.model_dir, device=args.device, batch_size=args.batch_size)
        output = parse_questions(read_jsonl(args.input), parser, args.structure_source,
                                 batch_size=args.batch_size, synchronize=lambda: _synchronize(args.device))
        for row in output:
            row.update(parser_model=Path(args.model_dir).name, parser_revision=args.parser_revision)
        write_jsonl(args.output,output)
    elif args.command=='run-qa':
        from .pipeline import run_qa
        asyncio.run(run_qa(args))
    elif args.command=='run-ablations':
        from .experiments.ablations.runner import run_suite
        asyncio.run(run_suite(args))
    elif args.command=='aggregate-ablations':
        from .experiments.ablations.reporting import aggregate_runs
        print(json.dumps(aggregate_runs(args.input, args.output), indent=2))
    elif args.command=='score-qa':
        from .score_qa import score_saved
        asyncio.run(score_saved(args))


if __name__=='__main__':
    main()
