"""Score saved sub-questions for length, fluency and reference alignment."""
import argparse
from collections import defaultdict
from statistics import mean

from .common import (gold_subquestions, load_gold, load_runs, new_output, parse_config,
                     plan_subquestions, question_key, read_jsonl, seed_summary,
                     source_files, validate_seed_cohorts, write_csv, write_json, write_jsonl)


METRICS = ['length_tokens', 'nll', 'semantic_similarity', 'wellformedness']


def average_complete(rows, field):
    values = [r.get(field) for r in rows]
    return mean(values) if values and all(v is not None for v in values) else None


class FluencyScorer:
    def __init__(self, model='gpt2', device='cpu', length_only=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = None if length_only else AutoModelForCausalLM.from_pretrained(model).to(device).eval()
        self.device = device

    def score(self, text):
        import torch
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.model is None or len(ids) < 2:
            return len(ids), None
        limit = getattr(self.model.config, 'max_position_embeddings', self.tokenizer.model_max_length)
        if len(ids) > limit:
            raise ValueError('Sub-question exceeds the fluency model context; no truncation is applied')
        tensor = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            nll = self.model(input_ids=tensor, labels=tensor).loss.item()
        return len(ids), nll


class WellformedScorer:
    def __init__(self, model, positive_label_id, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        self.positive_label_id, self.device = positive_label_id, device
        if not 0 <= positive_label_id < self.model.config.num_labels:
            raise ValueError('positive-label-id is outside the classifier labels')

    def score(self, text):
        import torch
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False).to(self.device)
        with torch.inference_mode():
            return self.model(**tokens).logits.softmax(-1)[0, self.positive_label_id].item()


def score_key(row, index=None):
    return (row['method'], row['seed'], *question_key(row), row['subquestion_index'] if index is None else index)


def load_wellformed(path):
    scores = {}
    for row in read_jsonl(path):
        key = score_key(row)
        value = row['score']
        if key in scores or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError('Duplicate or invalid well-formedness score')
        scores[key] = value
    return scores


def score(rows, fluency, gold=None, encoder=None, wellformed=None, saved_scores=None):
    subquestions, questions, used_scores = [], [], set()
    for row in rows:
        texts = plan_subquestions(row)
        similarities = [None] * len(texts)
        if texts and row['dataset'] == 'musique' and gold is not None:
            reference = gold.get(question_key(row))
            if reference is None or not gold_subquestions(reference):
                raise ValueError('MuSiQue reference decomposition is missing')
            generated = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            target = encoder.encode(gold_subquestions(reference), normalize_embeddings=True, show_progress_bar=False)
            similarities = (generated @ target.T).max(axis=1).tolist()
        scored = []
        for i, text in enumerate(texts):
            length, nll = fluency.score(text)
            key = score_key(row, i)
            if saved_scores is not None:
                if key not in saved_scores:
                    raise ValueError('Missing well-formedness score for ' + str(key))
                wf = saved_scores[key]
                used_scores.add(key)
            else:
                wf = wellformed.score(text) if wellformed is not None else None
            item = dict(method=row['method'], dataset=row['dataset'], example_id=row['example_id'],
                        seed=row['seed'], subquestion_index=i, subquestion=text,
                        length_tokens=length, nll=nll, semantic_similarity=similarities[i], wellformedness=wf)
            scored.append(item)
        subquestions.extend(scored)
        questions.append(dict(method=row['method'], dataset=row['dataset'], example_id=row['example_id'],
                              seed=row['seed'], subquestions=len(scored),
                              **{metric: average_complete(scored, metric) for metric in METRICS}))
    if saved_scores is not None and used_scores != set(saved_scores):
        raise ValueError('Well-formedness file contains scores outside the supplied cohort')
    return subquestions, questions


def summarize(subquestions, questions):
    groups = defaultdict(list)
    for row in subquestions:
        groups[row['method'], row['dataset'], row['seed']].append(row)
    cohorts = defaultdict(list)
    for row in questions:
        cohorts[row['method'], row['dataset'], row['seed']].append(row)
    result = []
    for key, cohort in sorted(cohorts.items()):
        scored = groups[key]
        result.append(dict(method=key[0], dataset=key[1], seed=key[2], questions=len(cohort),
                           questions_without_subquestions=sum(r['subquestions'] == 0 for r in cohort),
                           subquestions=len(scored),
                           **{metric: average_complete(scored, metric) for metric in METRICS}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gold', help='MuSiQue gold decompositions, required for semantic similarity')
    parser.add_argument('--fluency-model', default='gpt2')
    parser.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--length-only', action='store_true')
    wf = parser.add_mutually_exclusive_group()
    wf.add_argument('--wellformed-model')
    wf.add_argument('--wellformed-scores')
    parser.add_argument('--positive-label-id', type=int)
    args = parse_config(parser, argv)
    if args.wellformed_model and args.positive_label_id is None:
        parser.error('--wellformed-model requires --positive-label-id')
    rows = load_runs(args.predictions)
    validate_seed_cohorts(rows)
    gold = load_gold(args.gold) if args.gold else None
    if not args.length_only and any(r['dataset'] == 'musique' for r in rows) and gold is None:
        parser.error('Supply --gold for MuSiQue semantic similarity or use --length-only')
    encoder = None
    if gold and not args.length_only:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.encoder, device=args.device)
    fluency = FluencyScorer(args.fluency_model, args.device, args.length_only)
    wellformed = WellformedScorer(args.wellformed_model, args.positive_label_id, args.device) if args.wellformed_model else None
    scores = load_wellformed(args.wellformed_scores) if args.wellformed_scores else None
    subquestions, questions = score(rows, fluency, gold if encoder else None, encoder, wellformed, scores)
    output = new_output(args.output)
    sources = args.predictions + ([args.gold] if args.gold else []) + ([args.wellformed_scores] if args.wellformed_scores else [])
    write_json(output / 'configuration.json', {'experiment': 'subquestion_quality', 'sources': source_files(sources),
        'fluency_model': args.fluency_model, 'length_only': args.length_only, 'device': args.device,
        'encoder': args.encoder if encoder else None, 'wellformed_model': args.wellformed_model,
        'positive_label_id': args.positive_label_id, 'aggregation': 'equal weight per generated sub-question',
        'similarity': 'maximum cosine to any gold sub-question, without a threshold',
        'wellformedness': 'positive-class probability or supplied per-sub-question score'})
    write_jsonl(output / 'subquestions.jsonl', subquestions)
    write_jsonl(output / 'per_question.jsonl', questions)
    per_seed = summarize(subquestions, questions)
    write_csv(output / 'per_seed.csv', per_seed)
    write_csv(output / 'mean_sample_sd.csv', seed_summary(per_seed, ['method', 'dataset'], METRICS))


if __name__ == '__main__':
    main()
