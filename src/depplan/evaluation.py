"""Independent answer evaluation, excluded from generation time and tokens."""
import time

from .executor import semantic_accuracy
from .llm import CallUsage, sum_usage


async def score_prediction(row, judge):
    result = dict(semantic_correct=None, judge_status='disabled',
                  judge_seconds=0.0, judge_usage=sum_usage([]))
    if judge is None:
        return result
    if row.get('status') != 'success':
        # A QA failure stays in the full denominator, without an evaluator call.
        result.update(semantic_correct=0, judge_status='not_needed')
        return result
    judge.example_id = row['example_id']
    judge.records.clear()
    started = time.perf_counter()
    try:
        reference = row.get('gold_answers', row.get('references'))
        if not isinstance(reference, list) or not reference or not all(isinstance(x, str) for x in reference):
            raise ValueError('Semantic evaluation requires the saved reference answers')
        correct, _ = await semantic_accuracy(judge, row['question'], reference, row['prediction'])
        result.update(semantic_correct=correct, judge_status='success')
    except Exception as exc:
        # Evaluator failure is missing measurement, not a wrong QA prediction.
        result.update(judge_status='error', judge_error_type=type(exc).__name__)
    finally:
        result['judge_seconds'] = time.perf_counter() - started
        result['judge_usage'] = sum_usage([CallUsage(**r['usage']) for r in judge.records])
    return result


def semantic_summary(rows):
    values = [r.get('semantic_correct') for r in rows]
    for row, value in zip(rows, values):
        if value is not None and not (type(value) is int and value in (0, 1)):
            raise ValueError('Semantic correctness must be 0, 1 or null')
        if row.get('status') != 'success' and value not in (None, 0):
            raise ValueError('Failed QA questions must have zero semantic correctness')
    complete = bool(rows) and all(v is not None for v in values)
    correct = sum(values) if complete else None
    accuracy = 100 * correct / len(rows) if complete else None
    return dict(acc_correct=correct, acc_percent=accuracy,
                semantic_accuracy_percent=accuracy,
                semantic_evaluation_complete=complete,
                semantic_scored_questions=sum(v is not None for v in values),
                semantic_missing_questions=sum(v is None for v in values),
                semantic_judge_failures=sum(r.get('judge_status') == 'error' for r in rows))
