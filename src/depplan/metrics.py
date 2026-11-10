from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment



def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(str(text).lower())).split())


def exact_match(reference: str, prediction: str) -> float:
    return float(normalize_answer(reference) == normalize_answer(prediction))


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalize_answer(reference).split()
    prediction_tokens = normalize_answer(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    common = Counter(reference_tokens) & Counter(prediction_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_scores(
    references: Sequence[str], prediction: str, *, dataset: str | None = None
) -> dict[str, float]:
    """Score answers; named datasets apply their official answer-F1 rules.

    ``dataset=None`` uses normalized exact match and token-overlap F1 across
    all references. MuSiQue accepts supplied aliases. HotpotQA and 2WikiQA
    use the primary answer, with zero F1 for mismatched yes/no/noanswer
    predictions or empty token lists.
    """
    if dataset not in {None, 'hotpot', 'musique', '2wiki'}:
        raise ValueError('Unknown dataset for answer scoring')
    usable = [str(reference) for reference in references if str(reference).strip()]
    if not usable:
        usable = [""]
    if dataset in {'hotpot', '2wiki'}:
        usable = usable[:1]

    def benchmark_f1(reference):
        if dataset in {'hotpot', '2wiki'}:
            gold, predicted = normalize_answer(reference), normalize_answer(prediction)
            special = {'yes', 'no', 'noanswer'}
            if ((gold in special or predicted in special) and gold != predicted
                    or not gold.split() or not predicted.split()):
                return 0.0
        return token_f1(reference, prediction)

    return {
        "em": max(exact_match(reference, prediction) for reference in usable),
        "f1": max(benchmark_f1(reference) for reference in usable),
    }


def references_from_original(original: dict[str, Any]) -> list[str]:
    answer = original.get("answer", "")
    if isinstance(answer, dict):
        primary = str(answer.get("text", ""))
        aliases = answer.get("aliases", [])
    else:
        primary = str(answer)
        aliases = original.get("answer_aliases", [])
    return [primary, *[str(alias) for alias in aliases]]


