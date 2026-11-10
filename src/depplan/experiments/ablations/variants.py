"""The full model and the seven component ablations."""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path


LABELS = ('sequential', 'parallel', 'hierarchical')


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    units: str = 'amr'
    selector: str = 'gcn'
    verifier: bool = True
    judge: bool = True

    def to_dict(self):
        return asdict(self)


VARIANTS = {v.name: v for v in (
    Variant('full', 'DepPlan (Full)'),
    Variant('text_surrogate', 'w/ Length-Matched Text Surrogate', units='text'),
    Variant('no_semantic_units', 'w/o Semantic Units', units='none'),
    Variant('no_planning_strategy', 'w/o Planning Strategy', selector='none'),
    Variant('direct_llm_strategy', 'w/ Direct LLM Strategy', selector='llm'),
    Variant('majority_class_strategy', 'w/ Majority-Class Strategy', selector='majority'),
    Variant('proposer_only', 'w/ Proposer Only', verifier=False, judge=False),
    Variant('proposer_verifier', 'w/ Proposer + Verifier', judge=False),
)}


DIRECT_STRATEGY_SYSTEM = '''Classify the retrieval-planning topology of the question.
Return exactly one JSON object: {"strategy":"sequential"}, {"strategy":"parallel"}, or {"strategy":"hierarchical"}.
Definitions:
sequential: one dependency chain; an intermediate entity or answer is needed to resolve the next step.
parallel: independent evidence branches, directly grounded in known entities, combined in a final comparison or aggregation.
hierarchical: a dependency chain AND independent branches/fork/join; either bridge-then-comparison or comparison-then-bridge.
Use only the question. Do not answer the question. Named constants alone are not evidence branches. Mentioning two entities alone does not imply parallel. If a compared attribute needs a dependent chain, choose hierarchical. Do not invent dependencies.
Examples:
"Where was the author of book X born?": sequential, find author then birthplace.
"Were A and B born in the same country?": parallel, independently find two known people's birth countries then compare.
"Were the authors of books X and Y born in the same country?": hierarchical, author-to-birthplace chains on parallel branches then compare.'''


def training_majority(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(r.get('source_split') != 'train' for r in rows):
        raise ValueError('Majority selection requires a nonempty training-split manifest')
    keys = [(r['dataset'], r['example_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate questions in the training manifest')
    counts = Counter(r['silver_strategy'] for r in rows)
    if set(counts) - set(LABELS):
        raise ValueError('Unknown strategy label in the training manifest')
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError('The training majority must be unique')
    return ranked[0][0], dict(counts), set(keys)
