"""Load method configurations and retain the origin of runtime overrides."""
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORDER = ('direct_prompting', 'cot', 'ircot', 'react', 'self_ask', 'plan_and_solve',
         'genground', 'dec', 'deeprag', 'r_search', 'coa', 'gear', 'belle', 'per')


def load_config(method, overrides=None):
    if method not in ORDER:
        raise ValueError(f'Unknown method: {method}')
    common = json.loads((ROOT / 'configs/common.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT / 'configs/methods' / f'{method}.json').read_text(encoding='utf-8'))
    sources = dict(common.pop('parameter_sources', {}))
    sources.update(config.pop('parameter_sources', {}))
    merged = dict(common, **config, parameter_sources=sources)
    for key, value in (overrides or {}).items():
        if value is not None or key == 'tokenizer_revision':
            merged[key] = value
            sources[key] = {'kind': 'override', 'reference': 'run command or override file',
                            'detail': 'Recorded with the run manifest.'}
    if merged.get('method') != method:
        raise ValueError('Configuration method identity does not match the selected workflow')
    for name in ['max_tokens', 'answer_max_tokens', 'api_attempts', 'max_steps']:
        if name in merged and (type(merged[name]) is not int or merged[name] < 1):
            raise ValueError(f'{name} must be a positive integer')
    if merged.get('retriever') not in {'dense', 'bm25', 'hybrid', 'none'}:
        raise ValueError('retriever must be dense, bm25, hybrid, or none')
    return copy.deepcopy(merged)


def methods():
    from .methods.classic import METHODS as classic
    from .methods.agents import METHODS as agents
    from .methods.learned import METHODS as learned
    merged = {**classic, **agents, **learned}
    if set(merged) != set(ORDER):
        raise ValueError(f'Method registry differs from configuration list: {set(merged) ^ set(ORDER)}')
    return merged
