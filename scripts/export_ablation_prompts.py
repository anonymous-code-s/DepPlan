"""Export the shared prompts and component-control specification."""
import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from depplan import prompts
from depplan.experiments.ablations.variants import DIRECT_STRATEGY_SYSTEM, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = ['PLAN_SCHEMA', 'PROPOSER_SYSTEM', 'VERIFIER_SYSTEM', 'JUDGE_SYSTEM',
             'ANSWER_SYSTEM', 'FINAL_ANSWER_SYSTEM']
    value = {
        'shared_prompts': {name: getattr(prompts, name) for name in names},
        'direct_strategy_system': DIRECT_STRATEGY_SYSTEM,
        'direct_strategy_user': 'Question:\n{question}',
        'planning_context_renderer': inspect.getsource(prompts.planning_context),
        'variants': [variant.to_dict() for variant in VARIANTS.values()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
