"""Single-run entry point for the shared planning and execution pipeline."""
import json

from .experiments.ablations.runner import Resources, Settings, load_questions, run_cell
from .experiments.ablations.variants import VARIANTS


async def run_qa(args):
    rows = load_questions([args.input])
    summary = await run_cell(rows, VARIANTS['full'], args.seed, Settings.from_args(args),
                             Resources(args), args.output, resume=args.resume)
    print(json.dumps(summary, indent=2))
