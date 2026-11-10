"""Join published baseline scores with the released three-seed summaries."""

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = {"EM": "em_percent", "F1": "f1_percent"}
FIELDS = [
    "method", "original_variant", "original_model", "dataset", "metric",
    "published_percent", "this_study_mean", "this_study_sample_sd",
    "source_table", "source_url",
]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def display(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path,
                        default=ROOT / "baselines/references/published_results.csv")
    parser.add_argument("--summaries", type=Path,
                        default=ROOT / "results/main_results/mean_sample_sd.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "baselines/results/published_comparison.csv")
    args = parser.parse_args()

    summaries = {}
    for row in read_csv(args.summaries):
        key = row["method"], row["dataset"]
        if key in summaries:
            raise ValueError(f"Duplicate summary: {key}")
        summaries[key] = row

    comparison = []
    for row in read_csv(args.published):
        key = row["method"], row["dataset"]
        if key not in summaries:
            raise ValueError(f"Missing summary: {key}")
        metric = METRICS[row["metric"]]
        summary = summaries[key]
        comparison.append({
            **row,
            "this_study_mean": display(summary[f"{metric}_mean"]),
            "this_study_sample_sd": display(summary[f"{metric}_sample_sd"]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Wrote {len(comparison)} comparison rows to {args.output.name}")


if __name__ == "__main__":
    main()
