from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(
        Path(args.autoregressive).read_text(encoding="utf-8")
    )
    residual = json.loads(
        Path(args.residual).read_text(encoding="utf-8")
    )
    if baseline["method"] != "autoregressive":
        raise ValueError("--autoregressive is not an AR report.")
    if residual["method"] != "residual":
        raise ValueError("--residual is not a residual report.")

    report = {
        "autoregressive": baseline,
        "residual": residual,
        "speedup": (
            baseline["total_timed_seconds"]
            / residual["total_timed_seconds"]
        ),
        "throughput_ratio": (
            residual["throughput_samples_per_second"]
            / baseline["throughput_samples_per_second"]
        ),
        "metric_delta": {
            key: residual[key] - baseline[key]
            for key in (
                "recall_at_beam",
                "mrr_at_beam",
                "ndcg_at_beam",
                "exact_at_1",
            )
        },
        "layer_top1_accuracy_delta": [
            improved - base
            for improved, base in zip(
                residual["layer_top1_accuracy"],
                baseline["layer_top1_accuracy"],
            )
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
