#!/usr/bin/env python3
"""Compare final validation metrics for the paired 4,000-patch Betti pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


FINAL_EPOCH = 100
RUNS = {
    "control": "node_edge_betti_pilot4000_control_e100_seed364505",
    "node-aware Betti": "node_edge_betti_pilot4000_nodeaware_e100_seed364505",
}
METRICS = (
    "node_mAP",
    "edge_mAP",
    "node_f1",
    "edge_f1",
    "beta0_absolute_error",
    "beta1_absolute_error",
    "smd",
)


def _final_metrics(run_dir: Path) -> dict[str, float]:
    status_path = run_dir / "training-status.json"
    history_path = run_dir / "validation-metrics.jsonl"
    if not status_path.is_file() or not history_path.is_file():
        raise ValueError(f"Incomplete run artifacts in {run_dir}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("epoch", -1)) != FINAL_EPOCH:
        raise ValueError(
            f"{run_dir.name} ended at epoch {status.get('epoch')}; "
            f"expected {FINAL_EPOCH}"
        )
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = next(
        (record for record in reversed(records) if record.get("epoch") == FINAL_EPOCH),
        None,
    )
    if final is None:
        raise ValueError(f"No epoch-{FINAL_EPOCH} validation record in {history_path}")
    result = {}
    for metric in METRICS:
        value = float(final[metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {metric} in {run_dir.name}")
        result[metric] = value
    return result


def load_results(output_root: Path) -> dict[str, dict[str, float]]:
    return {
        label: _final_metrics(output_root / run_name)
        for label, run_name in RUNS.items()
    }


def render(results: dict[str, dict[str, float]]) -> str:
    control = results["control"]
    betti = results["node-aware Betti"]
    lines = [
        "Final epoch-100 validation comparison (one paired seed)",
        "metric                         control       Betti        delta",
    ]
    for metric in METRICS:
        delta = betti[metric] - control[metric]
        lines.append(
            f"{metric:<29} {control[metric]:>10.6f}  "
            f"{betti[metric]:>10.6f}  {delta:>+10.6f}"
        )
    lines.extend(
        (
            "",
            "Positive delta favors Betti for mAP/F1; negative delta favors Betti "
            "for beta errors and SMD.",
            "This single-seed pilot measures viability, not statistical significance.",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ["GNBM_OUTPUT_DIR"])
        if "GNBM_OUTPUT_DIR" in os.environ
        else None,
        required="GNBM_OUTPUT_DIR" not in os.environ,
    )
    args = parser.parse_args()
    try:
        print(render(load_results(args.output_root)))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot summarize Betti pilot: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
