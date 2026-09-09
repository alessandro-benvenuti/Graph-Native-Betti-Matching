#!/usr/bin/env python3
"""Summarize the paired epoch-100 medium-scale FGW confirmation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Mapping


DEFAULT_SEEDS = (364505, 364506, 364507)
FINAL_EPOCH = 100
METHODS = {"hungarian": "hungarian", "fgw_a04": "FGW alpha=.4"}
METRICS = (
    "node_mAP",
    "edge_mAP",
    "node_f1",
    "edge_f1",
    "beta0_absolute_error",
    "beta1_absolute_error",
    "smd",
)


def run_name(method: str, seed: int) -> str:
    return f"fgw_medium_pretrained_continue_{method}_e100_seed{seed}"


def _read_final_metrics(run_dir: Path, final_epoch: int) -> Dict[str, float]:
    status_path = run_dir / "training-status.json"
    history_path = run_dir / "validation-metrics.jsonl"
    if not status_path.is_file():
        raise ValueError(f"Missing completion status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("epoch") != final_epoch:
        raise ValueError(
            f"Run {run_dir.name} ended at epoch {status.get('epoch')}; "
            f"expected {final_epoch}"
        )
    if status.get("reason") not in {"execution_stop", "max_epochs"}:
        raise ValueError(
            f"Run {run_dir.name} has completion reason {status.get('reason')!r}"
        )
    if not history_path.is_file():
        raise ValueError(f"Missing validation history: {history_path}")

    final_record = None
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {history_path}:{line_number}: {error.msg}"
            ) from error
        if record.get("epoch") == final_epoch:
            final_record = record
    if final_record is None:
        raise ValueError(
            f"No epoch-{final_epoch} metrics found in {history_path}"
        )

    metrics: Dict[str, float] = {}
    for metric in METRICS:
        value = final_record.get(metric)
        if value is None:
            raise ValueError(
                f"Missing finite metric {metric!r} for {run_dir.name} at "
                f"epoch {final_epoch}"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metric {metric!r} is not numeric for {run_dir.name}: {value!r}"
            ) from error
        if not math.isfinite(number):
            raise ValueError(
                f"Metric {metric!r} is not finite for {run_dir.name}: {number}"
            )
        metrics[metric] = number
    return metrics


def load_results(
    output_root: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    final_epoch: int = FINAL_EPOCH,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    seed_list = tuple(seeds)
    if len(seed_list) < 2:
        raise ValueError("At least two paired seeds are required")
    return {
        method: {
            seed: _read_final_metrics(
                output_root / run_name(method, seed), final_epoch
            )
            for seed in seed_list
        }
        for method in METHODS
    }


def _summary(values: Iterable[float]) -> tuple[float, float]:
    values = tuple(values)
    return mean(values), stdev(values)


def render_summary(
    results: Mapping[str, Mapping[int, Mapping[str, float]]],
) -> str:
    seeds = tuple(results["hungarian"])
    lines = [
        f"Final epoch-{FINAL_EPOCH} results",
        "seed      method         node mAP   edge mAP    node F1    edge F1"
        "     beta0 err   beta1 err        SMD",
    ]
    for seed in seeds:
        for method, label in METHODS.items():
            row = results[method][seed]
            lines.append(
                f"{seed:<9} {label:<14} "
                f"{row['node_mAP']:.6f}   {row['edge_mAP']:.6f}   "
                f"{row['node_f1']:.6f}   {row['edge_f1']:.6f}   "
                f"{row['beta0_absolute_error']:.6f}   "
                f"{row['beta1_absolute_error']:.6f}   {row['smd']:.6f}"
            )

    lines.extend(("", "Three-seed mean +/- sample standard deviation"))
    for method, label in METHODS.items():
        lines.extend(("", label))
        for metric in METRICS:
            average, deviation = _summary(
                results[method][seed][metric] for seed in seeds
            )
            lines.append(f"  {metric:<26} {average:.6f} +/- {deviation:.6f}")

    lines.extend(
        (
            "",
            "Paired FGW change versus Hungarian; positive favors FGW for mAP/F1",
            "and negative favors FGW for errors",
            "",
            METHODS["fgw_a04"],
        )
    )
    for metric in METRICS:
        average, deviation = _summary(
            results["fgw_a04"][seed][metric]
            - results["hungarian"][seed][metric]
            for seed in seeds
        )
        lines.append(
            f"  delta {metric:<20} {average:+.6f} +/- {deviation:.6f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(os.environ["GNBM_OUTPUT_DIR"])
            if "GNBM_OUTPUT_DIR" in os.environ
            else None
        ),
        required="GNBM_OUTPUT_DIR" not in os.environ,
        help="Training output directory (defaults to GNBM_OUTPUT_DIR).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = load_results(args.output_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot summarize FGW confirmation: {error}") from error
    print(render_summary(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
