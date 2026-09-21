#!/usr/bin/env python3
"""Compare the saved best-validation checkpoints of the paired Betti pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping

from scripts.summarize_node_edge_betti_pilot_4000 import FINAL_EPOCH, RUNS


PRIMARY_METRICS = (
    "node_mAP",
    "edge_mAP",
    "node_f1",
    "edge_f1",
    "beta0_absolute_error",
    "beta1_absolute_error",
    "smd",
)
DIAGNOSTIC_METRICS = (
    "node_precision",
    "node_recall",
    "edge_precision",
    "edge_recall",
    "predicted_nodes",
    "target_nodes",
    "predicted_edges",
    "target_edges",
    "predicted_beta0",
    "target_beta0",
    "predicted_beta1",
    "target_beta1",
)


def _finite_metric(record: Mapping, name: str, *, context: str) -> float:
    if name not in record:
        raise ValueError(f"Missing metric {name!r} in {context}")
    try:
        value = float(record[name])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Metric {name!r} is not numeric in {context}: {record[name]!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Metric {name!r} is not finite in {context}: {value}")
    return value


def _read_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No validation records in {path}")
    return records


def _best_metrics(run_dir: Path) -> dict[str, float | int | str]:
    status_path = run_dir / "training-status.json"
    selection_path = run_dir / "best-metric.json"
    history_path = run_dir / "validation-metrics.jsonl"
    for path in (status_path, selection_path, history_path):
        if not path.is_file():
            raise ValueError(f"Missing run artifact: {path}")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("epoch", -1)) != FINAL_EPOCH:
        raise ValueError(
            f"{run_dir.name} ended at epoch {status.get('epoch')}; "
            f"expected {FINAL_EPOCH}"
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metric = str(selection.get("metric", ""))
    mode = str(selection.get("mode", ""))
    if metric != "edge_mAP" or mode != "max":
        raise ValueError(
            f"{selection_path} selects {metric!r} with mode {mode!r}; "
            "expected maximum edge_mAP"
        )
    epoch = int(selection.get("epoch", -1))
    records = _read_records(history_path)
    record = next(
        (item for item in reversed(records) if int(item.get("epoch", -1)) == epoch),
        None,
    )
    if record is None:
        raise ValueError(f"No epoch-{epoch} validation record in {history_path}")

    selected_value = _finite_metric(record, metric, context=f"{run_dir.name}@{epoch}")
    recorded_value = _finite_metric(
        selection, "value", context=str(selection_path)
    )
    if not math.isclose(selected_value, recorded_value, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(
            f"Selected {metric}={recorded_value} does not match "
            f"epoch-{epoch} history value {selected_value}"
        )

    result: dict[str, float | int | str] = {
        "epoch": epoch,
        "selection_metric": metric,
    }
    for name in PRIMARY_METRICS:
        result[name] = _finite_metric(
            record, name, context=f"{run_dir.name}@{epoch}"
        )
    for name in DIAGNOSTIC_METRICS:
        if name in record:
            result[name] = _finite_metric(
                record, name, context=f"{run_dir.name}@{epoch}"
            )
    return result


def load_best_results(output_root: Path) -> dict[str, dict[str, float | int | str]]:
    return {
        label: _best_metrics(output_root / run_name)
        for label, run_name in RUNS.items()
    }


def _metric_rows(results: Mapping[str, Mapping]) -> tuple[str, ...]:
    optional = tuple(
        name
        for name in DIAGNOSTIC_METRICS
        if all(name in result for result in results.values())
    )
    return PRIMARY_METRICS + optional


def render_best(results: Mapping[str, Mapping]) -> str:
    control = results["control"]
    betti = results["node-aware Betti"]
    lines = [
        "Best-validation checkpoint comparison (one paired seed)",
        "Selection: maximum edge_mAP within each arm",
        "",
        f"{'checkpoint':<29} {'control':>10}  {'Betti':>10}",
        f"{'selected_epoch':<29} {int(control['epoch']):>10d}  "
        f"{int(betti['epoch']):>10d}",
        "",
        f"{'metric':<29} {'control':>10}  {'Betti':>10}  {'delta':>10}",
    ]
    for metric in _metric_rows(results):
        control_value = float(control[metric])
        betti_value = float(betti[metric])
        lines.append(
            f"{metric:<29} {control_value:>10.6f}  "
            f"{betti_value:>10.6f}  {betti_value - control_value:>+10.6f}"
        )
    lines.extend(
        (
            "",
            "Positive delta favors Betti for mAP/F1/precision/recall; negative "
            "delta favors Betti for beta errors and SMD.",
            "Predicted/target counts and Betti numbers are diagnostic quantities, "
            "not directional scores.",
            "Each column comes from its own edge-mAP-selected checkpoint; epochs "
            "may differ.",
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
        print(render_best(load_best_results(args.output_root)))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot summarize best Betti checkpoints: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
