#!/usr/bin/env python3
"""Summarize a paired mechanistic topology-overfit experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.evaluation.interactive_visualization import load_prediction_records
from training.evaluation.metrics import canonical_edges, graph_betti_numbers


METHODS = ("before", "control", "betti")
METRICS = (
    "node_count_absolute_error",
    "edge_count_absolute_error",
    "beta0_absolute_error",
    "beta1_absolute_error",
    "node_f1",
    "edge_f1",
)


def _index(path):
    records = load_prediction_records(path)
    return {str(record["source_sample_id"]): record for record in records}


def _value(record, name):
    metrics = record.get("metrics") or {}
    value = metrics.get(name)
    if value is not None:
        return float(value)
    if name in {"node_f1", "edge_f1"}:
        prefix = name.split("_", 1)[0]
        tp = float(metrics.get(prefix + "_tp", 0))
        fp = float(metrics.get(prefix + "_fp", 0))
        fn = float(metrics.get(prefix + "_fn", 0))
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 1.0
    nodes = np.asarray(record.get("nodes_dhw", []), dtype=np.float32).reshape(-1, 3)
    edges = canonical_edges(record.get("edges", []), len(nodes))
    beta0, beta1 = graph_betti_numbers(len(nodes), edges)
    if name == "beta0_absolute_error":
        return float(abs(beta0 - float(metrics.get("target_beta0", 0))))
    if name == "beta1_absolute_error":
        return float(abs(beta1 - float(metrics.get("target_beta1", 0))))
    return float("nan")


def build_rows(selection, predictions):
    rows = []
    for selected in selection:
        sample_id = str(selected["source_sample_id"])
        row = {
            "source_sample_id": sample_id,
            "selection_category": selected["selection_category"],
        }
        for method in METHODS:
            record = predictions[method][sample_id]
            for metric in METRICS:
                row[f"{method}_{metric}"] = _value(record, metric)
        for metric in METRICS:
            row[f"betti_minus_control_{metric}"] = (
                row[f"betti_{metric}"] - row[f"control_{metric}"]
            )
        rows.append(row)
    return rows


def summarize(rows):
    categories = sorted({row["selection_category"] for row in rows})

    def group_summary(group):
        result = {"patches": len(group)}
        for method in METHODS:
            for metric in METRICS:
                values = [float(row[f"{method}_{metric}"]) for row in group]
                finite = [value for value in values if np.isfinite(value)]
                result[f"{method}_{metric}"] = (
                    sum(finite) / len(finite) if finite else None
                )
        for metric in METRICS:
            values = [
                float(row[f"betti_minus_control_{metric}"]) for row in group
            ]
            finite = [value for value in values if np.isfinite(value)]
            result[f"betti_minus_control_{metric}"] = (
                sum(finite) / len(finite) if finite else None
            )
        return result

    return {
        "interpretation": "mechanistic train-set overfit; not a generalization estimate",
        "overall": group_summary(rows),
        "categories": {
            category: group_summary(
                [row for row in rows if row["selection_category"] == category]
            )
            for category in categories
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--betti", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    predictions = {
        "before": _index(args.before),
        "control": _index(args.control),
        "betti": _index(args.betti),
    }
    selected_ids = {str(row["source_sample_id"]) for row in selection}
    for method, records in predictions.items():
        missing = selected_ids - set(records)
        if missing:
            raise KeyError(f"{method} predictions miss samples: {sorted(missing)[:5]}")
    rows = build_rows(selection, predictions)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per-patch.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per-patch.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
