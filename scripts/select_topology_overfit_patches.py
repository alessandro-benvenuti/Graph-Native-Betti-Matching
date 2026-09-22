#!/usr/bin/env python3
"""Select diverse training-patch topology errors for a mechanistic overfit test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loaders.discovery import discover_synthetic_mri
from data.loaders.io import read_vtp_graph
from training.evaluation.interactive_visualization import (
    classify_visualization_errors,
    load_prediction_records,
)
from training.evaluation.metrics import canonical_edges, graph_betti_numbers


CATEGORY_ORDER = (
    "genuine_loop_edge_break",
    "false_loop",
    "broken_h0",
    "false_positive_node",
    "correct_control",
)


def characterize(prediction, gt_nodes, gt_edges, *, max_node_distance: float):
    predicted_nodes = np.asarray(
        prediction.get("nodes_dhw", []), dtype=np.float32
    ).reshape(-1, 3)
    predicted_edges = canonical_edges(
        prediction.get("edges", []), len(predicted_nodes)
    )
    gt_nodes = np.asarray(gt_nodes, dtype=np.float32).reshape(-1, 3)
    gt_edges = canonical_edges(gt_edges, len(gt_nodes))
    errors = classify_visualization_errors(
        predicted_nodes,
        predicted_edges,
        gt_nodes,
        gt_edges,
        max_distance=max_node_distance,
    )
    target_beta0, target_beta1 = graph_betti_numbers(len(gt_nodes), gt_edges)
    predicted_beta0, predicted_beta1 = graph_betti_numbers(
        len(predicted_nodes), predicted_edges
    )
    matched_gt_fraction = (
        len(errors.predicted_to_gt) / len(gt_nodes) if len(gt_nodes) else 1.0
    )
    row = {
        "source_sample_id": str(prediction["source_sample_id"]),
        "target_nodes": int(len(gt_nodes)),
        "predicted_nodes": int(len(predicted_nodes)),
        "matched_nodes": int(len(errors.predicted_to_gt)),
        "matched_gt_fraction": float(matched_gt_fraction),
        "unmatched_target_nodes": int(len(errors.unmatched_gt_nodes)),
        "unmatched_predicted_nodes": int(len(errors.unmatched_predicted_nodes)),
        "target_edges": int(len(gt_edges)),
        "predicted_edges": int(len(predicted_edges)),
        "missing_gt_edges": int(len(errors.missing_gt_edges)),
        "false_positive_edges": int(len(errors.false_positive_edges)),
        "target_beta0": int(target_beta0),
        "predicted_beta0": int(predicted_beta0),
        "target_beta1": int(target_beta1),
        "predicted_beta1": int(predicted_beta1),
    }
    row["severity"] = float(
        3 * abs(predicted_beta1 - target_beta1)
        + 2 * abs(predicted_beta0 - target_beta0)
        + row["missing_gt_edges"]
        + row["false_positive_edges"]
        + row["unmatched_predicted_nodes"]
        + row["unmatched_target_nodes"]
    )
    return row


def category_candidates(rows):
    return {
        "genuine_loop_edge_break": [
            row
            for row in rows
            if row["target_beta1"] > 0
            and row["unmatched_target_nodes"] == 0
            and row["missing_gt_edges"] > 0
        ],
        "false_loop": [
            row
            for row in rows
            if row["predicted_beta1"] > row["target_beta1"]
            and row["matched_gt_fraction"] >= 0.8
        ],
        "broken_h0": [
            row
            for row in rows
            if row["predicted_beta0"] > row["target_beta0"]
            and row["matched_gt_fraction"] >= 0.8
        ],
        "false_positive_node": [
            row
            for row in rows
            if row["unmatched_predicted_nodes"] > 0
            and row["matched_gt_fraction"] >= 0.8
        ],
        "correct_control": [
            row
            for row in rows
            if row["predicted_beta0"] == row["target_beta0"]
            and row["predicted_beta1"] == row["target_beta1"]
            and row["unmatched_target_nodes"] == 0
            and row["unmatched_predicted_nodes"] == 0
            and row["missing_gt_edges"] == 0
            and row["false_positive_edges"] == 0
        ],
    }


def select_rows(rows, *, total: int):
    if total < len(CATEGORY_ORDER):
        raise ValueError(f"total must be at least {len(CATEGORY_ORDER)}")
    pools = category_candidates(rows)
    base, remainder = divmod(total, len(CATEGORY_ORDER))
    quotas = {
        category: base + int(index < remainder)
        for index, category in enumerate(CATEGORY_ORDER)
    }
    selected = []
    used = set()
    for category in CATEGORY_ORDER:
        reverse = category != "correct_control"
        candidates = sorted(
            pools[category],
            key=lambda row: (row["severity"], row["source_sample_id"]),
            reverse=reverse,
        )
        count = 0
        for row in candidates:
            sample_id = row["source_sample_id"]
            if sample_id in used:
                continue
            selected.append({**row, "selection_category": category})
            used.add(sample_id)
            count += 1
            if count >= quotas[category]:
                break
    if len(selected) < total:
        remaining = sorted(
            (row for row in rows if row["source_sample_id"] not in used),
            key=lambda row: (row["severity"], row["source_sample_id"]),
            reverse=True,
        )
        for row in remaining[: total - len(selected)]:
            selected.append({**row, "selection_category": "fallback_high_error"})
    return selected, {
        "requested": int(total),
        "selected": len(selected),
        "category_counts": {
            category: sum(
                row["selection_category"] == category for row in selected
            )
            for category in (*CATEGORY_ORDER, "fallback_high_error")
        },
        "category_pool_sizes": {
            category: len(values) for category, values in pools.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=25)
    parser.add_argument("--max-node-distance", type=float, default=0.1)
    args = parser.parse_args()
    records = load_prediction_records(args.predictions)
    dataset = {
        record.sample_id: record
        for record in discover_synthetic_mri(args.dataset_root, "train")
    }
    rows = []
    for prediction in records:
        sample_id = str(prediction.get("source_sample_id", ""))
        if sample_id not in dataset:
            raise KeyError(f"Prediction sample is absent from train split: {sample_id}")
        gt_nodes, gt_edges = read_vtp_graph(dataset[sample_id].graph)
        rows.append(
            characterize(
                prediction,
                gt_nodes,
                gt_edges,
                max_node_distance=args.max_node_distance,
            )
        )
    selected, summary = select_rows(rows, total=args.total)
    summary["split"] = "train"
    summary["screened_patches"] = len(rows)
    summary["max_node_distance"] = float(args.max_node_distance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_sample_ids.txt").write_text(
        "".join(row["source_sample_id"] + "\n" for row in selected),
        encoding="utf-8",
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
