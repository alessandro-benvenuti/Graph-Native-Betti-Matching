#!/usr/bin/env python3
"""Compare hard predicted cycle spaces on patches containing GT loops."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
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
from training.losses.betti_h1 import compute_cycle_space_matching


METHODS = ("control", "betti")
CONFIDENCE_THRESHOLDS = (0.1, 0.25, 0.5)


def _prediction_index(records, *, label: str):
    result = {}
    for record in records:
        sample_id = str(record.get("source_sample_id", ""))
        if not sample_id:
            raise ValueError(f"{label} prediction is missing source_sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate {label} prediction for {sample_id}")
        result[sample_id] = record
    return result


def _canonical_pair(left, right):
    return tuple(sorted((int(left), int(right))))


def _target_cycle_confidence_diagnostics(
    prediction,
    classification,
    gt_edges,
):
    """Explain GT basis-cycle failures using all exported pair probabilities.

    The individual cycle representatives come from a deterministic spanning-
    forest basis. Counts of representatives in each failure category are a
    diagnostic; the cycle-space ranks remain the basis-invariant headline
    metrics.
    """

    gt_edges = tuple(_canonical_pair(*edge) for edge in gt_edges)
    target_matching = compute_cycle_space_matching(
        [1.0] * len(gt_edges),
        gt_edges,
        gt_edges,
        num_vertices=(max((max(edge) for edge in gt_edges), default=-1) + 1),
    )
    predicted_to_gt = dict(classification.predicted_to_gt)
    gt_to_predicted = {gt: predicted for predicted, gt in predicted_to_gt.items()}
    hard_edges = {
        _canonical_pair(*edge) for edge in prediction.get("edges", [])
    }

    score_map = None
    if "all_candidate_edges" in prediction:
        candidate_edges = prediction.get("all_candidate_edges", [])
        candidate_scores = prediction.get("all_candidate_edge_scores", [])
        if len(candidate_edges) != len(candidate_scores):
            raise ValueError(
                "all_candidate_edges and all_candidate_edge_scores must agree"
            )
        score_map = {
            _canonical_pair(*edge): float(score)
            for edge, score in zip(candidate_edges, candidate_scores)
        }

    diagnostics = []
    for index, cycle in enumerate(target_matching.target_classes):
        cycle_edges = tuple(_canonical_pair(*edge) for edge in cycle.cycle_edges)
        cycle_vertices = sorted({vertex for edge in cycle_edges for vertex in edge})
        missing_nodes = [
            vertex for vertex in cycle_vertices if vertex not in gt_to_predicted
        ]
        item = {
            "basis_index": index,
            "gt_cycle_edges": [list(edge) for edge in cycle_edges],
            "missing_gt_nodes": missing_nodes,
            "scores_available": score_map is not None,
        }
        if missing_nodes:
            item.update(
                {
                    "failure_mode": "missing_node",
                    "hard_closed": False,
                    "missing_hard_gt_edges": [],
                    "weakest_edge_score": None,
                    "bottleneck_gt_edge": None,
                }
            )
            diagnostics.append(item)
            continue

        mapped_edges = {
            edge: _canonical_pair(
                gt_to_predicted[edge[0]], gt_to_predicted[edge[1]]
            )
            for edge in cycle_edges
        }
        missing_hard = [
            edge for edge, mapped in mapped_edges.items() if mapped not in hard_edges
        ]
        hard_closed = not missing_hard
        item.update(
            {
                "failure_mode": "hard_closed" if hard_closed else "edge_below_decision",
                "hard_closed": hard_closed,
                "missing_hard_gt_edges": [list(edge) for edge in missing_hard],
            }
        )
        if score_map is None:
            item.update(
                {"weakest_edge_score": None, "bottleneck_gt_edge": None}
            )
        else:
            missing_pairs = [
                mapped for mapped in mapped_edges.values() if mapped not in score_map
            ]
            if missing_pairs:
                raise ValueError(
                    "All-pair prediction export is incomplete for retained nodes: "
                    f"{missing_pairs[:3]}"
                )
            edge_scores = {
                edge: score_map[mapped] for edge, mapped in mapped_edges.items()
            }
            bottleneck = min(edge_scores, key=edge_scores.get)
            item.update(
                {
                    "weakest_edge_score": edge_scores[bottleneck],
                    "bottleneck_gt_edge": list(bottleneck),
                    "gt_cycle_edge_scores": [
                        {
                            "gt_edge": list(edge),
                            "score": edge_scores[edge],
                            "hard_retained": mapped_edges[edge] in hard_edges,
                        }
                        for edge in cycle_edges
                    ],
                }
            )
            for threshold in CONFIDENCE_THRESHOLDS:
                item[f"closed_above_{threshold:g}"] = all(
                    score > threshold for score in edge_scores.values()
                )
        diagnostics.append(item)
    return diagnostics


def spatial_cycle_result(
    prediction,
    gt_nodes,
    gt_edges,
    *,
    max_node_distance: float,
):
    predicted_nodes = np.asarray(
        prediction.get("nodes_dhw", []), dtype=np.float32
    ).reshape(-1, 3)
    predicted_edges = canonical_edges(prediction.get("edges", []), len(predicted_nodes))
    gt_nodes = np.asarray(gt_nodes, dtype=np.float32).reshape(-1, 3)
    gt_edges = canonical_edges(gt_edges, len(gt_nodes))
    classification = classify_visualization_errors(
        predicted_nodes,
        predicted_edges,
        gt_nodes,
        gt_edges,
        max_distance=max_node_distance,
    )
    target_cycle_diagnostics = _target_cycle_confidence_diagnostics(
        prediction, classification, gt_edges
    )

    mapping = dict(classification.predicted_to_gt)
    next_vertex = len(gt_nodes)
    for predicted_index in range(len(predicted_nodes)):
        if predicted_index not in mapping:
            mapping[predicted_index] = next_vertex
            next_vertex += 1
    mapped_prediction_edges = {
        tuple(sorted((mapping[left], mapping[right])))
        for left, right in predicted_edges
    }
    target_edge_set = set(gt_edges)
    candidate_edges = tuple(sorted(mapped_prediction_edges | target_edge_set))
    probabilities = tuple(
        1.0 if edge in mapped_prediction_edges else 0.0
        for edge in candidate_edges
    )
    matching = compute_cycle_space_matching(
        probabilities,
        candidate_edges,
        target_edge_set,
        num_vertices=next_vertex,
    )
    predicted_beta0, predicted_beta1 = graph_betti_numbers(
        len(predicted_nodes), predicted_edges
    )
    target_beta0, target_beta1 = graph_betti_numbers(len(gt_nodes), gt_edges)
    shared = int(matching.shared_rank)
    false = int(matching.false_prediction_rank)
    missed = int(matching.missed_target_rank)
    if shared + false != predicted_beta1:
        raise RuntimeError("Predicted cycle-space rank decomposition is inconsistent")
    if shared + missed != target_beta1:
        raise RuntimeError("Target cycle-space rank decomposition is inconsistent")
    return {
        "target_nodes": int(len(gt_nodes)),
        "predicted_nodes": int(len(predicted_nodes)),
        "matched_nodes": int(len(classification.predicted_to_gt)),
        "unmatched_predicted_nodes": int(len(classification.unmatched_predicted_nodes)),
        "unmatched_target_nodes": int(len(classification.unmatched_gt_nodes)),
        "target_edges": int(len(gt_edges)),
        "predicted_edges": int(len(predicted_edges)),
        "target_beta0": int(target_beta0),
        "predicted_beta0": int(predicted_beta0),
        "target_beta1": int(target_beta1),
        "predicted_beta1": int(predicted_beta1),
        "beta1_absolute_error": int(abs(predicted_beta1 - target_beta1)),
        "shared_cycle_rank": shared,
        "false_cycle_rank": false,
        "missed_cycle_rank": missed,
        "comparison_only_cycle_rank": int(matching.union_only_rank),
        "beta1_count_exact": bool(predicted_beta1 == target_beta1),
        "spatial_cycle_complete": bool(missed == 0),
        "spatial_cycle_exact": bool(false == 0 and missed == 0),
        "target_cycle_diagnostics": target_cycle_diagnostics,
    }


def compare_predictions(
    control_records,
    betti_records,
    dataset_records,
    *,
    graph_reader=read_vtp_graph,
    max_node_distance: float = 0.1,
):
    if max_node_distance < 0:
        raise ValueError("max_node_distance must be non-negative")
    predictions = {
        "control": _prediction_index(control_records, label="control"),
        "betti": _prediction_index(betti_records, label="betti"),
    }
    if set(predictions["control"]) != set(predictions["betti"]):
        only_control = sorted(set(predictions["control"]) - set(predictions["betti"]))
        only_betti = sorted(set(predictions["betti"]) - set(predictions["control"]))
        raise ValueError(
            "Prediction sample sets differ: only_control={} only_betti={}".format(
                only_control[:5], only_betti[:5]
            )
        )
    dataset_by_id = {record.sample_id: record for record in dataset_records}
    missing = sorted(set(predictions["control"]) - set(dataset_by_id))
    if missing:
        raise ValueError(f"Prediction samples are absent from the dataset: {missing[:5]}")

    rows = []
    for sample_id in predictions["control"]:
        gt_nodes, gt_edges = graph_reader(dataset_by_id[sample_id].graph)
        _, target_beta1 = graph_betti_numbers(len(gt_nodes), gt_edges)
        if target_beta1 <= 0:
            raise ValueError(f"Prediction subset contains non-loop GT patch {sample_id}")
        method_results = {
            method: spatial_cycle_result(
                predictions[method][sample_id],
                gt_nodes,
                gt_edges,
                max_node_distance=max_node_distance,
            )
            for method in METHODS
        }
        control_exact = method_results["control"]["spatial_cycle_exact"]
        betti_exact = method_results["betti"]["spatial_cycle_exact"]
        if control_exact and betti_exact:
            outcome = "both_exact"
        elif not control_exact and betti_exact:
            outcome = "betti_fixed"
        elif control_exact and not betti_exact:
            outcome = "betti_regressed"
        else:
            outcome = "both_inexact"
        rows.append(
            {
                "source_sample_id": sample_id,
                "outcome": outcome,
                **{
                    f"{method}_{key}": value
                    for method, result in method_results.items()
                    for key, value in result.items()
                },
            }
        )
    return rows


def _method_summary(rows, method: str):
    shared = sum(int(row[f"{method}_shared_cycle_rank"]) for row in rows)
    false = sum(int(row[f"{method}_false_cycle_rank"]) for row in rows)
    missed = sum(int(row[f"{method}_missed_cycle_rank"]) for row in rows)
    predicted = shared + false
    target = shared + missed
    cycle_diagnostics = [
        item
        for row in rows
        for item in row[f"{method}_target_cycle_diagnostics"]
    ]
    weakest_scores = [
        float(item["weakest_edge_score"])
        for item in cycle_diagnostics
        if item["weakest_edge_score"] is not None
    ]
    summary = {
        "patches": len(rows),
        "target_cycle_rank": target,
        "predicted_cycle_rank": predicted,
        "shared_cycle_rank": shared,
        "false_cycle_rank": false,
        "missed_cycle_rank": missed,
        "cycle_precision": shared / predicted if predicted else 0.0,
        "cycle_recall": shared / target if target else 0.0,
        "spatial_exact_patches": sum(
            bool(row[f"{method}_spatial_cycle_exact"]) for row in rows
        ),
        "spatial_exact_rate": sum(
            bool(row[f"{method}_spatial_cycle_exact"]) for row in rows
        )
        / len(rows),
        "spatial_complete_patches": sum(
            bool(row[f"{method}_spatial_cycle_complete"]) for row in rows
        ),
        "beta1_count_exact_patches": sum(
            bool(row[f"{method}_beta1_count_exact"]) for row in rows
        ),
        "mean_beta1_absolute_error": sum(
            float(row[f"{method}_beta1_absolute_error"]) for row in rows
        )
        / len(rows),
        "target_basis_cycles": len(cycle_diagnostics),
        "basis_cycles_missing_node": sum(
            item["failure_mode"] == "missing_node" for item in cycle_diagnostics
        ),
        "basis_cycles_broken_by_hard_edge": sum(
            item["failure_mode"] == "edge_below_decision"
            for item in cycle_diagnostics
        ),
        "basis_cycles_hard_closed": sum(
            item["failure_mode"] == "hard_closed" for item in cycle_diagnostics
        ),
        "all_edge_scores_available_patches": sum(
            all(
                item["scores_available"]
                for item in row[f"{method}_target_cycle_diagnostics"]
            )
            for row in rows
        ),
        "representable_cycle_weakest_score_mean": (
            sum(weakest_scores) / len(weakest_scores) if weakest_scores else None
        ),
        "representable_cycle_weakest_score_median": (
            statistics.median(weakest_scores) if weakest_scores else None
        ),
    }
    for threshold in CONFIDENCE_THRESHOLDS:
        summary[f"basis_cycles_closed_above_{threshold:g}"] = sum(
            bool(item.get(f"closed_above_{threshold:g}", False))
            for item in cycle_diagnostics
        )
    return summary


def summarize(rows, *, max_node_distance: float):
    if not rows:
        raise ValueError("Cannot summarize an empty loop-patch comparison")
    outcomes = {
        name: sum(row["outcome"] == name for row in rows)
        for name in ("both_exact", "betti_fixed", "betti_regressed", "both_inexact")
    }
    return {
        "max_node_distance": float(max_node_distance),
        "loop_patches": len(rows),
        "outcomes": outcomes,
        "methods": {method: _method_summary(rows, method) for method in METHODS},
    }


def _csv_value(value):
    if isinstance(value, bool):
        return int(value)
    return value


def write_report(output_dir: Path, rows, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        field for field in rows[0] if not field.endswith("target_cycle_diagnostics")
    ]
    with (output_dir / "per-patch.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    (output_dir / "per-patch.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for outcome in ("betti_fixed", "betti_regressed", "both_inexact", "both_exact"):
        (output_dir / f"{outcome}_sample_ids.txt").write_text(
            "".join(
                row["source_sample_id"] + "\n"
                for row in rows
                if row["outcome"] == outcome
            ),
            encoding="utf-8",
        )


def render_summary(summary):
    control = summary["methods"]["control"]
    betti = summary["methods"]["betti"]
    metrics = (
        "predicted_cycle_rank",
        "shared_cycle_rank",
        "false_cycle_rank",
        "missed_cycle_rank",
        "cycle_precision",
        "cycle_recall",
        "spatial_exact_patches",
        "spatial_exact_rate",
        "spatial_complete_patches",
        "beta1_count_exact_patches",
        "mean_beta1_absolute_error",
        "basis_cycles_missing_node",
        "basis_cycles_broken_by_hard_edge",
        "basis_cycles_hard_closed",
        "basis_cycles_closed_above_0.1",
        "basis_cycles_closed_above_0.25",
        "basis_cycles_closed_above_0.5",
        "representable_cycle_weakest_score_mean",
        "representable_cycle_weakest_score_median",
    )
    lines = [
        "GT-loop validation-patch comparison",
        f"patches={summary['loop_patches']} target_cycle_rank={control['target_cycle_rank']} "
        f"max_node_distance={summary['max_node_distance']:.4f}",
        "",
        f"{'metric':<41} {'control':>12} {'Betti':>12} {'delta':>12}",
    ]
    for name in metrics:
        left_value, right_value = control[name], betti[name]
        if left_value is None or right_value is None:
            lines.append(
                f"{name:<41} {str(left_value):>12} {str(right_value):>12} {'n/a':>12}"
            )
            continue
        left, right = float(left_value), float(right_value)
        lines.append(
            f"{name:<41} {left:>12.6f} {right:>12.6f} {right-left:>+12.6f}"
        )
    lines.extend(("", "Outcome counts:", json.dumps(summary["outcomes"], sort_keys=True)))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--betti-predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-node-distance", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    rows = compare_predictions(
        load_prediction_records(args.control_predictions),
        load_prediction_records(args.betti_predictions),
        discover_synthetic_mri(args.dataset_root, args.split),
        max_node_distance=args.max_node_distance,
    )
    summary = summarize(rows, max_node_distance=args.max_node_distance)
    write_report(args.output_dir, rows, summary)
    print(render_summary(summary))
    print(f"Detailed output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
