#!/usr/bin/env python3
"""Render representative control/Betti loop patches as interactive 3D HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

def select_examples(rows, *, limit: int):
    """Choose diverse examples without treating this small subset as an estimate."""

    selected = []
    seen = set()

    def add(row, reason):
        sample_id = str(row["source_sample_id"])
        if sample_id in seen or len(selected) >= limit:
            return
        seen.add(sample_id)
        selected.append({"source_sample_id": sample_id, "reason": reason})

    for outcome in ("betti_fixed", "betti_regressed"):
        for row in rows:
            if row["outcome"] == outcome:
                add(row, outcome)

    ranked_suppression = sorted(
        rows,
        key=lambda row: (
            int(row["control_false_cycle_rank"])
            - int(row["betti_false_cycle_rank"]),
            int(row["control_predicted_beta1"])
            - int(row["betti_predicted_beta1"]),
        ),
        reverse=True,
    )
    for row in ranked_suppression:
        reduction = int(row["control_false_cycle_rank"]) - int(
            row["betti_false_cycle_rank"]
        )
        add(row, f"largest_false_cycle_reduction={reduction}")

    for row in rows:
        add(row, "additional_loop_patch")
    return selected


def _render(
    *,
    predictions,
    dataset_root,
    sample_id,
    split,
    output_dir,
    match_distance,
    all_scores,
    edge_threshold,
):
    from scripts.visualize_graph_prediction_3d import main as visualize_main

    arguments = [
        "--dataset-root",
        str(dataset_root),
        "--split",
        split,
        "--predictions",
        str(predictions),
        "--sample-id",
        sample_id,
        "--output-dir",
        str(output_dir),
        "--edge-threshold",
        str(edge_threshold),
        "--error-analysis",
        "--match-distance",
        str(match_distance),
    ]
    if all_scores:
        arguments.append("--use-all-edge-scores")
    result = visualize_main(arguments)
    if result not in (None, 0):
        raise RuntimeError(f"3D visualizer failed with status {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--betti-predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--match-distance", type=float, default=0.1)
    parser.add_argument("--soft-edge-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if not 0.0 <= args.soft_edge_threshold <= 1.0:
        parser.error("--soft-edge-threshold must lie in [0, 1]")

    rows = json.loads(args.comparison.read_text(encoding="utf-8"))
    if args.sample_ids:
        by_id = {str(row["source_sample_id"]): row for row in rows}
        missing = [sample_id for sample_id in args.sample_ids if sample_id not in by_id]
        if missing:
            raise KeyError(f"Samples are absent from comparison: {missing}")
        examples = [
            {"source_sample_id": sample_id, "reason": "explicit"}
            for sample_id in args.sample_ids
        ]
    else:
        examples = select_examples(rows, limit=args.max_examples)

    predictions = {
        "control": args.control_predictions,
        "betti": args.betti_predictions,
    }
    for example in examples:
        sample_id = example["source_sample_id"]
        for method, prediction_path in predictions.items():
            base = args.output_dir / sample_id / method
            _render(
                predictions=prediction_path,
                dataset_root=args.dataset_root,
                sample_id=sample_id,
                split=args.split,
                output_dir=base / "hard",
                match_distance=args.match_distance,
                all_scores=False,
                edge_threshold=0.0,
            )
            _render(
                predictions=prediction_path,
                dataset_root=args.dataset_root,
                sample_id=sample_id,
                split=args.split,
                output_dir=base / f"all_scores_above_{args.soft_edge_threshold:g}",
                match_distance=args.match_distance,
                all_scores=True,
                edge_threshold=args.soft_edge_threshold,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(examples)} paired loop-patch examples: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
