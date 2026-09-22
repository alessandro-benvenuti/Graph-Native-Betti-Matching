#!/usr/bin/env python3
"""Render before/control/Betti 3D views for topology-overfit training patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def choose_examples(selection, *, per_category):
    chosen = []
    counts = {}
    for row in selection:
        category = str(row["selection_category"])
        if counts.get(category, 0) >= per_category:
            continue
        counts[category] = counts.get(category, 0) + 1
        chosen.append(
            {
                "source_sample_id": str(row["source_sample_id"]),
                "selection_category": category,
            }
        )
    return chosen


def _render(
    *, prediction_path, dataset_root, sample_id, output_dir, all_scores,
    edge_threshold, match_distance,
):
    from scripts.visualize_graph_prediction_3d import main as visualize_main

    arguments = [
        "--dataset-root", str(dataset_root),
        "--split", "train",
        "--predictions", str(prediction_path),
        "--sample-id", sample_id,
        "--output-dir", str(output_dir),
        "--edge-threshold", str(edge_threshold),
        "--error-analysis",
        "--match-distance", str(match_distance),
    ]
    if all_scores:
        arguments.append("--use-all-edge-scores")
    result = visualize_main(arguments)
    if result not in (None, 0):
        raise RuntimeError(f"3D visualizer failed with status {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--betti", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--soft-edge-threshold", type=float, default=0.25)
    parser.add_argument("--match-distance", type=float, default=0.1)
    args = parser.parse_args()
    if args.per_category <= 0:
        parser.error("--per-category must be positive")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    examples = choose_examples(selection, per_category=args.per_category)
    methods = {
        "before": args.before,
        "control": args.control,
        "betti": args.betti,
    }
    for example in examples:
        sample_id = example["source_sample_id"]
        category = example["selection_category"]
        for method, prediction_path in methods.items():
            base = args.output_dir / category / sample_id / method
            _render(
                prediction_path=prediction_path,
                dataset_root=args.dataset_root,
                sample_id=sample_id,
                output_dir=base / "hard",
                all_scores=False,
                edge_threshold=0.0,
                match_distance=args.match_distance,
            )
            _render(
                prediction_path=prediction_path,
                dataset_root=args.dataset_root,
                sample_id=sample_id,
                output_dir=base / f"all_scores_above_{args.soft_edge_threshold:g}",
                all_scores=True,
                edge_threshold=args.soft_edge_threshold,
                match_distance=args.match_distance,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rendered-selection.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(examples)} mechanistic examples: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
