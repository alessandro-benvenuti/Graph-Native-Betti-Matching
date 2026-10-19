#!/usr/bin/env python3
"""Select reproducible focal-loss qualitative examples from three evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_KEYS = ("baseline", "node_focal", "node_edge_focal")


def _read_metrics(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        sample_id = str(row.get("source_sample_id") or "")
        if not sample_id:
            raise ValueError(f"Missing source_sample_id in {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate source_sample_id={sample_id!r} in {path}")
        result[sample_id] = row
    return result


def _number(row: dict, name: str) -> float:
    value = row.get(name)
    if value in (None, ""):
        raise ValueError(f"Missing metric {name!r}")
    return float(value)


def _zero(row: dict, *names: str) -> bool:
    return all(_number(row, name) == 0 for name in names)


def classify(baseline: dict, node: dict, combined: dict) -> dict:
    baseline_extra_nodes = (
        _number(baseline, "node_fp") >= 1
        and _number(baseline, "predicted_nodes") > _number(baseline, "target_nodes")
    )
    baseline_extra_loops = (
        _number(baseline, "predicted_beta1") > _number(baseline, "target_beta1")
    )
    node_exact = _zero(
        node,
        "node_fp",
        "node_fn",
        "edge_fp",
        "edge_fn",
        "beta0_absolute_error",
        "beta1_absolute_error",
    )
    combined_nodes_exact = _zero(combined, "node_fp", "node_fn")
    combined_missing_edge = (
        _number(combined, "edge_fp") == 0
        and _number(combined, "edge_fn") >= 1
        and _number(combined, "predicted_edges") < _number(combined, "target_edges")
    )
    strict = (
        baseline_extra_nodes
        and baseline_extra_loops
        and node_exact
        and combined_nodes_exact
        and combined_missing_edge
    )
    return {
        "strict": strict,
        "baseline_extra_nodes": baseline_extra_nodes,
        "baseline_extra_loops": baseline_extra_loops,
        "node_focal_exact": node_exact,
        "node_edge_nodes_exact": combined_nodes_exact,
        "node_edge_missing_edge": combined_missing_edge,
    }


def _rank(entry: dict) -> tuple:
    baseline = entry["metrics"]["baseline"]
    combined = entry["metrics"]["node_edge_focal"]
    return (
        -int(entry["criteria"]["strict"]),
        -(
            _number(baseline, "predicted_beta1")
            - _number(baseline, "target_beta1")
        ),
        -_number(baseline, "node_fp"),
        abs(_number(combined, "edge_fn") - 1),
        entry["source_sample_id"],
    )


def select_examples(
    baseline_rows: dict[str, dict],
    node_rows: dict[str, dict],
    combined_rows: dict[str, dict],
    *,
    limit: int,
) -> tuple[list[dict], dict]:
    id_sets = [set(rows) for rows in (baseline_rows, node_rows, combined_rows)]
    if not id_sets[0] == id_sets[1] == id_sets[2]:
        raise ValueError("The three evaluations do not contain identical sample IDs")

    entries = []
    for sample_id in sorted(id_sets[0]):
        metrics = {
            "baseline": baseline_rows[sample_id],
            "node_focal": node_rows[sample_id],
            "node_edge_focal": combined_rows[sample_id],
        }
        criteria = classify(*metrics.values())
        entries.append({
            "source_sample_id": sample_id,
            "criteria": criteria,
            "metrics": metrics,
        })

    strict = [entry for entry in entries if entry["criteria"]["strict"]]
    eligible = strict
    mode = "strict"
    if not eligible:
        # Preserve the two characteristic failure modes and rank the node-focal
        # prediction by its total official FP/FN burden.  This fallback is
        # explicitly labelled and must not be described as an exact prediction.
        eligible = [
            entry
            for entry in entries
            if entry["criteria"]["baseline_extra_nodes"]
            and entry["criteria"]["baseline_extra_loops"]
            and entry["criteria"]["node_edge_nodes_exact"]
            and entry["criteria"]["node_edge_missing_edge"]
        ]
        mode = "relaxed_node_focal"
        eligible.sort(key=lambda entry: (
            sum(
                _number(entry["metrics"]["node_focal"], name)
                for name in ("node_fp", "node_fn", "edge_fp", "edge_fn")
            ),
            _rank(entry),
        ))
    else:
        eligible.sort(key=_rank)

    selected = eligible[:limit]
    summary = {
        "evaluated_patches": len(entries),
        "strict_candidates": len(strict),
        "eligible_candidates": len(eligible),
        "selection_mode": mode,
        "selected_patches": len(selected),
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--node-focal", type=Path, required=True)
    parser.add_argument("--node-edge-focal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    selected, summary = select_examples(
        _read_metrics(args.baseline),
        _read_metrics(args.node_focal),
        _read_metrics(args.node_edge_focal),
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps({"summary": summary, "examples": selected}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "source_sample_ids.txt").write_text(
        "".join(entry["source_sample_id"] + "\n" for entry in selected),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not selected:
        raise RuntimeError("No patch satisfies even the labelled relaxed criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
