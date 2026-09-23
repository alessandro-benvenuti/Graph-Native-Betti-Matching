#!/usr/bin/env python3
"""Summarize genuine-loop birth-edge gradients for two frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def _index(records):
    return {str(record["source_sample_id"]): record for record in records}


def aggregate(records, sample_ids):
    entries = [records[sample_id]["modes"]["node_aware"] for sample_id in sample_ids]
    valid = [entry for entry in entries if "skipped" not in entry]
    matches = [match for entry in valid for match in entry["h1_matches"]]
    probabilities = [
        float(match["selected_birth_effective_confidence"]) for match in matches
    ]
    gradients = [abs(float(match["selected_birth_dloss_dp"])) for match in matches]
    return {
        "patches": len(valid),
        "patches_with_missing_gt_node": sum(
            entry["missing_target_node_count"] > 0 for entry in valid
        ),
        "missing_gt_nodes": sum(
            entry["missing_target_node_count"] for entry in valid
        ),
        "original_target_cycle_rank": sum(
            entry["original_target_cycle_rank"] for entry in valid
        ),
        "original_cycles_blocked_by_missing_node": sum(
            entry["original_target_cycles_blocked_by_missing_node"] for entry in valid
        ),
        "local_target_cycle_rank": sum(
            entry["h1"]["target_rank"] for entry in valid
        ),
        "matched_cycle_rank": sum(
            entry["h1"]["matched_rank"] for entry in valid
        ),
        "missed_target_cycle_rank": sum(
            entry["h1"]["missed_target_rank"] for entry in valid
        ),
        "false_prediction_cycle_rank": sum(
            entry["h1"]["false_prediction_rank"] for entry in valid
        ),
        "matched_birth_edges": len(matches),
        "matched_birth_gradient_increase": sum(
            match["selected_birth_gradient_descent"] == "increase"
            for match in matches
        ),
        "matched_birth_gradient_none": sum(
            match["selected_birth_gradient_descent"] == "none" for match in matches
        ),
        "matched_birth_gradient_decrease": sum(
            match["selected_birth_gradient_descent"] == "decrease"
            for match in matches
        ),
        "selected_birth_is_true_edge": sum(
            match["selected_birth_is_local_true_edge"] for match in matches
        ),
        "selected_birth_is_in_shared_generator": sum(
            match["selected_birth_is_in_shared_generator"] for match in matches
        ),
        "selected_birth_is_shared_bottleneck": sum(
            match["selected_birth_is_shared_generator_bottleneck"]
            for match in matches
        ),
        "mean_selected_birth_effective_confidence": (
            statistics.fmean(probabilities) if probabilities else None
        ),
        "median_selected_birth_effective_confidence": (
            statistics.median(probabilities) if probabilities else None
        ),
        "mean_absolute_selected_birth_gradient": (
            statistics.fmean(gradients) if gradients else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--betti", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    methods = {
        "control": _index(json.loads(args.control.read_text(encoding="utf-8"))),
        "betti": _index(json.loads(args.betti.read_text(encoding="utf-8"))),
    }
    cohorts = {
        "all": [str(row["source_sample_id"]) for row in manifest],
    }
    for name in sorted({row["cohort"] for row in manifest}):
        cohorts[name] = [
            str(row["source_sample_id"])
            for row in manifest
            if row["cohort"] == name
        ]
    requested = set(cohorts["all"])
    for method, records in methods.items():
        missing = requested - set(records)
        if missing:
            raise KeyError(f"{method} diagnostic misses: {sorted(missing)[:5]}")

    summary = {
        "interpretation": (
            "frozen-checkpoint local H1 gradient diagnostic; cycle representatives "
            "are deterministic basis diagnostics, while reported ranks are invariant"
        ),
        "cohorts": {
            cohort: {
                method: aggregate(records, sample_ids)
                for method, records in methods.items()
            }
            for cohort, sample_ids in cohorts.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "per-match.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "method",
            "cohort",
            "source_sample_id",
            "match_index",
            "selected_birth_query_edge",
            "selected_birth_raw_relation_probability",
            "selected_birth_effective_confidence",
            "selected_birth_dloss_dp",
            "selected_birth_gradient_descent",
            "selected_birth_is_local_true_edge",
            "selected_birth_is_in_shared_generator",
            "selected_birth_is_shared_generator_bottleneck",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        cohort_by_id = {
            str(row["source_sample_id"]): row["cohort"] for row in manifest
        }
        for method, records in methods.items():
            for sample_id in cohorts["all"]:
                entry = records[sample_id]["modes"]["node_aware"]
                for match in entry.get("h1_matches", []):
                    writer.writerow(
                        {
                            field: (
                                method
                                if field == "method"
                                else cohort_by_id[sample_id]
                                if field == "cohort"
                                else sample_id
                                if field == "source_sample_id"
                                else match[field]
                            )
                            for field in fields
                        }
                    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
