#!/usr/bin/env python3
"""Compare topology-overfit arms on the complete fixed screening pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.summarize_topology_overfit import (  # noqa: E402
    _index,
    build_rows,
    group_summary,
)


def build_pool_rows(pool, selection, predictions):
    selected_ids = {str(row["source_sample_id"]) for row in selection}
    cohort = []
    for item in pool:
        target_beta1 = int(item["target_beta1"])
        cohort.append(
            {
                "source_sample_id": str(item["source_sample_id"]),
                "selection_category": (
                    "selected_for_optimization"
                    if str(item["source_sample_id"]) in selected_ids
                    else "unselected_target_loop"
                    if target_beta1 > 0
                    else "unselected_target_nonloop"
                ),
            }
        )
    rows = build_rows(cohort, predictions)
    for row, item in zip(rows, pool):
        row["selected_for_optimization"] = (
            str(item["source_sample_id"]) in selected_ids
        )
        row["target_beta0"] = int(item["target_beta0"])
        row["target_beta1"] = int(item["target_beta1"])
    return rows


def summarize_pool_rows(rows):
    selected = [row for row in rows if row["selected_for_optimization"]]
    unselected = [row for row in rows if not row["selected_for_optimization"]]
    unselected_loops = [row for row in unselected if row["target_beta1"] > 0]
    unselected_nonloops = [row for row in unselected if row["target_beta1"] == 0]
    return {
        "interpretation": (
            "paired fixed training-pool evaluation; the unselected cohort was not "
            "used by the 25-patch optimization, but this is not an independent "
            "validation or test split"
        ),
        "cohorts": {
            "all_pool": group_summary(rows),
            "selected_for_optimization": group_summary(selected),
            "unselected_pool": group_summary(unselected),
            "unselected_target_loop": group_summary(unselected_loops),
            "unselected_target_nonloop": group_summary(unselected_nonloops),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--betti", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    predictions = {
        "before": _index(args.before),
        "control": _index(args.control),
        "betti": _index(args.betti),
    }
    pool_ids = {str(row["source_sample_id"]) for row in pool}
    for method, records in predictions.items():
        missing = pool_ids - set(records)
        if missing:
            raise KeyError(f"{method} predictions miss samples: {sorted(missing)[:5]}")

    rows = build_pool_rows(pool, selection, predictions)
    summary = summarize_pool_rows(rows)
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
