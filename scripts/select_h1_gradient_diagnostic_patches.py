#!/usr/bin/env python3
"""Select genuine-loop patches for the H1 birth-edge gradient diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rank(sample_id: str, seed: int):
    payload = f"{int(seed)}\0h1-gradient-diagnostic\0{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).digest(), sample_id


def select_rows(pool, selection, *, unselected_limit: int, seed: int):
    if unselected_limit < 0:
        raise ValueError("unselected_limit must be non-negative")
    selected_ids = {str(row["source_sample_id"]) for row in selection}
    mechanism = [
        {
            "source_sample_id": str(row["source_sample_id"]),
            "cohort": "selected_genuine_loop_edge_break",
            "target_beta1": int(row["target_beta1"]),
        }
        for row in selection
        if row.get("selection_category") == "genuine_loop_edge_break"
    ]
    unselected = [
        {
            "source_sample_id": str(row["source_sample_id"]),
            "cohort": "unselected_target_loop",
            "target_beta1": int(row["target_beta1"]),
        }
        for row in pool
        if int(row["target_beta1"]) > 0
        and str(row["source_sample_id"]) not in selected_ids
    ]
    unselected.sort(key=lambda row: _rank(row["source_sample_id"], seed))
    result = mechanism + unselected[:unselected_limit]
    if not result:
        raise ValueError("No genuine-loop patches were selected")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unselected-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=364505)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = select_rows(
        pool,
        selection,
        unselected_limit=args.unselected_limit,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_sample_ids.txt").write_text(
        "".join(row["source_sample_id"] + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "samples": len(rows),
        "selected_genuine_loop_edge_break": sum(
            row["cohort"] == "selected_genuine_loop_edge_break" for row in rows
        ),
        "unselected_target_loop": sum(
            row["cohort"] == "unselected_target_loop" for row in rows
        ),
        "seed": int(args.seed),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
