#!/usr/bin/env python3
"""Build a deterministic train-patch screening pool for topology overfitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loaders.discovery import discover_synthetic_mri
from data.loaders.io import read_vtp_graph
from training.evaluation.metrics import graph_betti_numbers


def _rank(sample_id: str, seed: int):
    payload = f"{int(seed)}\0topology-overfit-pool\0{sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).digest(), sample_id


def build_pool(records, *, total: int, loop_fraction: float, seed: int):
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0.0 <= loop_fraction <= 1.0:
        raise ValueError("loop_fraction must lie in [0,1]")
    requested_loops = round(total * loop_fraction)
    requested_nonloops = total - requested_loops
    loops = []
    nonloops = []
    inspected = 0
    ranked_records = sorted(records, key=lambda record: _rank(record.sample_id, seed))
    for record in ranked_records:
        if len(loops) >= requested_loops and len(nonloops) >= requested_nonloops:
            break
        nodes, edges = read_vtp_graph(record.graph)
        beta0, beta1 = graph_betti_numbers(len(nodes), edges)
        inspected += 1
        row = {
            "source_sample_id": record.sample_id,
            "target_nodes": int(len(nodes)),
            "target_edges": int(len(edges)),
            "target_beta0": int(beta0),
            "target_beta1": int(beta1),
        }
        if beta1 > 0 and len(loops) < requested_loops:
            loops.append(row)
        elif beta1 == 0 and len(nonloops) < requested_nonloops:
            nonloops.append(row)
    chosen_loops = loops
    chosen_nonloops = nonloops
    if len(chosen_loops) + len(chosen_nonloops) < total:
        raise ValueError(
            "Dataset cannot satisfy requested loop/non-loop pool composition: "
            f"loops={len(chosen_loops)}/{requested_loops}, "
            f"nonloops={len(chosen_nonloops)}/{requested_nonloops}"
        )
    selected = sorted(
        chosen_loops + chosen_nonloops,
        key=lambda row: row["source_sample_id"],
    )
    return selected, {
        "split": "train",
        "requested_total": int(total),
        "selected_total": len(selected),
        "loop_fraction": float(loop_fraction),
        "loop_patches": sum(row["target_beta1"] > 0 for row in selected),
        "nonloop_patches": sum(row["target_beta1"] == 0 for row in selected),
        "seed": int(seed),
        "inspected_graphs": int(inspected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=512)
    parser.add_argument("--loop-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=364505)
    args = parser.parse_args()
    records = discover_synthetic_mri(args.dataset_root, "train")
    selected, summary = build_pool(
        records,
        total=args.total,
        loop_fraction=args.loop_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_sample_ids.txt").write_text(
        "".join(row["source_sample_id"] + "\n" for row in selected),
        encoding="utf-8",
    )
    (args.output_dir / "pool.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
