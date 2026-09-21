#!/usr/bin/env python3
"""List dataset patches whose ground-truth graph has non-zero cycle rank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loaders.discovery import discover_synthetic_mri
from data.loaders.io import read_vtp_graph
from training.evaluation.metrics import graph_betti_numbers


def find_loop_records(records, *, graph_reader=read_vtp_graph):
    result = []
    for record in records:
        nodes, edges = graph_reader(record.graph)
        beta0, beta1 = graph_betti_numbers(len(nodes), edges)
        if beta1 > 0:
            result.append(
                {
                    "source_sample_id": str(record.sample_id),
                    "target_nodes": int(len(nodes)),
                    "target_edges": int(len(edges)),
                    "target_beta0": int(beta0),
                    "target_beta1": int(beta1),
                }
            )
    return result


def write_selection(output_dir: Path, records, *, total_samples: int, split: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    list_path = output_dir / "source_sample_ids.txt"
    csv_path = output_dir / "gt_loop_patches.csv"
    summary_path = output_dir / "summary.json"
    list_path.write_text(
        "".join(record["source_sample_id"] + "\n" for record in records),
        encoding="utf-8",
    )
    fields = (
        "source_sample_id",
        "target_nodes",
        "target_edges",
        "target_beta0",
        "target_beta1",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "split": split,
        "total_samples": int(total_samples),
        "loop_patches": len(records),
        "loop_patch_fraction": len(records) / float(total_samples),
        "total_target_cycle_rank": sum(record["target_beta1"] for record in records),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    records = discover_synthetic_mri(args.dataset_root, args.split)
    loop_records = find_loop_records(records)
    if not loop_records:
        raise SystemExit(f"No GT loop patches found in split {args.split!r}")
    summary = write_selection(
        args.output_dir,
        loop_records,
        total_samples=len(records),
        split=args.split,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Sample list: {args.output_dir / 'source_sample_ids.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
