#!/usr/bin/env python3
"""Compare full-MRI H100 layouts that preserve global batch 32."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys


CASES = ("1gpu_batch32", "2gpu_batch16")


def key_values(path: Path):
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def read_case(path: Path):
    summary = key_values(path / "summary.txt")
    if summary.get("exit_code") != "0":
        raise RuntimeError(f"Benchmark failed: {path}")
    epochs = [
        json.loads(line)
        for line in (path / "performance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    representative = epochs[-1]
    utilization, power, memory = [], [], []
    with (path / "gpu-telemetry.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 9:
                continue
            try:
                utilization.append(float(row[2]))
                memory.append(float(row[4]))
                power.append(float(row[6]))
            except ValueError:
                continue
    return {
        "case": summary["case"],
        "gpus": int(summary["gpus"]),
        "batch": int(summary["batch_size_per_gpu"]),
        "global_batch": int(summary["global_batch_size"]),
        "train_seconds": float(representative["train_seconds"]),
        "samples_per_second": float(representative["samples_per_second"]),
        "peak_gib": float(representative["peak_allocated_gib"]),
        "gpu_hours": float(summary["allocated_gpu_hours"]),
        "mean_utilization": statistics.fmean(utilization),
        "mean_power": statistics.fmean(power),
        "max_memory_mib": max(memory),
    }


def main():
    root = Path(sys.argv[1])
    cases = [read_case(root / name) for name in CASES]
    print("Full boundary-MRI benchmark (representative final epoch)")
    print("case             GPUs batch global epoch_s samples/s peak_GiB util% power_W max_mem_MiB GPU-hours")
    for case in cases:
        print(
            "{case:<17} {gpus:>4} {batch:>5} {global_batch:>6} "
            "{train_seconds:>7.1f} {samples_per_second:>9.2f} "
            "{peak_gib:>8.2f} {mean_utilization:>5.1f} {mean_power:>7.1f} "
            "{max_memory_mib:>11.0f} {gpu_hours:>9.3f}".format(**case)
        )
    one, two = cases
    print()
    print("2-GPU wall-time speedup: {:.3f}x".format(
        one["train_seconds"] / two["train_seconds"]
    ))
    print("2-GPU throughput speedup: {:.3f}x".format(
        two["samples_per_second"] / one["samples_per_second"]
    ))
    print("2-GPU training GPU-hour ratio: {:.3f}x".format(
        (two["train_seconds"] * 2) / one["train_seconds"]
    ))


if __name__ == "__main__":
    main()
