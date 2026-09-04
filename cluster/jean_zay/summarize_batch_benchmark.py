#!/usr/bin/env python3
"""Summarize the two H100 batch-size benchmark cases without extra packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys


def read_key_values(path: Path):
    return dict(
        line.rstrip("\n").split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def read_case(path: Path):
    summary = read_key_values(path / "summary.txt")
    epochs = [
        json.loads(line)
        for line in (path / "performance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Epoch two excludes most one-off CUDA initialization and filesystem cache
    # warm-up effects; fall back to the final available epoch if needed.
    representative = epochs[-1]
    gpu_utilization = []
    power_watts = []
    memory_mib = []
    with (path / "gpu-telemetry.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 9:
                continue
            try:
                gpu_utilization.append(float(row[2]))
                memory_mib.append(float(row[4]))
                power_watts.append(float(row[6]))
            except ValueError:
                continue
    return {
        "name": summary["case"],
        "global_batch": int(summary["global_batch_size"]),
        "wall_seconds": int(summary["wall_seconds"]),
        "gpu_hours": float(summary["allocated_gpu_hours"]),
        "train_seconds": float(representative["train_seconds"]),
        "samples_per_second": float(representative["samples_per_second"]),
        "peak_allocated_gib": float(representative["peak_allocated_gib"]),
        "mean_gpu_utilization": statistics.fmean(gpu_utilization),
        "mean_power_watts": statistics.fmean(power_watts),
        "max_memory_mib": max(memory_mib),
    }


def main():
    root = Path(sys.argv[1])
    cases = [read_case(root / name) for name in ("batch8_per_gpu", "batch32_per_gpu")]
    baseline = cases[0]
    print("Representative training throughput (final benchmark epoch)")
    print("case                 global  epoch_s  samples/s  peak_GiB  GPU_util%  power_W  max_mem_MiB  GPU-hours")
    for case in cases:
        print(
            "{name:<20} {global_batch:>6} {train_seconds:>8.1f} "
            "{samples_per_second:>10.2f} {peak_allocated_gib:>9.2f} "
            "{mean_gpu_utilization:>10.1f} {mean_power_watts:>8.1f} "
            "{max_memory_mib:>12.0f} {gpu_hours:>10.3f}".format(**case)
        )
    candidate = cases[1]
    print()
    print("batch32 throughput speedup: {:.3f}x".format(
        candidate["samples_per_second"] / baseline["samples_per_second"]
    ))
    print("batch32 epoch-time speedup: {:.3f}x".format(
        baseline["train_seconds"] / candidate["train_seconds"]
    ))


if __name__ == "__main__":
    main()
