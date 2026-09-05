#!/usr/bin/env python3
"""Summarize full-dataset global-batch-32 A100 layouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys


CASES = ("1gpu_batch32", "2gpu_batch16", "4gpu_batch8")


def key_values(path):
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def read_case(path):
    summary = key_values(path / "summary.txt")
    epochs = [
        json.loads(line)
        for line in (path / "performance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if summary.get("exit_code") != "0" or not epochs:
        raise RuntimeError(f"Incomplete benchmark: {path}")
    epoch = epochs[-1]
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
        "epoch_seconds": float(epoch["train_seconds"]),
        "throughput": float(epoch["samples_per_second"]),
        "peak_gib": float(epoch["peak_allocated_gib"]),
        "physical_gpu_hours": float(summary["allocated_gpu_hours"]),
        "mean_utilization": statistics.fmean(utilization),
        "mean_power": statistics.fmean(power),
        "max_memory_mib": max(memory),
    }


def main():
    root = Path(sys.argv[1])
    cases = [read_case(root / name) for name in CASES]
    baseline = cases[0]
    print("Full boundary-MRI A100 benchmark; all layouts use global batch 32")
    print("case             GPUs batch epoch_s samples/s peak_GiB util% power_W max_mem_MiB physical_GPUh")
    for case in cases:
        print(
            "{case:<17} {gpus:>4} {batch:>5} {epoch_seconds:>7.1f} "
            "{throughput:>9.2f} {peak_gib:>8.2f} {mean_utilization:>5.1f} "
            "{mean_power:>7.1f} {max_memory_mib:>11.0f} "
            "{physical_gpu_hours:>13.3f}".format(**case)
        )
    print()
    for case in cases[1:]:
        print(
            "{}: wall speedup {:.3f}x; training GPU-hour ratio {:.3f}x versus 1x32".format(
                case["case"],
                baseline["epoch_seconds"] / case["epoch_seconds"],
                case["gpus"] * case["epoch_seconds"] / baseline["epoch_seconds"],
            )
        )


if __name__ == "__main__":
    main()
