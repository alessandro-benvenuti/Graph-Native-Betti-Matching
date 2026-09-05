#!/usr/bin/env python3
"""Summarize full-dataset global-batch-32 A100 layouts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import statistics
import sys


CASES = ("1gpu_batch32", "2gpu_batch16", "4gpu_batch8")
CASE_LAYOUTS = {
    "1gpu_batch32": (1, 32),
    "2gpu_batch16": (2, 16),
    "4gpu_batch8": (4, 8),
}
PERFORMANCE_PATTERN = re.compile(
    r"^performance epoch=(?P<epoch>\d+) "
    r"train_seconds=(?P<train_seconds>[0-9.]+) "
    r"samples_per_second=(?P<samples_per_second>[0-9.]+).*"
    r"peak_allocated_gib=(?P<peak_allocated_gib>[0-9.]+)"
)


def key_values(path):
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def read_case(path):
    name = path.name
    gpus, batch = CASE_LAYOUTS[name]
    summary_path = path / "summary.txt"
    summary = key_values(summary_path) if summary_path.is_file() else {}
    performance_path = path / "performance.jsonl"
    if performance_path.is_file():
        epochs = [
            json.loads(line)
            for line in performance_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        epochs = []
        train_log = path / "train.log"
        if train_log.is_file():
            for line in train_log.read_text(encoding="utf-8").splitlines():
                match = PERFORMANCE_PATTERN.match(line)
                if match:
                    epochs.append({
                        "epoch": int(match.group("epoch")),
                        "train_seconds": float(match.group("train_seconds")),
                        "samples_per_second": float(match.group("samples_per_second")),
                        "peak_allocated_gib": float(match.group("peak_allocated_gib")),
                    })
    if not epochs:
        raise RuntimeError(f"No completed training epoch: {path}")
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
        "case": name,
        "gpus": int(summary.get("gpus", gpus)),
        "batch": int(summary.get("batch_size_per_gpu", batch)),
        "epoch_seconds": float(epoch["train_seconds"]),
        "throughput": float(epoch["samples_per_second"]),
        "peak_gib": float(epoch["peak_allocated_gib"]),
        "physical_gpu_hours": (
            float(summary["allocated_gpu_hours"])
            if "allocated_gpu_hours" in summary
            else None
        ),
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
            "{mean_power:>7.1f} {max_memory_mib:>11.0f} {gpu_hours:>13}".format(
                **case,
                gpu_hours=(
                    "{:.3f}".format(case["physical_gpu_hours"])
                    if case["physical_gpu_hours"] is not None
                    else "incomplete"
                ),
            )
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
