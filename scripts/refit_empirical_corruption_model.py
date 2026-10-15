#!/usr/bin/env python3
"""Refit saved corruption samples without repeating raw-shard extraction."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

try:
    from scripts.build_empirical_corruption_model import COUNT_FIELDS, EVENT_TYPES, _fit_metric, _markdown
except ModuleNotFoundError:
    from build_empirical_corruption_model import COUNT_FIELDS, EVENT_TYPES, _fit_metric, _markdown


def refit(study_root, output):
    study_root, output = Path(study_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    settings = json.loads((study_root / "model_settings.json").read_text())
    threshold = settings["threshold"]
    with gzip.open(study_root / "patch_profiles.csv.gz", "rt") as handle:
        profiles = list(csv.DictReader(handle))
    with np.load(study_root / "empirical_samples.npz", allow_pickle=False) as arrays:
        keys = sorted(key for key in arrays.files if key.endswith("__fit"))
        fits = []
        for key in keys:
            epoch_label, population, metric, _ = key.split("__")
            holdout_key = key[:-3] + "holdout"
            if holdout_key not in arrays:
                continue
            fits.append({
                "epoch": int(epoch_label[1:]), "population": population, "metric": metric,
                "candidates": _fit_metric(arrays[key], arrays[holdout_key], population, metric, threshold),
            })
    for epoch in sorted({int(row["epoch"]) for row in profiles}):
        rows = [row for row in profiles if int(row["epoch"]) == epoch]
        for event, field in zip(EVENT_TYPES, COUNT_FIELDS):
            fits.append({
                "epoch": epoch, "population": event, "metric": "count",
                "candidates": _fit_metric(
                    [int(row[field]) for row in rows if row["study_split"] == "fit"],
                    [int(row[field]) for row in rows if row["study_split"] == "holdout"],
                    event, "count", threshold,
                ),
            })
    def read_csv(name):
        with (study_root / name).open() as handle:
            rows = list(csv.DictReader(handle))
        return [{key: (value if key in {"event_type", "pattern"} else float(value))
                 for key, value in row.items()} for row in rows]

    empirical = json.loads((study_root / "empirical_distributions.json").read_text())
    report = _markdown(profiles, read_csv("count_occurrence.csv"), read_csv("joint_patterns.csv"),
                       empirical, fits, max(int(row["epoch"]) for row in profiles))
    output.mkdir(parents=True, exist_ok=True)
    (output / "parametric_fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report)
    (output / "COMPLETED.json").write_text(json.dumps({
        "source_study": str(study_root.resolve()), "profiles": len(profiles),
        "fit_groups": len(fits), "fit_version": 2,
    }, indent=2) + "\n")
    print(f"Corrected fits saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    refit(args.study_root, args.output_dir)
