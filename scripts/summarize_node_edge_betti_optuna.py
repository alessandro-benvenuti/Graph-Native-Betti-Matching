#!/usr/bin/env python3
"""Create deterministic Pareto reports for a node/edge Betti study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path: sys.path.insert(0, str(REPOSITORY))

from scripts.optimize_node_edge_betti import (
    CampaignError, OBJECTIVES, PRIMARY_METRICS, create_storage,
)


def dominates(left, right, objectives=OBJECTIVES):
    no_worse, strictly_better = True, False
    for metric, direction in objectives:
        a, b = float(left[metric]), float(right[metric])
        if direction == "maximize":
            no_worse &= a >= b; strictly_better |= a > b
        else:
            no_worse &= a <= b; strictly_better |= a < b
    return bool(no_worse and strictly_better)


def pareto_front(rows):
    complete = [row for row in rows if row["state"] == "COMPLETE"
                and all(row.get(metric) is not None for metric in PRIMARY_METRICS)]
    return sorted([row for row in complete
                   if not any(dominates(other, row) for other in complete if other is not row)],
                  key=lambda row: row["number"])


def _best(front, metric, direction):
    if not front: return None
    sign = -1 if direction == "maximize" else 1
    return min(front, key=lambda row: (sign * float(row[metric]), row["number"]))


def balanced_representative(front):
    if not front: return None
    ranges = {}
    for metric, _ in OBJECTIVES:
        values = [float(row[metric]) for row in front]
        ranges[metric] = (min(values), max(values))
    ranked = []
    for row in front:
        normalized = {}
        for metric, direction in OBJECTIVES:
            low, high = ranges[metric]
            if high == low: score = 1.0
            elif direction == "maximize": score = (float(row[metric]) - low) / (high - low)
            else: score = (high - float(row[metric])) / (high - low)
            normalized[metric] = score
        distance = math.sqrt(sum((1.0 - value) ** 2 for value in normalized.values()))
        ranked.append((distance, row["number"], row, normalized))
    distance, _, row, normalized = min(ranked)
    return {"trial": row, "ideal_distance": distance, "normalized_objectives": normalized,
            "label": "conventional equal-weight balanced representative; not a unique optimum"}


def representatives(front):
    result = {}
    labels = (("best_node_mAP", "node_mAP", "maximize"),
              ("best_edge_mAP", "edge_mAP", "maximize"),
              ("best_beta0_error", "beta0_absolute_error", "minimize"),
              ("best_beta1_error", "beta1_absolute_error", "minimize"))
    for label, metric, direction in labels:
        row = _best(front, metric, direction)
        result[label] = None if row is None else {"trial": row["number"], "config_path": row.get("config_path")}
    balanced = balanced_representative(front)
    result["balanced"] = None if balanced is None else {
        "trial": balanced["trial"]["number"], "config_path": balanced["trial"].get("config_path"),
        "ideal_distance": balanced["ideal_distance"],
        "normalized_objectives": balanced["normalized_objectives"], "label": balanced["label"],
    }
    return result


def study_rows(study, control):
    rows = []
    control_metrics = control["metrics"]
    for trial in sorted(study.trials, key=lambda item: item.number):
        attrs, aggregation = trial.user_attrs, trial.user_attrs.get("aggregation", {})
        metrics = aggregation.get("metrics", {})
        values = list(trial.values) if trial.values is not None else [None] * len(OBJECTIVES)
        row = {"number": trial.number, "state": trial.state.name,
               **{metric: values[index] for index, (metric, _) in enumerate(OBJECTIVES)},
               **{name: value for name, value in metrics.items() if name not in PRIMARY_METRICS},
               "aggregation_method": aggregation.get("method"),
               "contributing_epochs": aggregation.get("contributing_epochs"),
               "observation_count": aggregation.get("observations"),
               "final_epoch": attrs.get("final_epoch"),
               "training_completed": attrs.get("training_completed", False),
               "config_path": attrs.get("config_path"), "run_dir": attrs.get("run_dir"),
               "slurm_job_id": attrs.get("slurm_job_id"),
               "duration_seconds": attrs.get("duration_seconds",
                   trial.duration.total_seconds() if trial.duration else None),
               "failure_reason": attrs.get("failure_reason"), **trial.params}
        for name, value in metrics.items():
            if name in control_metrics: row[f"delta_{name}"] = float(value) - float(control_metrics[name])
        rows.append(row)
    front_numbers = {row["number"] for row in pareto_front(rows)}
    for row in rows: row["pareto_front"] = row["number"] in front_numbers
    return rows


def _csv_value(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    if not keys: keys = ["number", "state"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
        for row in rows: writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _representative_markdown(values):
    lines = ["# Pareto representatives", "",
             "These are descriptive anchors, not a unique mathematically best configuration.", "",
             "| representative | trial | configuration |", "|---|---:|---|"]
    for label, item in values.items():
        lines.append(f"| {label} | {item['trial'] if item else '-'} | {item.get('config_path', '-') if item else '-'} |")
    return "\n".join(lines) + "\n"


def write_summary(output: Path, study_name: str, storage_kind="sqlite"):
    try: import optuna
    except ImportError as error: raise CampaignError("Optuna is required to summarize a study") from error
    control_path = output / "control-reference.json"
    if not control_path.is_file(): raise CampaignError("control-reference.json is missing")
    control = json.loads(control_path.read_text())
    study = optuna.load_study(study_name=study_name, storage=create_storage(optuna, output, storage_kind))
    rows = study_rows(study, control); front = pareto_front(rows); reps = representatives(front)
    for item in reps.values():
        if item and item.get("config_path") and Path(item["config_path"]).is_file():
            import yaml
            item["configuration"] = yaml.safe_load(Path(item["config_path"]).read_text())
    last_generation_start = max((row["number"] for row in rows), default=-1) - 11
    report = {"study_name": study_name, "objectives": list(OBJECTIVES),
              "aggregation": control["aggregation"]["method"], "control": control,
              "trial_count": len(rows), "complete_trial_count": sum(r["state"] == "COMPLETE" for r in rows),
              "pareto_trial_numbers": [r["number"] for r in front], "representatives": reps,
              "pareto_trials_in_last_12_allocations": [r["number"] for r in front if r["number"] >= last_generation_start],
              "trials": rows}
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "pareto-front.json").write_text(json.dumps(front, indent=2, sort_keys=True) + "\n")
    (output / "representative-configurations.json").write_text(json.dumps(reps, indent=2, sort_keys=True) + "\n")
    (output / "representative-configurations.md").write_text(_representative_markdown(reps))
    _write_csv(output / "trials.csv", rows); _write_csv(output / "pareto-front.csv", front)
    print(f"study={study_name} trials={len(rows)} complete={report['complete_trial_count']} pareto={len(front)}")
    print("trial state      pareto node_mAP edge_mAP beta0_err beta1_err tail_epochs")
    for row in rows:
        fmt = lambda value: "-" if value is None else f"{value:.6f}"
        print(f"{row['number']:>5} {row['state']:<10} {str(row['pareto_front']):<6} "
              f"{fmt(row['node_mAP']):>8} {fmt(row['edge_mAP']):>8} "
              f"{fmt(row['beta0_absolute_error']):>9} {fmt(row['beta1_absolute_error']):>9} "
              f"{row['contributing_epochs'] or '-'}")
    print("Pareto representatives (descriptive, no unique winner):")
    for label, item in reps.items(): print(f"  {label}: {item['trial'] if item else '-'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", choices=("sqlite", "journal"), default="sqlite")
    args = parser.parse_args()
    try: write_summary(args.output.expanduser().resolve(), args.study_name, args.storage)
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot summarize study: {error}") from error


if __name__ == "__main__": main()
