#!/usr/bin/env python3
"""Write constrained-selection reports for a node/edge Betti Optuna study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.optimize_node_edge_betti import CONSTRAINT_METRICS, METRICS, CampaignError


def select_best(rows):
    complete = [row for row in rows if row["state"] == "COMPLETE"]
    feasible = [row for row in complete if row["feasible"]]
    if feasible:
        return min(feasible, key=lambda row: (
            row["topology_score"], -row["edge_f1"], -row["node_f1"], row["number"]
        )), True
    if complete:
        return min(complete, key=lambda row: (
            row["total_violation"], row["topology_score"], row["number"]
        )), False
    return None, False


def study_rows(study):
    rows = []
    for trial in study.trials:
        attrs = trial.user_attrs
        violations = attrs.get("constraint_violations", {})
        row = {
            "number": trial.number,
            "state": trial.state.name,
            **trial.params,
            "best_objective": attrs.get("best_objective_value", trial.value),
            "best_objective_epoch": attrs.get("best_objective_epoch"),
            "topology_score": attrs.get("topology_score"),
            "feasible": bool(attrs.get("feasible", False)),
            "total_violation": sum(float(violations.get(name, 0)) for name in CONSTRAINT_METRICS),
            **{f"violation_{name}": violations.get(name) for name in CONSTRAINT_METRICS},
            **{name: attrs.get(name) for name in METRICS},
            "best_edge_mAP_epoch": attrs.get("best_edge_mAP_epoch"),
            "last_epoch": attrs.get("last_epoch"),
            "selected_epoch_feasible": attrs.get("selected_epoch_feasible"),
            "run_dir": attrs.get("run_dir"),
            "slurm_job_id": attrs.get("slurm_job_id"),
        }
        rows.append(row)
    return rows


def write_summary(output: Path, study_name: str):
    try:
        import optuna
    except ImportError as error:
        raise CampaignError("Optuna is required to summarize a study") from error
    storage = f"sqlite:///{(output / 'study.sqlite3').resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = study_rows(study)
    selected, feasible = select_best(rows)
    report = {
        "study_name": study_name,
        "trial_count": len(rows),
        "feasible_trial_count": sum(row["state"] == "COMPLETE" and row["feasible"] for row in rows),
        "selection_status": "best_feasible" if feasible else ("no_feasible_trial" if selected else "no_complete_trial"),
        "selected_trial": selected,
        "trials": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fieldnames = list(rows[0]) if rows else ["number", "state"]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best_path = output / "best-feasible.yaml"
    if feasible:
        config_path = Path(selected["run_dir"]) / "resolved-config.yaml"
        value = yaml.safe_load(config_path.read_text()) if config_path.is_file() else selected
        best_path.write_text(yaml.safe_dump(value, sort_keys=False))
    elif best_path.exists():
        best_path.unlink()
    print(f"study={study_name} trials={len(rows)} feasible={report['feasible_trial_count']}")
    print("trial state     feasible topology  edge_F1  node_F1 total_violation")
    for row in rows:
        def fmt(value): return "-" if value is None else f"{value:.6f}"
        print(f"{row['number']:>5} {row['state']:<9} {str(row['feasible']):<8} "
              f"{fmt(row['topology_score']):>8} {fmt(row['edge_f1']):>8} "
              f"{fmt(row['node_f1']):>8} {row['total_violation']:.6f}")
    if feasible:
        print(f"best feasible trial: {selected['number']}")
    elif selected:
        print(f"NO FEASIBLE TRIAL; least-violating trial: {selected['number']}")
    else:
        print("NO COMPLETE TRIAL")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    args = parser.parse_args()
    try:
        write_summary(args.output.expanduser().resolve(), args.study_name)
    except (CampaignError, OSError, ValueError) as error:
        raise SystemExit(f"Cannot summarize study: {error}") from error


if __name__ == "__main__":
    main()
