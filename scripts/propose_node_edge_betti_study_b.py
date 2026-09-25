#!/usr/bin/env python3
"""Generate an inspect-before-launch Study B overlay from Study A Pareto candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


BETTI_PARAMETERS = (
    "topology.betti_h0.weight", "topology.betti_h1.weight",
    "topology.betti_h1.false_positive_weight", "betti_warmup_epochs",
    "betti_ramp_epochs",
)


def proposed_space(summary, selected_numbers=None):
    trials = {row["number"]: row for row in summary["trials"]}
    if selected_numbers is None:
        selected_numbers = sorted({item["trial"] for item in summary["representatives"].values() if item})
    selected = [trials[number] for number in selected_numbers]
    if not selected:
        raise ValueError("no Study A Pareto candidates selected")
    result = {name: sorted({row[name] for row in selected}) for name in BETTI_PARAMETERS}
    result["edge_loss"] = ["cross_entropy", "focal_gamma_0.5", "focal_gamma_1.0", "focal_gamma_2.0"]
    result["topology.complex.alpha"] = [0.0, 0.25, 0.5, 0.75, 1.0]
    return result, selected_numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-a-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        default=Path("configs/experiments/node_edge_betti_optuna/study_b_proposed.yaml"))
    parser.add_argument("--trials", type=int, nargs="*")
    args = parser.parse_args()
    summary = json.loads(args.study_a_summary.read_text())
    space, selected = proposed_space(summary, args.trials)
    proposal = {
        "defaults": ["study_b_template.yaml"],
        "optuna": {"search_space": space},
        "study_b_proposal": {
            "source_summary": str(args.study_a_summary.resolve()),
            "source_trial_numbers": selected,
            "status": "REQUIRES_USER_INSPECTION_BEFORE_LAUNCH",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(proposal, sort_keys=False))
    print(f"Wrote {args.output}; inspect it before launching Study B")


if __name__ == "__main__": main()
