#!/usr/bin/env python3
"""Materialize full-data paired control/Betti configs after validation selection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path: sys.path.insert(0, str(REPOSITORY))

from configs import validate_config


def prepare(selected: dict, output: Path, seeds: list[int]):
    if "optuna" in selected:
        selected = copy.deepcopy(selected); selected.pop("optuna")
    validate_config(selected)
    generated = []
    for seed in seeds:
        base = copy.deepcopy(selected)
        base["experiment"]["seed"] = seed
        dataset = base["data"]["datasets"]["synthetic_mri"]
        dataset["train_samples"] = None; dataset["validation_samples"] = None
        base["training"].update(epochs=500, stop_after_epoch=None)
        base["training"]["early_stopping"]["enabled"] = False
        base["training"]["checkpoint"].update(
            policy="interval_and_best", interval_epochs=500, latest_interval_epochs=1
        )
        base["evaluation"]["interval_epochs"] = 5
        base["tracking"]["mode"] = "offline"
        base["tracking"]["group"] = "node-edge-betti-final-paired-full-data"
        for arm in ("control", "betti"):
            config = copy.deepcopy(base)
            name = f"node_edge_betti_final_full_{arm}_seed{seed}"
            config["experiment"]["name"] = name
            config["tracking"]["tags"] = ["gnbm", "full-data", "paired", arm, f"seed-{seed}"]
            if arm == "control":
                for loss in ("betti_h0", "betti_h1"):
                    config["topology"][loss].update(enabled=False, log_only=True, weight=0.0)
            validate_config(config)
            path = output / f"{arm}_seed{seed}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            generated.append({"arm": arm, "seed": seed, "run_name": name,
                              "config": str(path.resolve()),
                              "fixed_checkpoint": f"models/checkpoint_epoch=500.pt"})
    plan = {"schema_version": 1, "selection_source": "validation only",
            "test_split_used": False, "checkpoint_rule": "fixed epoch 500 for both arms",
            "epochs": 500, "full_training_split": True, "full_validation_split": True,
            "runs": generated}
    (output / "experiment-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[364505, 364506, 364507])
    args = parser.parse_args()
    plan = prepare(yaml.safe_load(args.selected_config.read_text()), args.output, args.seeds)
    print(f"Prepared {len(plan['runs'])} configs in {args.output}")
    print("No training was submitted. Inspect experiment-plan.json before launch.")


if __name__ == "__main__": main()
