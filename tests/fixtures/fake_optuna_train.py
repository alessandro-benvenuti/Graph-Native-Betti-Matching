#!/usr/bin/env python3
"""Tiny train.py stand-in used only by CPU infrastructure tests."""

import argparse
import json
import os
from pathlib import Path

import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--initial-weights", required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
run = args.output_dir / args.run_name
run.mkdir(parents=True, exist_ok=False)
(run / "resolved-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
manifest = {
    "schema_version": 1,
    "experiment_seed": config["experiment"]["seed"],
    "train": ["sample-a", "sample-b"],
    "validation": ["sample-v"],
}
(run / "dataset-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
with (run / "validation-metrics.jsonl").open("w") as handle:
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        record = {
            "epoch": epoch, "iteration": epoch,
            "node_mAP": .80, "edge_mAP": .70,
            "node_mAR": .79, "edge_mAR": .69,
            "node_precision": .76, "node_recall": .74, "node_f1": .75,
            "edge_precision": .66, "edge_recall": .64, "edge_f1": .65,
            "beta0_absolute_error": 2.0, "beta1_absolute_error": 4.0,
            "smd": 1.25, "target_beta0": 3.0, "predicted_beta0": 4.0,
            "target_beta1": 2.0, "predicted_beta1": 5.0,
            "target_nodes": 20.0, "predicted_nodes": 19.0,
            "target_edges": 22.0, "predicted_edges": 20.0,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
if os.environ.get("FAKE_TRAIN_FAIL") == "1" and "trial_" in args.run_name:
    raise SystemExit(3)
