#!/usr/bin/env python3
"""Tiny train.py stand-in used only by CPU infrastructure tests."""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--initial-weights")
parser.add_argument("--resume")
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
run = args.output_dir / args.run_name
new_run = not run.exists()
run.mkdir(parents=True, exist_ok=bool(args.resume))
if not args.resume or new_run:
    (run / "resolved-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
manifest = {
    "schema_version": 1,
    "experiment_seed": config["experiment"]["seed"],
    "train": ["sample-a", "sample-b"],
    "validation": ["sample-v"],
}
manifest_path = run / "dataset-manifest.json"
if not args.resume or new_run:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
history_path = run / "validation-metrics.jsonl"
start_epoch = 1
resume_payload = None
if args.resume:
    try:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = int(resume_payload["epoch"]) + 1
    except Exception:
        raw = Path(args.resume).read_bytes().decode(errors="ignore")
        if raw.startswith("epoch="):
            start_epoch = int(raw.split("=", 1)[1]) + 1
if args.resume and history_path.is_file():
    existing = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if existing:
        start_epoch = max(start_epoch, max(record["epoch"] for record in existing) + 1)
(run / "models").mkdir(exist_ok=True)
if args.resume:
    (run / "resume-provenance.json").write_text(json.dumps({
        "checkpoint": str(Path(args.resume).resolve()),
        "checkpoint_epoch": start_epoch - 1,
        "optimizer_restored": bool(resume_payload and "optimizer" in resume_payload),
        "scheduler_restored": bool(resume_payload and "scheduler" in resume_payload),
    }, indent=2, sort_keys=True) + "\n")

def checkpoint_payload(epoch):
    return {
        "net": {"fake": torch.tensor([float(epoch)])},
        "optimizer": {"fake_step": epoch},
        "scheduler": {"last_epoch": epoch},
        "scaler": None,
        "epoch": epoch,
        "iteration": epoch,
        "global_step": epoch,
        "runtime_states": [{"fake_rng": torch.tensor([epoch])}],
        "trainer_state": {"world_size": 1},
        "training_config": config,
    }

configured_stop = config["training"].get("stop_after_epoch")
final_epoch = int(config["training"]["epochs"])
if configured_stop is not None:
    final_epoch = min(final_epoch, int(configured_stop))
with history_path.open("a" if args.resume else "w") as handle:
    for epoch in range(start_epoch, final_epoch + 1):
        staged = config.get("staged_metadata")
        topology_bonus = 0.0
        beta1_bonus = 0.0
        if staged:
            topology_bonus = float(config["topology"]["betti_h0"]["weight"]) * 0.2
            beta1_bonus = float(config["topology"]["betti_h1"]["weight"]) * 2.0
        record = {
            "epoch": epoch, "iteration": epoch,
            "node_mAP": .80 - topology_bonus, "edge_mAP": .70 - topology_bonus,
            "node_mAR": .79, "edge_mAR": .69,
            "node_precision": .76, "node_recall": .74, "node_f1": .75,
            "edge_precision": .66, "edge_recall": .64, "edge_f1": .65,
            "beta0_absolute_error": 2.0 - topology_bonus,
            "beta1_absolute_error": 4.0 - beta1_bonus,
            "smd": 1.25, "target_beta0": 3.0, "predicted_beta0": 4.0,
            "target_beta1": 2.0, "predicted_beta1": 5.0,
            "target_nodes": 20.0, "predicted_nodes": 19.0,
            "target_edges": 22.0, "predicted_edges": 20.0,
        }
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        payload = checkpoint_payload(epoch)
        torch.save(payload, run / "models/latest_checkpoint.pt")
        milestones = config["training"]["checkpoint"].get("milestone_epochs", [])
        if epoch in milestones:
            directory = run / "checkpoints"
            directory.mkdir(exist_ok=True)
            torch.save(payload, directory / f"epoch_{epoch:04d}.pt")
        time.sleep(float(os.environ.get("FAKE_TRAIN_SLEEP_PER_EPOCH", "0")))
if os.environ.get("FAKE_TRAIN_FAIL") == "1" and "trial_" in args.run_name:
    raise SystemExit(3)
