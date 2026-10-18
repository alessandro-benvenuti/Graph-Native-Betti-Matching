#!/usr/bin/env python3
"""Audit epoch-matched staged Betti models using the existing spatial cycle matcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze(path, payload):
    if path.exists() and read_json(path) != payload:
        raise ValueError(f"Inputs changed; use a fresh output directory: {path}")
    write_json(path, payload)


def checkpoint_epoch(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def prepare_inputs(campaign, epoch):
    from configs import load_config

    inputs = {}
    thresholds = None
    for label in ("control", "O07", "S07"):
        run = campaign / ("shared-prefix" if label == "control" else f"promotion-{epoch:04d}/{label}")
        checkpoint = run / (f"checkpoints/epoch_{epoch:04d}.pt" if label == "control" else "models/latest_checkpoint.pt")
        config_path = run / "resolved-config.yaml"
        if label != "control" and not (run / "training-complete").is_file():
            raise ValueError(f"Training is not complete: {run}")
        if checkpoint_epoch(checkpoint) != epoch:
            raise ValueError(f"Checkpoint is not at epoch {epoch}: {checkpoint}")
        config = load_config(config_path)
        current = {key: config["evaluation"][key] for key in ("node_threshold", "edge_threshold")}
        if thresholds is not None and thresholds != current:
            raise ValueError("Models have different inference thresholds")
        thresholds = current
        inputs[label] = {
            "config": str(config_path), "checkpoint": str(checkpoint),
            "dataset_root": str(Path(config["data"]["datasets"]["synthetic_mri"]["root"]).resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        }
    return inputs, thresholds


def recovery_counts(rows):
    """Report rank gains and losses separately; a net gain can hide regressions."""
    result = {}
    for metric, gain_name, loss_name in (
        ("shared_cycle_rank", "genuine_rank_gained", "genuine_rank_lost"),
        ("false_cycle_rank", "false_rank_added", "false_rank_removed"),
    ):
        differences = [int(row[f"betti_{metric}"]) - int(row[f"control_{metric}"]) for row in rows]
        result[gain_name] = sum(max(delta, 0) for delta in differences)
        result[loss_name] = sum(max(-delta, 0) for delta in differences)
    return result


def infer_model(directory, source, args, selection_args):
    """Only trust exports after evaluate.py exits successfully."""
    marker = directory / "inference-complete.json"
    if marker.is_file():
        if not (directory / "predictions.json").is_file():
            raise ValueError(f"Completed inference has no predictions: {directory}")
        return
    subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), "--config", source["config"],
        "--checkpoint", source["checkpoint"], "--output-dir", str(directory),
        "--dataset", "synthetic_mri", "--split", "val", "--device", "cuda",
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--bn-calibration-batches", "0", "--export-all-edge-scores", "--visualizations", "0",
        *selection_args,
    ], check=True, cwd=ROOT)
    write_json(marker, {"epoch": args.epoch})


def build_report(output, predictions, dataset, distance, scope):
    from scripts.compare_loop_patch_predictions import compare_predictions, summarize, write_report

    summary = {
        "scope": scope, "split": "val", "max_node_distance": distance,
        "interpretation": "Cycle-space overlap after one-to-one spatial node matching; relative to GT annotations. Rank gains are not unique named loop identities.",
        "candidates": {},
    }
    for label in ("O07", "S07"):
        rows = compare_predictions(
            predictions["control"], predictions[label], dataset,
            max_node_distance=distance, require_gt_loops=(scope == "gt-loops"),
        )
        pair_summary = summarize(rows, max_node_distance=distance)
        write_report(output / "comparison" / label, rows, pair_summary)
        cohorts = {"all_evaluated": rows}
        if scope == "all-validation":
            cohorts.update(
                target_loop=[r for r in rows if r["control_target_beta1"] > 0],
                target_nonloop=[r for r in rows if r["control_target_beta1"] == 0],
            )
        summary["candidates"][label] = {}
        for name, selected in cohorts.items():
            if not selected:
                continue
            cohort = summarize(selected, max_node_distance=distance)
            cohort["rank_changes"] = recovery_counts(selected)
            summary["candidates"][label][name] = cohort

    lines = ["# Epoch-matched loop audit", "", f"Scope: {scope}; validation only.", "",
             "Genuine cycles are measured by spatially shared cycle-space rank, not merely equal Betti counts.",
             "GT-loop-only results do not measure false-loop suppression on GT-nonloop patches.", ""]
    for label, cohorts in summary["candidates"].items():
        for name, cohort in cohorts.items():
            lines += [f"## {label}: {name} ({cohort['loop_patches']} patches)", "",
                      "| Metric | Control | Betti | Delta |", "|---|---:|---:|---:|"]
            for metric in ("shared_cycle_rank", "false_cycle_rank", "missed_cycle_rank", "cycle_precision", "cycle_recall", "mean_beta1_absolute_error", "basis_cycles_missing_node", "basis_cycles_broken_by_hard_edge"):
                left, right = (cohort["methods"][method][metric] for method in ("control", "betti"))
                lines.append(f"| {metric} | {left:.6f} | {right:.6f} | {right-left:+.6f} |")
            lines += ["", "Rank changes across patches: " + json.dumps(cohort["rank_changes"], sort_keys=True), ""]
    lines += ["Annotation errors and the spatial matching tolerance limit interpretation. A lower cycle-count error alone does not establish genuine-loop recovery.", ""]
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--scope", choices=("gt-loops", "all-validation"), default="gt-loops")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-node-distance", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_node_distance <= 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size, worker count, or spatial distance")
    campaign, output = args.campaign_root.resolve(), args.output_dir.resolve()
    inputs, thresholds = prepare_inputs(campaign, args.epoch)
    for label, source in inputs.items():
        if Path(source["dataset_root"]) != args.dataset_root.resolve():
            raise ValueError(f"Configured inference dataset differs from audit dataset: {label}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "epoch": args.epoch, "scope": args.scope, "dataset": str(args.dataset_root.resolve()),
        "inputs": inputs, "thresholds": thresholds, "bn_calibration_batches": 0,
        "max_node_distance": args.max_node_distance, "batch_size": args.batch_size,
        "workers": args.workers, "test_split_used": False,
    }
    freeze(output / "inputs.json", provenance)
    from data.loaders.discovery import discover_synthetic_mri
    from scripts.find_gt_loop_patches import find_loop_records, write_selection
    from training.evaluation.interactive_visualization import load_prediction_records

    dataset = discover_synthetic_mri(args.dataset_root, "val")
    selection_args = []
    if args.scope == "gt-loops":
        selected = find_loop_records(dataset)
        if not selected:
            raise ValueError("No GT-loop validation patches found")
        selection_dir = output / "selection"
        write_selection(selection_dir, selected, total_samples=len(dataset), split="val")
        selection_args = ["--sample-list", str(selection_dir / "source_sample_ids.txt")]
    predictions = {}
    for label, source in inputs.items():
        directory = output / label
        infer_model(directory, source, args, selection_args)
        predictions[label] = load_prediction_records(directory / "predictions.json")
    expected_ids = {str(r.sample_id) for r in dataset} if args.scope == "all-validation" else {r["source_sample_id"] for r in selected}
    for label, records in predictions.items():
        ids = [str(r["source_sample_id"]) for r in records]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"Prediction sample IDs differ from requested selection: {label}")
    build_report(output, predictions, dataset, args.max_node_distance, args.scope)
    write_json(output / "analysis-complete.json", {"epoch": args.epoch, "scope": args.scope})


if __name__ == "__main__":
    main()
