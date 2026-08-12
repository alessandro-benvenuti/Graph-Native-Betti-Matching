#!/usr/bin/env python3
"""Evaluate a RelationFormer checkpoint and export graphs, metrics, and plots."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from configs import load_config, validate_config
from data.loaders import build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from training.evaluation import calibrate_batch_norm, evaluate_model


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--node-threshold", type=float)
    parser.add_argument("--edge-threshold", type=float)
    parser.add_argument("--bn-calibration-batches", type=int)
    parser.add_argument("--visualizations", type=int, default=0)
    parser.add_argument("--no-export-predictions", action="store_true")
    return parser


def _dataset_name(config, requested):
    if requested is not None:
        return requested
    target = [
        name
        for name, settings in config["data"]["datasets"].items()
        if settings["role"] == "target"
    ]
    if len(target) != 1:
        raise ValueError("--dataset is required unless exactly one target is configured")
    return target[0]


def main():
    args = _parser().parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.device:
        config["runtime"]["device"] = args.device
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.workers is not None:
        config["runtime"]["workers"] = args.workers
    if args.node_threshold is not None:
        config["evaluation"]["node_threshold"] = args.node_threshold
    if args.edge_threshold is not None:
        config["evaluation"]["edge_threshold"] = args.edge_threshold
    if args.bn_calibration_batches is not None:
        config["evaluation"]["bn_calibration_batches"] = args.bn_calibration_batches
    validate_config(config)
    dataset_name = _dataset_name(config, args.dataset)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Evaluation requests CUDA but CUDA is unavailable")
    loader = build_evaluation_loader(
        config,
        dataset_name=dataset_name,
        split=args.split,
        max_samples=args.max_samples,
    )
    model = build_model(config).to(device)
    checkpoint_report = load_legacy_model_checkpoint(
        model, args.checkpoint, map_location="cpu"
    )

    calibration_batches = int(config["evaluation"]["bn_calibration_batches"])
    if calibration_batches > 0:
        calibration_loader = build_evaluation_loader(
            config,
            dataset_name=dataset_name,
            split="val",
            max_samples=None,
        )
        consumed = calibrate_batch_norm(
            model,
            calibration_loader,
            config,
            device,
            batches=calibration_batches,
        )
        print("BN calibration batches: {}".format(consumed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    summary, _ = evaluate_model(
        model,
        loader,
        config,
        device,
        output_dir=output_dir,
        max_visualizations=args.visualizations,
        export_predictions=not args.no_export_predictions,
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    metadata = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "ignored_removed_parameters": list(checkpoint_report.ignored_removed),
        },
        "dataset": dataset_name,
        "split": args.split,
        "requested_max_samples": args.max_samples,
        "evaluated_samples": summary["samples"],
        "bn_calibration_batches": calibration_batches,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print((output_dir / "summary.json").read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
