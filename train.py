#!/usr/bin/env python3
"""Train the supported 3D RelationFormer configurations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from configs import load_config, validate_config
from data.loaders import build_data_loaders
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from training import (
    Trainer,
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from training.checkpoint import load_training_checkpoint


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="resume complete training state")
    parser.add_argument("--initial-weights", help="load model weights without optimizer state")
    parser.add_argument("--device", help="override runtime.device")
    parser.add_argument("--output-dir", help="override experiment.output_dir")
    parser.add_argument("--run-name", help="override experiment.name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    return parser


def _apply_operational_overrides(config, args):
    config = copy.deepcopy(config)
    if args.device:
        config["runtime"]["device"] = args.device
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.run_name:
        config["experiment"]["name"] = args.run_name
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.workers is not None:
        config["runtime"]["workers"] = args.workers
    return config


def _writer(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is unavailable; continuing without event logging")
        return None
    return SummaryWriter(log_dir=str(path))


def _enable_deterministic_algorithms():
    """Enable determinism across legacy and current PyTorch APIs."""

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    else:
        raise RuntimeError(
            "This PyTorch version cannot enforce runtime.deterministic=true"
        )


def main():
    args = _parser().parse_args()
    config = _apply_operational_overrides(load_config(args.config), args)
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config["runtime"]["deterministic"]:
        _enable_deterministic_algorithms()

    device = torch.device(config["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA but CUDA is unavailable")
    train_loader, validation_loader = build_data_loaders(config)
    model = build_model(config).to(device)
    criterion = build_criterion(config, model).to(device)
    validation_config = copy.deepcopy(config)
    validation_config["loss"]["edge"]["balancing"]["mode"] = "none"
    # Preserve the effective legacy evaluator: it used the full focal settings,
    # no ratio upsampling, and capped validation positives at 40 per graph.
    validation_config["loss"]["edge"]["candidates"]["positive_cap"] = 40
    validation_criterion = build_criterion(
        validation_config, model, validation=True
    ).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, len(train_loader))

    if args.resume and args.initial_weights:
        raise ValueError("--resume and --initial-weights are mutually exclusive")
    start_epoch = int(config["training"]["start_epoch"])
    start_iteration = 0
    if args.initial_weights:
        load_legacy_model_checkpoint(model, args.initial_weights, map_location="cpu")
    if args.resume:
        start_epoch, start_iteration = load_training_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    run_dir = (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    writer = _writer(run_dir / "tensorboard")
    Trainer(
        model,
        criterion,
        validation_criterion,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        config,
        device,
        writer,
    ).fit(start_epoch=start_epoch, start_iteration=start_iteration)


if __name__ == "__main__":
    main()
