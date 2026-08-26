#!/usr/bin/env python3
"""Train the supported 3D RelationFormer configurations."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from configs import load_config, validate_config
from data.loaders import build_data_loaders, build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from training import (
    Trainer,
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from training.checkpoint import load_training_checkpoint
from training.distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
    prepare_model_for_distributed,
)
from training.tracking import build_tracker


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
    parser.add_argument(
        "--distributed", action="store_true", help="enable torchrun/DDP execution"
    )
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
    if args.distributed:
        config["runtime"]["distributed"] = True
    return config


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
    rank, world_size, local_rank = initialize_distributed(
        bool(config["runtime"].get("distributed", False))
    )
    seed = int(config["experiment"]["seed"])
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if config["runtime"]["deterministic"]:
        _enable_deterministic_algorithms()

    device = torch.device(
        "cuda:{}".format(local_rank)
        if world_size > 1
        else config["runtime"]["device"]
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA but CUDA is unavailable")
    train_loader, validation_loader = build_data_loaders(
        config, rank=rank, world_size=world_size
    )
    metric_loader = None
    metric_config = config["evaluation"]["training_metrics"]
    if metric_config["enabled"] and rank == 0:
        metric_loader = build_evaluation_loader(
            config,
            dataset_name=metric_config["dataset"],
            split="val",
            max_samples=metric_config["max_samples"],
        )
    raw_model = build_model(config)
    if world_size > 1:
        # Per-rank batches become small when the fixed global batch is split.
        # Synchronizing BatchNorm preserves global-batch statistics.
        raw_model = prepare_model_for_distributed(raw_model)
    raw_model = raw_model.to(device)
    criterion = build_criterion(config, raw_model).to(device)
    validation_config = copy.deepcopy(config)
    validation_config["loss"]["edge"]["balancing"]["mode"] = "none"
    # Preserve the effective legacy evaluator: it used the full focal settings,
    # no ratio upsampling, and capped validation positives at 40 per graph.
    validation_config["loss"]["edge"]["candidates"]["positive_cap"] = 40
    validation_criterion = build_criterion(
        validation_config, raw_model, validation=True
    ).to(device)
    optimizer = build_optimizer(config, raw_model)
    scheduler = build_scheduler(config, optimizer, len(train_loader))

    if args.resume and args.initial_weights:
        raise ValueError("--resume and --initial-weights are mutually exclusive")
    start_epoch = int(config["training"]["start_epoch"])
    start_iteration = 0
    trainer_state = {}
    if args.initial_weights:
        load_legacy_model_checkpoint(raw_model, args.initial_weights, map_location="cpu")
    if args.resume:
        start_epoch, start_iteration, trainer_state = load_training_checkpoint(
            args.resume,
            raw_model,
            optimizer,
            scheduler,
            rank=rank,
            loader_generator=getattr(train_loader, "generator", None),
            return_trainer_state=True,
        )
        checkpoint_world_size = trainer_state.get("world_size")
        if (
            checkpoint_world_size is not None
            and int(checkpoint_world_size) != world_size
        ):
            raise ValueError(
                "resume requires the original world size: checkpoint={} current={}".format(
                    checkpoint_world_size, world_size
                )
            )

    model = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            # relation_embed is intentionally evaluated by GraphCriterion from
            # the tokens returned by forward(), so unused-parameter graph
            # discovery would incorrectly mark it before the loss is built.
            find_unused_parameters=False,
        )

    run_dir = (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
    )
    tracker = None
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        tracker = build_tracker(
            config,
            run_dir,
            resume=bool(args.resume),
            launch_metadata={
                "initial_weights": args.initial_weights,
                "resume_checkpoint": args.resume,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "world_size": world_size,
            },
        )
    barrier()
    try:
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
            tracker,
            metric_loader,
            evaluation_model=raw_model,
            rank=rank,
            world_size=world_size,
        ).fit(
            start_epoch=start_epoch,
            start_iteration=start_iteration,
            trainer_state=trainer_state,
        )
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise
    else:
        if tracker is not None:
            tracker.finish(exit_code=0)
        # On failure, let process exit tear down NCCL. Destroying the process
        # group before PyTorch's distributed excepthook runs hides the original
        # traceback behind "default process group has not been initialized".
        cleanup_distributed()


if __name__ == "__main__":
    main()
