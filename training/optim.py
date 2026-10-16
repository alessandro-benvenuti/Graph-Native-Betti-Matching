"""Baseline optimizer and per-iteration polynomial learning-rate schedule."""

from __future__ import annotations

from typing import Mapping

import torch


def build_optimizer(config: Mapping, model):
    settings = config["training"]["optimizer"]
    if settings["name"] != "adamw":
        raise ValueError("only AdamW is supported")
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["base_lr"]),
        weight_decay=float(settings["weight_decay"]),
        eps=float(settings["epsilon"]),
        betas=tuple(float(value) for value in settings["betas"]),
    )


def build_scheduler(config: Mapping, optimizer, iterations_per_epoch: int):
    if iterations_per_epoch <= 0:
        raise ValueError("iterations_per_epoch must be positive")
    settings = config["training"]["scheduler"]
    if settings["name"] != "polynomial":
        raise ValueError("only the polynomial scheduler is supported")
    warmup_iterations = float(config["training"]["warmup_epochs"]) * iterations_per_epoch
    # A continuation may deliberately use a fresh schedule shorter than the
    # absolute training epoch counter (for example epochs 501--750 with a
    # 250-epoch tail schedule).  The option is part of the resolved scientific
    # configuration, so interruption resumes rebuild the same LambdaLR closure.
    duration_epochs = int(
        settings.get("duration_epochs", config["training"]["epochs"])
    )
    training_iterations = duration_epochs * iterations_per_epoch
    initial_ratio = float(settings["warmup_lr"]) / float(
        config["training"]["optimizer"]["base_lr"]
    )
    power = float(settings["power"])

    def multiplier(iteration):
        if warmup_iterations > 0 and iteration < warmup_iterations:
            return initial_ratio + (1.0 - initial_ratio) * iteration / warmup_iterations
        progress = (iteration - warmup_iterations) / float(training_iterations)
        return max(0.0, 1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def reset_optimizer_learning_rate(config: Mapping, optimizer):
    """Apply the configured base LR after restoring AdamW moments.

    ``Optimizer.load_state_dict`` restores the exhausted source-run learning
    rate.  A deliberate continuation-schedule restart keeps the moment tensors
    while rebasing every parameter group to the new configured base LR.
    """

    base_lr = float(config["training"]["optimizer"]["base_lr"])
    for group in optimizer.param_groups:
        group["lr"] = base_lr
        group["initial_lr"] = base_lr


__all__ = ["build_optimizer", "build_scheduler", "reset_optimizer_learning_rate"]
