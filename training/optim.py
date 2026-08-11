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
    training_iterations = int(config["training"]["epochs"]) * iterations_per_epoch
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


__all__ = ["build_optimizer", "build_scheduler"]
