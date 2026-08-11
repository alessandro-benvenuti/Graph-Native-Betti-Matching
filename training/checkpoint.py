"""Training-state checkpoint persistence and strict resume."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping

import torch


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, iteration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "iteration": int(iteration),
        },
        str(path),
    )


def load_training_checkpoint(path, model, optimizer=None, scheduler=None):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    checkpoint = torch.load(str(Path(path)), **options)
    if not isinstance(checkpoint, Mapping) or "net" not in checkpoint:
        raise ValueError("training checkpoint must contain a 'net' state")
    model.load_state_dict(checkpoint["net"], strict=True)
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("iteration", 0))


__all__ = ["load_training_checkpoint", "save_training_checkpoint"]
