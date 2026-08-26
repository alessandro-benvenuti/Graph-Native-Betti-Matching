"""Training-state checkpoint persistence and strict resume."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import random
import shutil
from typing import Mapping

import numpy as np
import torch


def capture_runtime_state(loader_generator=None):
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_runtime_state(state, loader_generator=None):
    if not state:
        return
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    iteration,
    *,
    runtime_states=None,
    trainer_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "iteration": int(iteration),
        "runtime_states": runtime_states,
        "trainer_state": dict(trainer_state or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(temporary, path)


def save_runtime_state(path, state):
    """Atomically persist one rank's small RNG/runtime state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, str(temporary))
    os.replace(temporary, path)


def load_runtime_state(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(str(Path(path)), **options)


def alias_training_checkpoint(source, destination):
    """Atomically make another name for an identical checkpoint payload.

    A hard link avoids serializing and storing the same multi-gigabyte payload
    several times when best/latest/interval checkpoints coincide.  The copy
    fallback covers filesystems that do not support hard links.
    """
    source = Path(source)
    destination = Path(destination)
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def load_training_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    rank: int = 0,
    loader_generator=None,
    return_trainer_state: bool = False,
):
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
    runtime_states = checkpoint.get("runtime_states")
    if runtime_states:
        selected = runtime_states[min(int(rank), len(runtime_states) - 1)]
        restore_runtime_state(selected, loader_generator)
    result = (int(checkpoint.get("epoch", 0)), int(checkpoint.get("iteration", 0)))
    if return_trainer_state:
        return (*result, dict(checkpoint.get("trainer_state") or {}))
    return result


__all__ = [
    "alias_training_checkpoint",
    "capture_runtime_state",
    "load_runtime_state",
    "load_training_checkpoint",
    "restore_runtime_state",
    "save_runtime_state",
    "save_training_checkpoint",
]
