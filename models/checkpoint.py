"""Compatibility helpers for checkpoints produced by the legacy 3D project."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


_MODEL_CONTAINER_KEYS = ("net", "model", "state_dict")
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")
_REMOVED_PREFIXES = (
    "backbone_domain_discriminator.",
    "instance_domain_discriminator.",
)


@dataclass(frozen=True)
class CheckpointCompatibility:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    ignored_removed: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.missing and not self.unexpected and not self.shape_mismatches


def extract_model_state(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    """Extract and normalize a model state dict from common legacy containers."""

    candidate = checkpoint
    for key in _MODEL_CONTAINER_KEYS:
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("Checkpoint does not contain a non-empty model state dict")

    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value
    if not normalized:
        raise ValueError("Checkpoint model state contains no tensors")
    return normalized


def checkpoint_compatibility(
    model: nn.Module, checkpoint: Mapping
) -> CheckpointCompatibility:
    """Compare active model tensors while allowing explicitly removed domain keys."""

    expected = model.state_dict()
    observed = extract_model_state(checkpoint)
    ignored = tuple(
        sorted(
            key
            for key in observed
            if any(key.startswith(prefix) for prefix in _REMOVED_PREFIXES)
        )
    )
    active_observed = {key: value for key, value in observed.items() if key not in ignored}
    missing = tuple(sorted(set(expected) - set(active_observed)))
    unexpected = tuple(sorted(set(active_observed) - set(expected)))
    shape_mismatches = tuple(
        sorted(
            f"{key}: checkpoint={tuple(active_observed[key].shape)} "
            f"model={tuple(expected[key].shape)}"
            for key in set(expected) & set(active_observed)
            if expected[key].shape != active_observed[key].shape
        )
    )
    return CheckpointCompatibility(
        missing=missing,
        unexpected=unexpected,
        shape_mismatches=shape_mismatches,
        ignored_removed=ignored,
    )


def load_checkpoint_file(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping:
    """Load a trusted local checkpoint, using memory mapping when supported."""

    options = {"map_location": map_location}
    parameters = inspect.signature(torch.load).parameters
    if "mmap" in parameters:
        options["mmap"] = True
    if "weights_only" in parameters:
        # Legacy MONAI checkpoints include optimizer/scheduler containers.
        options["weights_only"] = False
    # PyTorch releases used by the cluster predate os.PathLike support here.
    checkpoint = torch.load(str(Path(path)), **options)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    return checkpoint


def load_legacy_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointCompatibility:
    """Load all active tensors and reject silent active-model incompatibilities."""

    checkpoint = load_checkpoint_file(path, map_location=map_location)
    report = checkpoint_compatibility(model, checkpoint)
    if not report.compatible:
        raise RuntimeError(
            "Checkpoint is incompatible with the active model: "
            f"missing={report.missing}, unexpected={report.unexpected}, "
            f"shape_mismatches={report.shape_mismatches}"
        )
    state = extract_model_state(checkpoint)
    active_state = {
        key: value
        for key, value in state.items()
        if not any(key.startswith(prefix) for prefix in _REMOVED_PREFIXES)
    }
    model.load_state_dict(active_state, strict=True)
    return report


__all__ = [
    "CheckpointCompatibility",
    "checkpoint_compatibility",
    "extract_model_state",
    "load_checkpoint_file",
    "load_legacy_model_checkpoint",
]
