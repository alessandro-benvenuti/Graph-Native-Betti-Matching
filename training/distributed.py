"""Small helpers for single-node torchrun/DDP execution."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def prepare_model_for_distributed(model: torch.nn.Module) -> torch.nn.Module:
    """Convert BatchNorm safely for one-process-per-GPU DDP execution."""

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # SyncBatchNorm uses custom autograd.  Avoid view/in-place conflicts in
    # both the local encoder stem and MONAI's nested residual blocks.  This
    # does not alter model parameters, state-dict keys, or activation values.
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model


def initialize_distributed(enabled: bool):
    """Initialize the process group and return ``(rank, world_size, local_rank)``."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not enabled:
        raise RuntimeError(
            "torchrun launched multiple processes but distributed training is disabled"
        )
    if enabled and world_size <= 1:
        raise RuntimeError(
            "runtime.distributed=true requires torchrun with more than one process"
        )
    if world_size == 1:
        return 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("distributed H100 training requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size(), local_rank


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "barrier",
    "cleanup_distributed",
    "initialize_distributed",
    "prepare_model_for_distributed",
]
