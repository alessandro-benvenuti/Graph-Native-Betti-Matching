"""Shared sample types, collation, and deterministic worker seeding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SamplePaths:
    """Files forming one image/segmentation/graph training example."""

    image: Path
    segmentation: Path
    graph: Path
    sample_id: str


# Keep the six-part interface consumed by the original model/trainer. Each
# dataset item wraps every component in a one-element list; collation removes
# that wrapper while preserving variable graph sizes.
DatasetSample = Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[object],
    List[int],
]


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Remove framework tensor subclasses before model-facing collation.

    MONAI spatial/intensity transforms may return ``MetaTensor`` even when the
    input is an ordinary tensor.  Letting that subclass reach SyncBatchNorm
    makes its custom-autograd output a view, which is incompatible with the
    encoder's in-place activations.  Dataset metadata is not consumed anywhere
    in this project, so retain only the underlying torch.Tensor here.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(value).__name__}")
    if type(value) is not torch.Tensor:
        as_tensor = getattr(value, "as_tensor", None)
        if callable(as_tensor):
            value = as_tensor()
        if type(value) is not torch.Tensor:
            value = value.as_subclass(torch.Tensor)
    if type(value) is not torch.Tensor:
        raise TypeError(
            f"Could not convert {type(value).__name__} to a plain torch.Tensor"
        )
    return value


def image_graph_collate(batch: Sequence[DatasetSample]):
    """Stack dense volumes and retain variable-sized graphs as Python lists."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    images = torch.cat(
        [torch.stack([_plain_tensor(value) for value in item[0]]) for item in batch]
    ).contiguous()
    segmentations = torch.cat(
        [torch.stack([_plain_tensor(value) for value in item[1]]) for item in batch]
    ).contiguous()
    nodes = [value for item in batch for value in item[2]]
    edges = [value for item in batch for value in item[3]]
    projection_positions = [value for item in batch for value in item[4]]
    domains = torch.tensor(
        [value for item in batch for value in item[5]], dtype=torch.long
    )
    return [images, segmentations, nodes, edges, projection_positions, domains]


def seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


__all__ = ["DatasetSample", "SamplePaths", "image_graph_collate", "seed_data_worker"]
