"""Shared metrics and deterministic sampling for optional dataset tests."""

from __future__ import annotations

from typing import Sequence, TypeVar

import torch
import torch.nn.functional as F

from data.augmentations import coordinates_to_voxel_indices


Item = TypeVar("Item")


def select_evenly_spaced(items: Sequence[Item], count: int) -> list[Item]:
    """Select up to ``count`` deterministic entries across a sorted sequence."""

    if count <= 0:
        raise ValueError("Sample count must be positive")
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    indices = [round(position * last / (count - 1)) for position in range(count)]
    return [items[index] for index in indices]


def node_foreground_hit_rate(
    segmentation: torch.Tensor,
    nodes: torch.Tensor,
) -> float:
    indices = coordinates_to_voxel_indices(nodes, segmentation.shape[-3:])
    values = segmentation[0, indices[:, 0], indices[:, 1], indices[:, 2]]
    return float((values > 0).float().mean()) if values.numel() else 1.0


def node_foreground_neighbourhood_hit_rate(
    segmentation: torch.Tensor,
    nodes: torch.Tensor,
    *,
    radius_voxels: int,
) -> float:
    """Return the node rate within a Chebyshev-radius foreground neighbourhood."""

    if radius_voxels < 0:
        raise ValueError("radius_voxels must be non-negative")
    if radius_voxels == 0:
        return node_foreground_hit_rate(segmentation, nodes)
    node_indices = coordinates_to_voxel_indices(nodes, segmentation.shape[-3:])
    if not node_indices.numel():
        return 1.0
    kernel_size = 2 * radius_voxels + 1
    foreground = (segmentation > 0).float().unsqueeze(0)
    neighbourhood = F.max_pool3d(
        foreground,
        kernel_size=kernel_size,
        stride=1,
        padding=radius_voxels,
    ).squeeze(0)
    values = neighbourhood[
        0,
        node_indices[:, 0],
        node_indices[:, 1],
        node_indices[:, 2],
    ]
    return float((values > 0).float().mean())
