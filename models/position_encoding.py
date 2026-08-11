"""Checkpoint-compatible 3D sine positional encoding."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionEmbeddingSine3D(nn.Module):
    """Generate the baseline `(W, H, D)` sine/cosine channel ordering."""

    def __init__(
        self,
        channels: int = 64,
        temperature: float = 10_000.0,
        normalize: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if scale is not None and not normalize:
            raise ValueError("normalize must be true when scale is provided")

        self.orig_channels = int(channels)
        # Each spatial axis contributes an even number of sine/cosine channels.
        self.channels = math.ceil(channels / 6) * 2
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.scale = 2 * math.pi if scale is None else float(scale)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Return `[B,C,D,H,W]` positions for a boolean `[B,D,H,W]` mask."""

        if mask.ndim != 4 or mask.dtype is not torch.bool:
            raise ValueError("mask must be a boolean tensor with shape [B,D,H,W]")

        not_mask = ~mask
        depth = not_mask.cumsum(1, dtype=torch.float32)
        height = not_mask.cumsum(2, dtype=torch.float32)
        width = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            epsilon = 1.0e-6
            depth = (depth - 0.5) / (depth[:, -1:, :, :] + epsilon) * self.scale
            height = (height - 0.5) / (height[:, :, -1:, :] + epsilon) * self.scale
            width = (width - 0.5) / (width[:, :, :, -1:] + epsilon) * self.scale

        frequencies = torch.arange(
            self.channels, dtype=torch.float32, device=mask.device
        )
        frequencies = self.temperature ** (
            2 * torch.floor(frequencies / 2) / self.channels
        )

        def encode(coordinates: torch.Tensor) -> torch.Tensor:
            phase = coordinates[..., None] / frequencies
            return torch.stack(
                (phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=4
            ).flatten(4)

        # Preserve the original channel order: height, depth, width.
        position = torch.cat(
            (encode(height), encode(depth), encode(width)), dim=4
        ).permute(0, 4, 1, 2, 3)
        return position[:, : self.orig_channels]


__all__ = ["PositionEmbeddingSine3D"]
