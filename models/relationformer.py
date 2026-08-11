"""Minimal 3D RelationFormer model used by the supported baseline."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .deformable_transformer import (
    DeformableTransformer,
    build_def_detr_transformer,
)
from .position_encoding import PositionEmbeddingSine3D


class MLP(nn.Module):
    """Baseline feed-forward head with checkpoint-compatible layer names."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, destination)
            for source, destination in zip(
                [input_dim, *hidden], [*hidden, output_dim]
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, layer in enumerate(self.layers):
            output = (
                F.relu(layer(output))
                if index < self.num_layers - 1
                else layer(output)
            )
        return output


class RelationFormer(nn.Module):
    """Encode a 3D volume and predict graph-node tokens and coordinates."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: DeformableTransformer,
        config: Mapping,
    ) -> None:
        super().__init__()
        model = config["model"]
        decoder_config = model["decoder"]
        hidden_dim = int(decoder_config["hidden_dim"])
        object_queries = int(decoder_config["object_queries"])
        relation_tokens = int(decoder_config["relation_tokens"])
        dummy_tokens = int(decoder_config["dummy_tokens"])

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = object_queries + relation_tokens + dummy_tokens
        self.hidden_dim = hidden_dim
        self.position_embedding = PositionEmbeddingSine3D(channels=hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv3d(encoder.num_features, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, int(model["num_classes"]))
        self.coord_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.obj_token = object_queries

        relation_input_dim = hidden_dim * (2 + relation_tokens)
        self.relation_embed = MLP(relation_input_dim, hidden_dim, 2, 3)

    def forward(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Return all tokens, node predictions, and projected encoder features.

        `samples` has shape `[B,C,D,H,W]`. The returned token tensor has shape
        `[B,Q,H]`; `pred_logits` is `[B,object_queries,num_classes]` and
        `pred_nodes` is `[B,object_queries,6]` in normalized coordinates.
        """

        if samples.ndim != 5:
            raise ValueError("samples must have shape [B,C,D,H,W]")

        encoded = self.encoder(samples)
        mask = torch.zeros(
            encoded.shape[0],
            *encoded.shape[2:],
            dtype=torch.bool,
            device=encoded.device,
        )
        positions = self.position_embedding(mask)
        projected_features = self.input_proj(encoded)
        tokens = self.decoder(
            projected_features,
            mask,
            self.query_embed.weight,
            positions,
        )
        object_tokens = tokens[..., : self.obj_token, :]
        predictions = {
            "pred_logits": self.class_embed(object_tokens),
            "pred_nodes": self.coord_embed(object_tokens).sigmoid(),
        }
        return tokens, predictions, projected_features


def build_relationformer(config: Mapping) -> RelationFormer:
    """Build the supported checkpoint-compatible 3D model."""

    from .seresnet import build_seresnet

    return RelationFormer(
        encoder=build_seresnet(config),
        decoder=build_def_detr_transformer(config),
        config=config,
    )


__all__ = ["MLP", "RelationFormer", "build_relationformer"]
