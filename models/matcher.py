"""Bipartite matching used by the 3D RelationFormer baseline."""

from __future__ import annotations

from typing import Mapping

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """Match predicted queries to graph nodes using class and L1 costs."""

    def __init__(self, class_cost: float, node_cost: float, dimensions: int = 3):
        super().__init__()
        if class_cost == 0 and node_cost == 0:
            raise ValueError("at least one matching cost must be non-zero")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.class_cost = float(class_cost)
        self.node_cost = float(node_cost)
        self.dimensions = int(dimensions)

    @torch.no_grad()
    def forward(self, outputs: Mapping, targets: Mapping):
        predicted_nodes = outputs["pred_nodes"]
        predicted_logits = outputs["pred_logits"]
        target_nodes = targets["nodes"]
        batch_size, query_count = predicted_nodes.shape[:2]
        if len(target_nodes) != batch_size:
            raise ValueError("targets['nodes'] must contain one tensor per sample")

        assignments = []
        object_probability = predicted_logits.softmax(-1)[..., 1]
        for sample in range(batch_size):
            truth = target_nodes[sample].to(predicted_nodes.device)
            if truth.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64)
                assignments.append((empty, empty.clone()))
                continue
            node_cost = torch.cdist(
                predicted_nodes[sample, :, : self.dimensions],
                truth[:, : self.dimensions],
                p=1,
            )
            class_cost = -object_probability[sample].unsqueeze(1).expand_as(node_cost)
            cost = self.node_cost * node_cost + self.class_cost * class_cost
            source, target = linear_sum_assignment(cost.detach().cpu())
            assignments.append(
                (
                    torch.as_tensor(source, dtype=torch.int64),
                    torch.as_tensor(target, dtype=torch.int64),
                )
            )
        return assignments


def build_matcher(config: Mapping, dimensions: int | None = None) -> HungarianMatcher:
    matcher = config["model"]["matcher"]
    if matcher["type"] != "hungarian":
        raise ValueError("only the Hungarian matcher is supported")
    if dimensions is None:
        dimensions = int(config["data"]["spatial_dims"])
    return HungarianMatcher(
        class_cost=matcher["class_cost"],
        node_cost=matcher["node_cost"],
        dimensions=dimensions,
    )


__all__ = ["HungarianMatcher", "build_matcher"]
