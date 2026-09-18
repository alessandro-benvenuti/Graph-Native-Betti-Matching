"""Differentiable node-and-edge filtration helpers for graph Betti losses."""

from __future__ import annotations

import torch


def node_edge_confidences(
    node_probabilities: torch.Tensor,
    edge_probabilities: torch.Tensor,
    candidate_edges: torch.Tensor,
    *,
    aggregation: str = "hybrid",
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combine endpoint existence and relation confidence.

    The returned confidence never exceeds either endpoint probability or the
    relation probability.  Consequently ``1 - confidence`` is a valid edge
    filtration for vertex filtrations ``1 - node_probability``.
    """
    if node_probabilities.ndim != 1:
        raise ValueError("node_probabilities must have shape [num_nodes].")
    if edge_probabilities.ndim != 1:
        raise ValueError("edge_probabilities must have shape [num_edges].")
    if candidate_edges.ndim != 2 or candidate_edges.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2].")
    if edge_probabilities.shape[0] != candidate_edges.shape[0]:
        raise ValueError("candidate_edges and edge_probabilities must agree.")
    if candidate_edges.numel():
        if int(candidate_edges.min()) < 0 or int(candidate_edges.max()) >= len(
            node_probabilities
        ):
            raise ValueError("candidate_edges contains an invalid node index.")
    if aggregation not in {"min", "product", "hybrid"}:
        raise ValueError("aggregation must be min, product, or hybrid.")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")

    edges = candidate_edges.to(
        device=node_probabilities.device, dtype=torch.long
    )
    edge_probabilities = edge_probabilities.to(node_probabilities.device)
    left = node_probabilities[edges[:, 0]]
    right = node_probabilities[edges[:, 1]]
    product = left * right * edge_probabilities
    minimum = torch.stack((left, right, edge_probabilities), dim=-1).amin(-1)
    if aggregation == "min":
        return minimum
    if aggregation == "product":
        return product
    return alpha * minimum + (1.0 - alpha) * product


__all__ = ["node_edge_confidences"]
