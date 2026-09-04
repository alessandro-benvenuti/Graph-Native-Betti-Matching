"""Coordinate-only and structure-aware matching for 3D RelationFormer."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _empty_assignment():
    empty = torch.empty(0, dtype=torch.int64)
    return empty, empty.clone()


class HungarianMatcher(nn.Module):
    """Match predicted queries to graph nodes using class and L1 costs."""

    requires_structure = False

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
    def forward(
        self,
        outputs: Mapping,
        targets: Mapping,
        *,
        predicted_structure: torch.Tensor | None = None,
        candidate_indices=None,
    ):
        del predicted_structure, candidate_indices
        predicted_nodes = outputs["pred_nodes"]
        predicted_logits = outputs["pred_logits"]
        target_nodes = targets["nodes"]
        batch_size, _ = predicted_nodes.shape[:2]
        if len(target_nodes) != batch_size:
            raise ValueError("targets['nodes'] must contain one tensor per sample")

        assignments = []
        object_probability = predicted_logits.softmax(-1)[..., 1]
        for sample in range(batch_size):
            truth = target_nodes[sample].to(predicted_nodes.device)
            if truth.numel() == 0:
                assignments.append(_empty_assignment())
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


class FusedGromovWassersteinMatcher(nn.Module):
    """Match nodes with semi-relaxed FGW followed by one-to-one projection.

    Ground-truth nodes are the fixed-mass source measure and prediction queries
    are the relaxed target measure. This lets the coupling select a subset of
    surplus RelationFormer queries before the final discrete projection.
    """

    requires_structure = True

    def __init__(
        self,
        class_cost: float,
        node_cost: float,
        structure_weight: float,
        *,
        dimensions: int = 3,
        candidate_count: int = 32,
        max_iter: int = 100,
        tolerance: float = 1e-7,
        random_state: int = 0,
    ):
        super().__init__()
        if class_cost == 0 and node_cost == 0:
            raise ValueError("at least one feature matching cost must be non-zero")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not 0.0 <= structure_weight <= 1.0:
            raise ValueError("structure_weight must lie in [0, 1]")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.class_cost = float(class_cost)
        self.node_cost = float(node_cost)
        self.structure_weight = float(structure_weight)
        self.dimensions = int(dimensions)
        self.candidate_count = int(candidate_count)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)

    def _feature_cost(self, outputs: Mapping, truth: torch.Tensor, sample: int):
        predicted_nodes = outputs["pred_nodes"]
        object_probability = outputs["pred_logits"].softmax(-1)[..., 1]
        node_cost = torch.cdist(
            truth[:, : self.dimensions],
            predicted_nodes[sample, :, : self.dimensions],
            p=1,
        )
        class_cost = -object_probability[sample].unsqueeze(0).expand_as(node_cost)
        return self.node_cost * node_cost + self.class_cost * class_cost

    @torch.no_grad()
    def matching_candidates(self, outputs: Mapping, targets: Mapping):
        """Select a bounded pool while retaining every unary Hungarian match."""

        predicted_nodes = outputs["pred_nodes"]
        object_probability = outputs["pred_logits"].softmax(-1)[..., 1]
        query_count = predicted_nodes.shape[1]
        candidates = []
        for sample, nodes in enumerate(targets["nodes"]):
            truth = nodes.to(predicted_nodes.device)
            target_count = int(truth.shape[0])
            if target_count == 0:
                candidates.append(
                    torch.empty(0, dtype=torch.long, device=predicted_nodes.device)
                )
                continue
            if target_count > query_count:
                raise ValueError(
                    "hard one-to-one matching requires at least as many queries as targets"
                )
            feature_cost = self._feature_cost(outputs, truth, sample)
            _, unary_sources = linear_sum_assignment(feature_cost.detach().cpu())
            selected = [int(index) for index in unary_sources]
            selected_set = set(selected)
            requested = min(query_count, max(target_count, self.candidate_count))
            for index in object_probability[sample].argsort(descending=True).tolist():
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
                if len(selected) == requested:
                    break
            candidates.append(
                torch.as_tensor(
                    selected, dtype=torch.long, device=predicted_nodes.device
                )
            )
        return candidates

    @staticmethod
    def target_structure(
        node_count: int,
        edges,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a symmetric, loop-free target adjacency matrix."""

        structure = torch.zeros((node_count, node_count), dtype=dtype, device=device)
        edge_tensor = torch.as_tensor(
            edges, dtype=torch.long, device=device
        ).reshape(-1, 2)
        if edge_tensor.numel() == 0:
            return structure
        if bool(((edge_tensor < 0) | (edge_tensor >= node_count)).any()):
            raise ValueError("target edge index lies outside the target node range")
        edge_tensor = edge_tensor[edge_tensor[:, 0] != edge_tensor[:, 1]]
        if edge_tensor.numel():
            structure[edge_tensor[:, 0], edge_tensor[:, 1]] = 1.0
            structure[edge_tensor[:, 1], edge_tensor[:, 0]] = 1.0
        return structure

    @staticmethod
    def harden_transport(transport) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a target-by-query coupling onto a global 1:1 assignment."""

        array = np.asarray(transport, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("transport must be a matrix")
        if not np.isfinite(array).all():
            raise ValueError("transport contains non-finite values")
        target, source = linear_sum_assignment(-array)
        return (
            torch.as_tensor(source, dtype=torch.int64),
            torch.as_tensor(target, dtype=torch.int64),
        )

    def _solve_transport(self, feature_cost, target_structure, predicted_structure):
        try:
            from ot.gromov import semirelaxed_fused_gromov_wasserstein
        except ImportError as error:
            raise ImportError(
                "The FGW matcher requires Python Optimal Transport (POT). "
                "Install the project's requirements before using matcher.type=fgw."
            ) from error

        target_count, query_count = feature_cost.shape
        target_mass = np.full(target_count, 1.0 / target_count, dtype=np.float64)

        # This sparse coordinate/class plan is feasible for the semi-relaxed
        # problem and gives the non-convex solver a stable initialization.
        initial_target, initial_source = linear_sum_assignment(feature_cost)
        initial = np.zeros((target_count, query_count), dtype=np.float64)
        initial[initial_target, initial_source] = target_mass[initial_target]

        return semirelaxed_fused_gromov_wasserstein(
            feature_cost,
            target_structure,
            predicted_structure,
            p=target_mass,
            loss_fun="square_loss",
            symmetric=True,
            alpha=self.structure_weight,
            G0=initial,
            max_iter=self.max_iter,
            tol_rel=self.tolerance,
            tol_abs=self.tolerance,
            random_state=self.random_state,
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping,
        targets: Mapping,
        *,
        predicted_structure=None,
        candidate_indices=None,
    ):
        if predicted_structure is None:
            raise ValueError("FGW matching requires predicted_structure")
        predicted_nodes = outputs["pred_nodes"]
        target_nodes = targets["nodes"]
        target_edges = targets["edges"]
        batch_size, query_count = predicted_nodes.shape[:2]
        if len(target_nodes) != batch_size or len(target_edges) != batch_size:
            raise ValueError("targets must contain nodes and edges for every sample")
        if candidate_indices is None:
            candidate_indices = self.matching_candidates(outputs, targets)
        if (
            len(predicted_structure) != batch_size
            or len(candidate_indices) != batch_size
        ):
            raise ValueError("FGW structures/candidates must contain one item per sample")
        assignments = []
        for sample in range(batch_size):
            truth = target_nodes[sample].to(predicted_nodes.device)
            target_count = int(truth.shape[0])
            if target_count == 0:
                assignments.append(_empty_assignment())
                continue
            if target_count > query_count:
                raise ValueError(
                    "hard one-to-one matching requires at least as many queries as targets"
                )

            candidates = candidate_indices[sample].to(
                predicted_nodes.device, dtype=torch.long
            )
            if candidates.numel() < target_count:
                raise ValueError(
                    "FGW candidate pool cannot contain fewer queries than targets"
                )
            if candidates.unique().numel() != candidates.numel():
                raise ValueError("FGW candidate indices must be unique")
            feature_cost = self._feature_cost(outputs, truth, sample)[:, candidates]
            truth_structure = self.target_structure(
                target_count,
                target_edges[sample],
                dtype=feature_cost.dtype,
                device=feature_cost.device,
            )
            prediction_structure = predicted_structure[sample].clone()
            if prediction_structure.shape != (
                candidates.numel(),
                candidates.numel(),
            ):
                raise ValueError("predicted structure does not match its candidate pool")
            prediction_structure.fill_diagonal_(0.0)

            transport = self._solve_transport(
                feature_cost.detach().double().cpu().numpy(),
                truth_structure.detach().double().cpu().numpy(),
                prediction_structure.detach().double().cpu().numpy(),
            )
            transport = np.asarray(transport, dtype=np.float64)
            if transport.shape != (target_count, candidates.numel()):
                raise RuntimeError("FGW solver returned an unexpected transport shape")
            if not np.isfinite(transport).all() or (transport < -self.tolerance).any():
                raise RuntimeError("FGW solver returned an invalid transport plan")
            if not np.allclose(
                transport.sum(axis=1),
                1.0 / target_count,
                rtol=1e-5,
                atol=max(self.tolerance, 1e-8),
            ):
                raise RuntimeError("FGW transport violates the fixed target marginal")
            local_source, matched_target = self.harden_transport(transport)
            assignments.append((candidates.cpu()[local_source], matched_target))
        return assignments


def build_matcher(config: Mapping, dimensions: int | None = None) -> nn.Module:
    matcher = config["model"]["matcher"]
    if dimensions is None:
        dimensions = int(config["data"]["spatial_dims"])
    common = dict(
        class_cost=matcher["class_cost"],
        node_cost=matcher["node_cost"],
        dimensions=dimensions,
    )
    if matcher["type"] == "hungarian":
        return HungarianMatcher(**common)
    if matcher["type"] == "fgw":
        return FusedGromovWassersteinMatcher(
            **common,
            structure_weight=matcher["structure_weight"],
            candidate_count=matcher["candidate_count"],
            max_iter=matcher["max_iter"],
            tolerance=matcher["tolerance"],
            random_state=matcher.get("random_state", 0),
        )
    raise ValueError("matcher.type must be hungarian or fgw")


__all__ = [
    "FusedGromovWassersteinMatcher",
    "HungarianMatcher",
    "build_matcher",
]
