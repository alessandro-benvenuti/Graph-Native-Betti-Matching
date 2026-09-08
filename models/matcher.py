"""Coordinate-only and structure-aware matching for 3D RelationFormer."""

from __future__ import annotations

import time
import warnings
from typing import Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _empty_assignment():
    empty = torch.empty(0, dtype=torch.int64)
    return empty, empty.clone()


def fgw_objective_terms(
    feature_cost,
    target_structure,
    predicted_structure,
    transport,
    alpha: float,
):
    """Evaluate the squared-loss FGW objective without a rank-four tensor."""

    feature_cost = np.asarray(feature_cost, dtype=np.float64)
    target_structure = np.asarray(target_structure, dtype=np.float64)
    predicted_structure = np.asarray(predicted_structure, dtype=np.float64)
    transport = np.asarray(transport, dtype=np.float64)
    if transport.shape != feature_cost.shape:
        raise ValueError("transport and feature cost must have the same shape")
    if target_structure.shape != (transport.shape[0], transport.shape[0]):
        raise ValueError("target structure has an incompatible shape")
    if predicted_structure.shape != (transport.shape[1], transport.shape[1]):
        raise ValueError("predicted structure has an incompatible shape")
    if not all(
        np.isfinite(value).all()
        for value in (
            feature_cost,
            target_structure,
            predicted_structure,
            transport,
        )
    ):
        raise ValueError("FGW objective inputs must be finite")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    row_mass = transport.sum(axis=1)
    column_mass = transport.sum(axis=0)
    feature = float(np.sum(feature_cost * transport))
    structural = float(
        row_mass @ np.square(target_structure) @ row_mass
        + column_mass @ np.square(predicted_structure) @ column_mass
        - 2.0
        * np.sum(
            (target_structure @ transport) * (transport @ predicted_structure)
        )
    )
    # Roundoff can produce a tiny negative value for this sum of squares.
    if structural < 0.0 and abs(structural) < 1e-12:
        structural = 0.0
    total = (1.0 - float(alpha)) * feature + float(alpha) * structural
    return {
        "feature": feature,
        "structural": structural,
        "weighted_total": float(total),
    }


@torch.no_grad()
def score_candidate_structures(
    tokens: torch.Tensor,
    relation_embed: nn.Module,
    candidate_indices,
    *,
    object_queries: int,
    relation_tokens: int,
    pair_chunk_size: int,
):
    """Return symmetric relation probabilities for each query candidate pool."""

    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    object_features = tokens[..., :object_queries, :]
    shared_relations = tokens[
        ..., object_queries : object_queries + relation_tokens, :
    ]
    structures = []
    for batch, raw_candidates in enumerate(candidate_indices):
        candidates = raw_candidates.to(tokens.device, dtype=torch.long)
        count = int(candidates.numel())
        structure = tokens.new_zeros((count, count))
        pairs = torch.combinations(
            torch.arange(count, device=tokens.device), r=2
        )
        selected_features = object_features[batch, candidates]
        for chunk in pairs.split(pair_chunk_size):
            left = selected_features[chunk[:, 0]]
            right = selected_features[chunk[:, 1]]
            if relation_tokens:
                relation = shared_relations[batch].reshape(1, -1).expand(
                    chunk.shape[0], -1
                )
                forward = torch.cat((left, right, relation), dim=-1)
                reverse = torch.cat((right, left, relation), dim=-1)
            else:
                forward = torch.cat((left, right), dim=-1)
                reverse = torch.cat((right, left), dim=-1)
            probabilities = 0.5 * (
                relation_embed(forward).softmax(-1)[:, 1]
                + relation_embed(reverse).softmax(-1)[:, 1]
            )
            structure[chunk[:, 0], chunk[:, 1]] = probabilities
            structure[chunk[:, 1], chunk[:, 0]] = probabilities
        structures.append(structure)
    return structures


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
    """Match nodes with partial FGW followed by one-to-one projection.

    Ground-truth nodes carry all transported mass while each prediction query
    has capacity for at most one target. This lets the coupling select a subset
    of surplus RelationFormer queries without allowing many-to-one matches.
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
        max_iter: int = 10_000,
        tolerance: float = 1e-7,
        random_state: int = 0,
        schedule: Mapping | None = None,
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
        self.target_structure_weight = float(structure_weight)
        self.schedule = dict(schedule or {})
        if self.schedule:
            if self.schedule.get("type") != "linear":
                raise ValueError("FGW schedule.type must be linear")
            start_epoch = self.schedule.get("start_epoch")
            ramp_epochs = self.schedule.get("ramp_epochs")
            if not isinstance(start_epoch, int) or start_epoch <= 0:
                raise ValueError("FGW schedule.start_epoch must be positive")
            if not isinstance(ramp_epochs, int) or ramp_epochs <= 0:
                raise ValueError("FGW schedule.ramp_epochs must be positive")
            initial_weight = float(self.schedule.get("initial_weight", 0.0))
            if not 0.0 <= initial_weight <= self.target_structure_weight:
                raise ValueError(
                    "FGW schedule.initial_weight must lie between 0 and the target"
                )
        self.structure_weight = self._scheduled_structure_weight(epoch=1)
        self.dimensions = int(dimensions)
        self.candidate_count = int(candidate_count)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        # Accepted for configuration compatibility. The unary initialization
        # and POT conditional-gradient path used here contain no random step.
        self.random_state = int(random_state)

    def _scheduled_structure_weight(self, epoch: int) -> float:
        """Return the effective FGW alpha for a global training epoch."""

        if not self.schedule:
            return self.target_structure_weight
        if self.schedule.get("type") != "linear":
            raise ValueError("FGW schedule.type must be linear")
        start_epoch = int(self.schedule["start_epoch"])
        ramp_epochs = int(self.schedule["ramp_epochs"])
        initial_weight = float(self.schedule.get("initial_weight", 0.0))
        if epoch < start_epoch:
            return initial_weight
        fraction = min(1.0, (int(epoch) - start_epoch + 1) / ramp_epochs)
        return initial_weight + fraction * (
            self.target_structure_weight - initial_weight
        )

    def set_training_progress(
        self, epoch: int, progress_percent: float = 0.0
    ) -> None:
        """Advance an epoch-based schedule; progress is accepted for API symmetry."""

        del progress_percent
        self.structure_weight = self._scheduled_structure_weight(
            max(1, int(epoch))
        )

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
                if len(selected) >= requested:
                    break
                if index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
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

    @staticmethod
    def validate_transport(transport, target_count, query_count, tolerance=1e-7):
        """Validate the partial-transport invariants and return a float array."""

        plan = np.asarray(transport, dtype=np.float64)
        if plan.shape != (target_count, query_count):
            raise RuntimeError("FGW solver returned an unexpected transport shape")
        if not np.isfinite(plan).all() or (plan < -tolerance).any():
            raise RuntimeError("FGW solver returned an invalid transport plan")
        represented_mass = float(
            np.full(target_count, 1.0 / target_count, dtype=np.float64).sum()
        )
        absolute_tolerance = max(float(tolerance), 1e-8)
        if not np.allclose(
            plan.sum(axis=1),
            1.0 / target_count,
            rtol=1e-5,
            atol=absolute_tolerance,
        ):
            raise RuntimeError("FGW transport violates the fixed target marginal")
        if (plan.sum(axis=0) > 1.0 / target_count + absolute_tolerance).any():
            raise RuntimeError("FGW transport violates prediction capacity")
        if not np.isclose(
            plan.sum(), represented_mass, rtol=1e-5, atol=absolute_tolerance
        ):
            raise RuntimeError("FGW transport violates total transported mass")
        return plan

    @staticmethod
    def initial_transport(feature_cost):
        """Return the feasible uniform-mass unary Hungarian initialization."""

        feature_cost = np.asarray(feature_cost, dtype=np.float64)
        if feature_cost.ndim != 2 or feature_cost.shape[0] == 0:
            raise ValueError("feature_cost must have at least one target row")
        target_count, query_count = feature_cost.shape
        if target_count > query_count:
            raise ValueError("feature_cost must have at least as many columns as rows")
        initial_target, initial_source = linear_sum_assignment(feature_cost)
        target_mass = np.full(target_count, 1.0 / target_count, dtype=np.float64)
        initial = np.zeros((target_count, query_count), dtype=np.float64)
        initial[initial_target, initial_source] = target_mass[initial_target]
        return initial

    def _solve_transport(
        self,
        feature_cost,
        target_structure,
        predicted_structure,
        *,
        return_diagnostics: bool = False,
    ):
        try:
            from ot.gromov import partial_fused_gromov_wasserstein
        except ImportError as error:
            raise ImportError(
                "The FGW matcher requires Python Optimal Transport (POT). "
                "Install the project's requirements before using matcher.type=fgw."
            ) from error

        target_count, query_count = feature_cost.shape
        target_mass = np.full(target_count, 1.0 / target_count, dtype=np.float64)
        # Partial OT interprets these as column capacities. Giving every query
        # the same capacity as one target prevents two targets from consuming
        # the same prediction while allowing surplus predictions to stay empty.
        query_capacity = np.full(query_count, 1.0 / target_count, dtype=np.float64)

        # The unary Hungarian plan is feasible for the partial problem and gives
        # the non-convex solver a stable, injective initialization.
        initial = self.initial_transport(feature_cost)
        # Use the represented sum rather than the literal 1.0. For node counts
        # such as 15, floating-point summation can make sum(target_mass) one ULP
        # smaller than 1, which POT otherwise rejects as an infeasible mass.
        transported_mass = float(target_mass.sum())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = partial_fused_gromov_wasserstein(
                feature_cost,
                target_structure,
                predicted_structure,
                p=target_mass,
                q=query_capacity,
                m=transported_mass,
                loss_fun="square_loss",
                symmetric=True,
                alpha=self.structure_weight,
                G0=initial,
                numItermax=self.max_iter,
                tol=self.tolerance,
                log=return_diagnostics,
                warn=True,
            )
        if not return_diagnostics:
            return result

        transport, solver_log = result
        history = [float(value) for value in solver_log.get("loss", ())]
        iterations = max(0, len(history) - 1)
        tolerance_met = None
        relative_change = None
        absolute_change = None
        if len(history) >= 2:
            absolute_change = abs(history[-1] - history[-2])
            relative_change = (
                absolute_change / abs(history[-1])
                if history[-1] != 0.0
                else None
            )
            tolerance_met = bool(
                absolute_change < self.tolerance
                or (
                    relative_change is not None
                    and relative_change < self.tolerance
                )
            )
        diagnostics = {
            "iterations": iterations,
            "iteration_limit": self.max_iter,
            "tolerance": self.tolerance,
            "tolerance_met": tolerance_met,
            "termination_evidence": (
                "tolerance_met"
                if tolerance_met
                else "iteration_limit"
                if iterations >= self.max_iter
                else "not_exposed_by_pot"
            ),
            "last_absolute_objective_change": absolute_change,
            "last_relative_objective_change": relative_change,
            "pot_objective_history": history,
            "pot_final_objective": (
                float(solver_log["partial_fgw_dist"])
                if "partial_fgw_dist" in solver_log
                else None
            ),
            "emd_result_code": solver_log.get("result_code"),
            "emd_warning": solver_log.get("warning"),
            "warnings": [str(item.message) for item in caught],
        }
        return transport, diagnostics

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping,
        targets: Mapping,
        *,
        predicted_structure=None,
        candidate_indices=None,
        return_transport: bool = False,
        return_diagnostics: bool = False,
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
        transports = []
        solver_diagnostics = []
        for sample in range(batch_size):
            truth = target_nodes[sample].to(predicted_nodes.device)
            target_count = int(truth.shape[0])
            if target_count == 0:
                assignments.append(_empty_assignment())
                transports.append(np.empty((0, 0), dtype=np.float64))
                solver_diagnostics.append(
                    {"iterations": 0, "termination_evidence": "empty_target"}
                )
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

            solve_args = (
                feature_cost.detach().double().cpu().numpy(),
                truth_structure.detach().double().cpu().numpy(),
                prediction_structure.detach().double().cpu().numpy(),
            )
            solve_started = time.perf_counter()
            solved = (
                self._solve_transport(*solve_args, return_diagnostics=True)
                if return_diagnostics
                else self._solve_transport(*solve_args)
            )
            solve_seconds = time.perf_counter() - solve_started
            if return_diagnostics:
                transport, diagnostics = solved
                diagnostics["solver_seconds"] = solve_seconds
                solver_diagnostics.append(diagnostics)
            else:
                transport = solved
            transport = self.validate_transport(
                transport, target_count, int(candidates.numel()), self.tolerance
            )
            local_source, matched_target = self.harden_transport(transport)
            assignments.append((candidates.cpu()[local_source], matched_target))
            transports.append(transport)
        if return_transport and return_diagnostics:
            return assignments, transports, solver_diagnostics
        if return_transport:
            return assignments, transports
        if return_diagnostics:
            return assignments, solver_diagnostics
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
            schedule=matcher.get("schedule"),
        )
    raise ValueError("matcher.type must be hungarian or fgw")


__all__ = [
    "FusedGromovWassersteinMatcher",
    "HungarianMatcher",
    "build_matcher",
    "fgw_objective_terms",
    "score_candidate_structures",
]
