"""Induced graph H0 matching for matched-only or node-aware filtrations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


Edge = Tuple[int, int]
_TOL = 1.0e-12


def _edge(u: int, v: int) -> Edge:
    u, v = int(u), int(v)
    if u == v:
        raise ValueError("Self-loops are not supported.")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class H0Pair:
    birth_vertex: int
    birth: float
    death_edge: Edge
    death_edge_index: int
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True)
class H0Match:
    prediction_index: int
    target_index: int


@dataclass(frozen=True)
class H0Matching:
    prediction_pairs: Tuple[H0Pair, ...]
    target_pairs: Tuple[H0Pair, ...]
    comparison_pairs: Tuple[H0Pair, ...]
    matches: Tuple[H0Match, ...]
    unmatched_prediction_indices: Tuple[int, ...]
    unmatched_target_indices: Tuple[int, ...]

    @property
    def matched_rank(self) -> int:
        return len(self.matches)

    @property
    def false_prediction_rank(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def missed_target_rank(self) -> int:
        return len(self.unmatched_target_indices)


@dataclass(frozen=True)
class _FilteredGraph:
    vertices: Tuple[int, ...]
    vertex_values: Mapping[int, float]
    edge_order: Tuple[Edge, ...]
    edge_values: Tuple[float, ...]

    def __post_init__(self):
        if len(self.edge_order) != len(self.edge_values):
            raise ValueError("edge_order and edge_values must agree.")
        if len(set(self.edge_order)) != len(self.edge_order):
            raise ValueError("Duplicate candidate edge.")
        if set(self.vertex_values) != set(self.vertices):
            raise ValueError("vertex_values must cover all vertices.")
        for edge, value in zip(self.edge_order, self.edge_values):
            if not isfinite(value):
                raise ValueError("Filtration values must be finite.")
            u, v = edge
            required = max(self.vertex_values[u], self.vertex_values[v])
            if value + _TOL < required:
                raise ValueError("An edge enters before one of its endpoints.")


class _UnionFind:
    def __init__(self, graph: _FilteredGraph):
        self.parent = {vertex: vertex for vertex in graph.vertices}
        self.birth_vertex = {vertex: vertex for vertex in graph.vertices}
        self.birth_values = graph.vertex_values

    def find(self, vertex: int) -> int:
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def _birth_key(self, root: int):
        vertex = self.birth_vertex[root]
        return self.birth_values[vertex], vertex

    def link(self, edge: Edge) -> Optional[int]:
        left, right = self.find(edge[0]), self.find(edge[1])
        if left == right:
            return None
        if self._birth_key(left) <= self._birth_key(right):
            older, younger = left, right
        else:
            older, younger = right, left
        killed = self.birth_vertex[younger]
        self.parent[younger] = older
        return killed


def _graph_from_probabilities(
    probabilities: Sequence[float],
    edge_order: Tuple[Edge, ...],
    *,
    num_vertices: int,
    terminal_value: float,
) -> _FilteredGraph:
    node_probabilities = {vertex: 0.0 for vertex in range(num_vertices)}
    for edge, probability in zip(edge_order, probabilities):
        u, v = edge
        probability = float(probability)
        node_probabilities[u] = max(node_probabilities[u], probability)
        node_probabilities[v] = max(node_probabilities[v], probability)
    vertex_values = {
        vertex: terminal_value * (1.0 - probability)
        for vertex, probability in node_probabilities.items()
    }
    edge_values = tuple(
        max(
            terminal_value * (1.0 - float(probability)),
            vertex_values[edge[0]],
            vertex_values[edge[1]],
        )
        for edge, probability in zip(edge_order, probabilities)
    )
    return _FilteredGraph(
        tuple(range(num_vertices)),
        vertex_values,
        edge_order,
        edge_values,
    )


def _graph_from_node_and_edge_probabilities(
    node_probabilities: Sequence[float],
    edge_probabilities: Sequence[float],
    edge_order: Tuple[Edge, ...],
    *,
    num_vertices: int,
    terminal_value: float,
) -> _FilteredGraph:
    if len(node_probabilities) != num_vertices:
        raise ValueError("node_probabilities must contain one value per vertex.")
    if len(edge_probabilities) != len(edge_order):
        raise ValueError("edge_probabilities must contain one value per edge.")
    vertex_values = {
        vertex: terminal_value * (1.0 - float(node_probabilities[vertex]))
        for vertex in range(num_vertices)
    }
    raw_edge_values = tuple(
        terminal_value * (1.0 - float(probability))
        for probability in edge_probabilities
    )
    edge_values = tuple(
        max(value, vertex_values[edge[0]], vertex_values[edge[1]])
        for edge, value in zip(edge_order, raw_edge_values)
    )
    return _FilteredGraph(
        tuple(range(num_vertices)),
        vertex_values,
        edge_order,
        edge_values,
    )


def _pairs(graph: _FilteredGraph) -> Tuple[H0Pair, ...]:
    union_find = _UnionFind(graph)
    result = []
    ordered_indices = sorted(
        range(len(graph.edge_order)),
        key=lambda index: (
            graph.edge_values[index],
            graph.edge_order[index],
        ),
    )
    for edge_index in ordered_indices:
        edge = graph.edge_order[edge_index]
        killed = union_find.link(edge)
        if killed is None:
            continue
        birth = graph.vertex_values[killed]
        death = graph.edge_values[edge_index]
        if death - birth > _TOL:
            result.append(
                H0Pair(
                    birth_vertex=killed,
                    birth=birth,
                    death_edge=edge,
                    death_edge_index=edge_index,
                    death=death,
                )
            )
    return tuple(result)


def _minimum_union(
    prediction: _FilteredGraph,
    target: _FilteredGraph,
) -> _FilteredGraph:
    return _FilteredGraph(
        prediction.vertices,
        {
            vertex: min(
                prediction.vertex_values[vertex],
                target.vertex_values[vertex],
            )
            for vertex in prediction.vertices
        },
        prediction.edge_order,
        tuple(
            min(prediction_value, target_value)
            for prediction_value, target_value
            in zip(prediction.edge_values, target.edge_values)
        ),
    )


def compute_h0_matching(
    detached_edge_probabilities: Sequence[float],
    candidate_edges: Sequence[Edge],
    true_edges: Iterable[Edge],
    *,
    num_vertices: int,
    terminal_value: float = 1.0,
    detached_node_probabilities: Optional[Sequence[float]] = None,
    target_node_presence: Optional[Sequence[float]] = None,
) -> H0Matching:
    """Compute extended union-induced H0 matching on detached values.

    Without node probabilities this preserves the original edge-derived
    vertex births. With them, vertices use ``1-q`` and target presence marks
    padded unmatched slots as absent at the terminal filtration value.
    """
    edge_order = tuple(_edge(*edge) for edge in candidate_edges)
    truth = {_edge(*edge) for edge in true_edges}
    if not truth <= set(edge_order):
        raise ValueError("Every true edge must be a candidate edge.")
    target_probabilities = tuple(
        1.0 if edge in truth else 0.0
        for edge in edge_order
    )
    if detached_node_probabilities is None:
        if target_node_presence is not None:
            raise ValueError(
                "target_node_presence requires detached_node_probabilities."
            )
        prediction = _graph_from_probabilities(
            detached_edge_probabilities,
            edge_order,
            num_vertices=num_vertices,
            terminal_value=terminal_value,
        )
        target = _graph_from_probabilities(
            target_probabilities,
            edge_order,
            num_vertices=num_vertices,
            terminal_value=terminal_value,
        )
    else:
        if target_node_presence is None:
            target_node_presence = (1.0,) * num_vertices
        prediction = _graph_from_node_and_edge_probabilities(
            detached_node_probabilities,
            detached_edge_probabilities,
            edge_order,
            num_vertices=num_vertices,
            terminal_value=terminal_value,
        )
        target = _graph_from_node_and_edge_probabilities(
            target_node_presence,
            target_probabilities,
            edge_order,
            num_vertices=num_vertices,
            terminal_value=terminal_value,
        )
    comparison = _minimum_union(prediction, target)
    prediction_pairs = _pairs(prediction)
    target_pairs = _pairs(target)
    prediction_by_birth = {
        pair.birth_vertex: index
        for index, pair in enumerate(prediction_pairs)
    }
    target_by_birth = {
        pair.birth_vertex: index
        for index, pair in enumerate(target_pairs)
    }
    prediction_uf = _UnionFind(prediction)
    target_uf = _UnionFind(target)
    comparison_uf = _UnionFind(comparison)
    comparison_pairs = []
    matches = []
    matched_prediction = set()
    matched_target = set()
    ordered_indices = sorted(
        range(len(edge_order)),
        key=lambda index: (
            comparison.edge_values[index],
            edge_order[index],
        ),
    )
    for edge_index in ordered_indices:
        edge = edge_order[edge_index]
        if comparison_uf.find(edge[0]) == comparison_uf.find(edge[1]):
            continue
        comparison_birth_vertex = comparison_uf.link(edge)
        prediction_birth_vertex = prediction_uf.link(edge)
        target_birth_vertex = target_uf.link(edge)
        if (
            comparison_birth_vertex is None
            or prediction_birth_vertex is None
            or target_birth_vertex is None
        ):
            raise RuntimeError("H0 comparison merge forests became inconsistent.")
        birth = comparison.vertex_values[comparison_birth_vertex]
        death = comparison.edge_values[edge_index]
        if death - birth <= _TOL:
            continue
        comparison_pairs.append(
            H0Pair(
                comparison_birth_vertex,
                birth,
                edge,
                edge_index,
                death,
            )
        )
        prediction_index = prediction_by_birth.get(prediction_birth_vertex)
        target_index = target_by_birth.get(target_birth_vertex)
        if prediction_index is not None and target_index is not None:
            matches.append(H0Match(prediction_index, target_index))
            matched_prediction.add(prediction_index)
            matched_target.add(target_index)

    return H0Matching(
        prediction_pairs=prediction_pairs,
        target_pairs=target_pairs,
        comparison_pairs=tuple(comparison_pairs),
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(
            index
            for index in range(len(prediction_pairs))
            if index not in matched_prediction
        ),
        unmatched_target_indices=tuple(
            index
            for index in range(len(target_pairs))
            if index not in matched_target
        ),
    )


def h0_betti_matching_loss(
    edge_probabilities: torch.Tensor,
    candidate_edges: torch.Tensor,
    true_edges: torch.Tensor,
    *,
    num_vertices: int,
    terminal_value: float = 1.0,
    unmatched_weight: float = 1.0,
    unmatched_node_weight: float = 0.0,
    diagonal_factor: float = 0.5,
    normalize: bool = True,
    normalization: str = "feature_count",
    node_probabilities: Optional[torch.Tensor] = None,
    target_node_presence: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, H0Matching]:
    """Evaluate induced H0 matching on the original probability tensors."""
    edge_order = tuple(
        _edge(*edge)
        for edge in candidate_edges.detach().cpu().tolist()
    )
    truth = tuple(
        _edge(*edge)
        for edge in true_edges.detach().cpu().reshape(-1, 2).tolist()
    )
    matching = compute_h0_matching(
        edge_probabilities.detach().cpu().tolist(),
        edge_order,
        truth,
        num_vertices=num_vertices,
        terminal_value=terminal_value,
        detached_node_probabilities=(
            None
            if node_probabilities is None
            else node_probabilities.detach().cpu().tolist()
        ),
        target_node_presence=(
            None
            if target_node_presence is None
            else target_node_presence.detach().cpu().tolist()
        ),
    )
    edge_filtration = terminal_value * (1.0 - edge_probabilities)
    node_filtration: Dict[int, torch.Tensor] = {}
    if node_probabilities is None:
        for vertex in range(num_vertices):
            incident = [
                edge_probabilities[index]
                for index, edge in enumerate(edge_order)
                if vertex in edge
            ]
            node_filtration[vertex] = (
                terminal_value * (1.0 - torch.stack(incident).max())
                if incident
                else edge_probabilities.new_tensor(terminal_value)
            )
    else:
        if node_probabilities.ndim != 1 or len(node_probabilities) != num_vertices:
            raise ValueError(
                "node_probabilities must have shape [num_vertices]."
            )
        node_probabilities = node_probabilities.to(edge_probabilities.device)
        node_filtration = {
            vertex: terminal_value * (1.0 - node_probabilities[vertex])
            for vertex in range(num_vertices)
        }

    if normalization not in {"feature_count", "matched_mean"}:
        raise ValueError(
            "normalization must be 'feature_count' or 'matched_mean'."
        )
    zero = edge_probabilities.sum() * 0.0
    matched_terms = []
    for match in matching.matches:
        prediction = matching.prediction_pairs[match.prediction_index]
        target = matching.target_pairs[match.target_index]
        matched_terms.append(
            (node_filtration[prediction.birth_vertex] - target.birth).pow(2)
            + (edge_filtration[prediction.death_edge_index] - target.death).pow(2)
        )
    false_terms = []
    for index in matching.unmatched_prediction_indices:
        pair = matching.prediction_pairs[index]
        persistence = (
            edge_filtration[pair.death_edge_index]
            - node_filtration[pair.birth_vertex]
        )
        false_terms.append(
            unmatched_weight * diagonal_factor * persistence.pow(2)
        )
    missed_terms = [
        edge_probabilities.new_tensor(
            unmatched_weight
            * diagonal_factor
            * matching.target_pairs[index].persistence**2
        )
        for index in matching.unmatched_target_indices
    ]
    node_terms = []
    if (
        node_probabilities is not None
        and target_node_presence is not None
        and unmatched_node_weight > 0.0
    ):
        target_presence = target_node_presence.to(edge_probabilities.device)
        target_filtration = terminal_value * (1.0 - target_presence)
        for vertex in torch.nonzero(
            target_presence.detach() < 0.5, as_tuple=False
        ).flatten().tolist():
            node_terms.append(
                unmatched_node_weight
                * (node_filtration[vertex] - target_filtration[vertex]).pow(2)
            )

    groups = (matched_terms, false_terms, missed_terms, node_terms)
    if not normalize:
        loss = sum((sum(terms, zero) for terms in groups), zero)
    elif normalization == "feature_count":
        count = sum(len(terms) for terms in groups)
        loss = sum((sum(terms, zero) for terms in groups), zero) / max(1, count)
    else:
        # Average only correspondence terms, whose count scales with target
        # graph size.  False, missed, and absent-node penalties remain sums:
        # adding an error can therefore never dilute existing supervision.
        matched_loss = (
            sum(matched_terms, zero) / len(matched_terms)
            if matched_terms
            else zero
        )
        loss = matched_loss + sum(
            (sum(terms, zero) for terms in groups[1:]), zero
        )
    return loss, matching


__all__ = [
    "H0Match",
    "H0Matching",
    "H0Pair",
    "compute_h0_matching",
    "h0_betti_matching_loss",
]
