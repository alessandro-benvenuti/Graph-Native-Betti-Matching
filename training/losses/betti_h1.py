"""Graph-native filtered H1 cycle-space matching for RelationFormer.

The complex is strictly one-dimensional: triangles are graph cycles and are
never filled.  Union-find identifies H1 birth edges and a spanning-forest path
gives one fundamental-cycle basis.  Sparse Gaussian elimination over F2 then
compares the *cycle spaces*, so matching does not depend on that arbitrary
basis.

Discrete topology is computed on detached probabilities.  The selected birth
edge indices are subsequently evaluated on the original tensors, giving the
usual piecewise-differentiable persistent-homology loss.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


Edge = Tuple[int, int]
_TOL = 1.0e-12


def _edge(u: int, v: int) -> Edge:
    u, v = int(u), int(v)
    if u == v:
        raise ValueError("Self-loops are not supported.")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class CycleClass:
    birth_edge: Edge
    birth_edge_index: int
    birth: float
    cycle_edges: Tuple[Edge, ...]


@dataclass(frozen=True)
class CycleMatch:
    prediction_index: int
    target_index: int
    shared_birth: float
    shared_cycle_edges: Tuple[Edge, ...]


@dataclass(frozen=True)
class CycleSpaceMatching:
    prediction_classes: Tuple[CycleClass, ...]
    target_classes: Tuple[CycleClass, ...]
    comparison_classes: Tuple[CycleClass, ...]
    matches: Tuple[CycleMatch, ...]
    unmatched_prediction_indices: Tuple[int, ...]
    unmatched_target_indices: Tuple[int, ...]
    union_only_cycles: Tuple[Tuple[Edge, ...], ...]

    @property
    def shared_rank(self) -> int:
        return len(self.matches)

    @property
    def false_prediction_rank(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def missed_target_rank(self) -> int:
        return len(self.unmatched_target_indices)

    @property
    def union_only_rank(self) -> int:
        return len(self.union_only_cycles)


class _UnionFind:
    def __init__(self, num_vertices: int):
        self.parent = list(range(num_vertices))

    def find(self, vertex: int) -> int:
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def union(self, left: int, right: int) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        # Deterministic structural root; H1 does not need the H0 elder rule.
        if root_left > root_right:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        return True


def _forest_path(
    adjacency: Mapping[int, Sequence[int]],
    start: int,
    goal: int,
) -> Tuple[int, ...]:
    queue = deque([start])
    predecessor = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbour in sorted(adjacency[current]):
            if neighbour not in predecessor:
                predecessor[neighbour] = current
                queue.append(neighbour)
    if goal not in predecessor:
        raise RuntimeError("Union-find and spanning forest became inconsistent.")
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessor[current]
    return tuple(reversed(path))


def _h1_basis(
    num_vertices: int,
    edge_order: Sequence[Edge],
    filtration_values: Sequence[float],
    *,
    terminal_value: float,
) -> Tuple[CycleClass, ...]:
    if len(edge_order) != len(filtration_values):
        raise ValueError("Edges and filtration values must have equal length.")
    union_find = _UnionFind(num_vertices)
    adjacency = {vertex: [] for vertex in range(num_vertices)}
    classes = []
    ordered_indices = sorted(
        range(len(edge_order)),
        key=lambda index: (
            float(filtration_values[index]),
            edge_order[index],
        ),
    )
    for edge_index in ordered_indices:
        edge = edge_order[edge_index]
        birth = float(filtration_values[edge_index])
        u, v = edge
        if union_find.union(u, v):
            adjacency[u].append(v)
            adjacency[v].append(u)
            continue
        path_vertices = _forest_path(adjacency, u, v)
        path_edges = tuple(
            _edge(left, right)
            for left, right in zip(path_vertices, path_vertices[1:])
        )
        # Classes born at the cap have zero lifetime in the truncated module.
        if terminal_value - birth > _TOL:
            classes.append(
                CycleClass(
                    birth_edge=edge,
                    birth_edge_index=edge_index,
                    birth=birth,
                    cycle_edges=path_edges + (edge,),
                )
            )
    return tuple(classes)


class _GF2Basis:
    """Sparse elimination with Python integers as edge bit-vectors."""

    def __init__(self) -> None:
        self.rows: Dict[int, Tuple[int, int]] = {}

    def reduce(self, vector: int, provenance: int = 0) -> Tuple[int, int]:
        while vector:
            pivot = vector.bit_length() - 1
            row = self.rows.get(pivot)
            if row is None:
                break
            vector ^= row[0]
            provenance ^= row[1]
        return vector, provenance

    def add(self, vector: int, provenance: int = 0) -> Tuple[bool, int]:
        reduced, reduced_provenance = self.reduce(vector, provenance)
        if reduced == 0:
            return False, reduced_provenance
        self.rows[reduced.bit_length() - 1] = reduced, reduced_provenance
        return True, reduced_provenance

    def coordinates(self, vector: int) -> int:
        reduced, coordinates = self.reduce(vector)
        if reduced:
            raise ValueError("Cycle is outside the represented cycle space.")
        return coordinates


def _bit_indices(bits: int) -> Tuple[int, ...]:
    indices = []
    while bits:
        lowest = bits & -bits
        indices.append(lowest.bit_length() - 1)
        bits ^= lowest
    return tuple(indices)


def _xor_selected(vectors: Sequence[int], selection: int) -> int:
    result = 0
    for index in _bit_indices(selection):
        result ^= vectors[index]
    return result


def _cycle_vectors(
    classes: Sequence[CycleClass],
    edge_to_index: Mapping[Edge, int],
) -> Tuple[int, ...]:
    vectors = []
    for item in classes:
        vector = 0
        for edge in item.cycle_edges:
            vector ^= 1 << edge_to_index[edge]
        vectors.append(vector)
    return tuple(vectors)


def _solver(vectors: Sequence[int]) -> _GF2Basis:
    result = _GF2Basis()
    for index, vector in enumerate(vectors):
        independent, _ = result.add(vector, 1 << index)
        if not independent:
            raise RuntimeError("Expected an independent cycle basis.")
    return result


def _intersection(left: Sequence[int], right: Sequence[int]) -> Tuple[int, ...]:
    joint = _GF2Basis()
    left_count = len(left)
    for index, vector in enumerate(left):
        independent, _ = joint.add(vector, 1 << index)
        if not independent:
            raise RuntimeError("Left cycle basis is dependent.")

    result = []
    for index, vector in enumerate(right):
        independent, relation = joint.add(
            vector,
            1 << (left_count + index),
        )
        if independent:
            continue
        left_selection = relation & ((1 << left_count) - 1)
        shared = _xor_selected(left, left_selection)
        if shared == 0:
            raise RuntimeError("Unexpected zero shared-cycle generator.")
        result.append(shared)
    return tuple(result)


def _shared_generators(
    prediction_classes: Sequence[CycleClass],
    target_classes: Sequence[CycleClass],
    prediction_vectors: Sequence[int],
    target_vectors: Sequence[int],
) -> Tuple[Tuple[float, int], ...]:
    thresholds = sorted(
        {item.birth for item in (*prediction_classes, *target_classes)}
    )
    accumulated = _GF2Basis()
    generators = []
    for threshold in thresholds:
        prediction_prefix = [
            vector
            for item, vector in zip(prediction_classes, prediction_vectors)
            if item.birth <= threshold + _TOL
        ]
        target_prefix = [
            vector
            for item, vector in zip(target_classes, target_vectors)
            if item.birth <= threshold + _TOL
        ]
        for vector in sorted(_intersection(prediction_prefix, target_prefix)):
            independent, _ = accumulated.add(vector)
            if independent:
                generators.append((threshold, vector))

    expected = len(_intersection(prediction_vectors, target_vectors))
    if len(generators) != expected:
        raise RuntimeError("Filtered intersection did not reach its final rank.")
    return tuple(generators)


def _source_pivots(
    shared_generators: Sequence[Tuple[float, int]],
    source_vectors: Sequence[int],
) -> Tuple[int, ...]:
    source_solver = _solver(source_vectors)
    reduced_columns = _GF2Basis()
    pivots = []
    for _, shared_vector in shared_generators:
        coordinates = source_solver.coordinates(shared_vector)
        reduced, _ = reduced_columns.reduce(coordinates)
        if reduced == 0:
            raise RuntimeError("Shared generator unexpectedly became dependent.")
        pivots.append(reduced.bit_length() - 1)
        independent, _ = reduced_columns.add(coordinates)
        if not independent:
            raise RuntimeError("Could not assign a unique H1 source pivot.")
    return tuple(pivots)


def _bits_to_edges(bits: int, edge_order: Sequence[Edge]) -> Tuple[Edge, ...]:
    return tuple(edge_order[index] for index in _bit_indices(bits))


def compute_cycle_space_matching(
    detached_edge_probabilities: Sequence[float],
    candidate_edges: Sequence[Edge],
    true_edges: Iterable[Edge],
    *,
    num_vertices: int,
    terminal_value: float = 1.0,
) -> CycleSpaceMatching:
    """Compute the detached graph H1 cycle-space matching."""
    edge_order = tuple(_edge(*edge) for edge in candidate_edges)
    if len(set(edge_order)) != len(edge_order):
        raise ValueError("candidate_edges contains duplicate undirected pairs.")
    edge_to_index = {edge: index for index, edge in enumerate(edge_order)}
    truth = {_edge(*edge) for edge in true_edges}
    if not truth <= set(edge_order):
        raise ValueError("Every true edge must be a candidate edge.")

    prediction_filtration = tuple(
        terminal_value * (1.0 - float(probability))
        for probability in detached_edge_probabilities
    )
    target_filtration = tuple(
        0.0 if edge in truth else terminal_value
        for edge in edge_order
    )
    comparison_filtration = tuple(
        min(prediction_value, target_value)
        for prediction_value, target_value
        in zip(prediction_filtration, target_filtration)
    )

    prediction_classes = _h1_basis(
        num_vertices,
        edge_order,
        prediction_filtration,
        terminal_value=terminal_value,
    )
    target_classes = _h1_basis(
        num_vertices,
        edge_order,
        target_filtration,
        terminal_value=terminal_value,
    )
    comparison_classes = _h1_basis(
        num_vertices,
        edge_order,
        comparison_filtration,
        terminal_value=terminal_value,
    )
    prediction_vectors = _cycle_vectors(prediction_classes, edge_to_index)
    target_vectors = _cycle_vectors(target_classes, edge_to_index)
    comparison_vectors = _cycle_vectors(comparison_classes, edge_to_index)

    # Verify both inclusion maps into the comparison cycle space.
    comparison_solver = _solver(comparison_vectors)
    for vector in (*prediction_vectors, *target_vectors):
        comparison_solver.coordinates(vector)

    shared = _shared_generators(
        prediction_classes,
        target_classes,
        prediction_vectors,
        target_vectors,
    )
    prediction_pivots = _source_pivots(shared, prediction_vectors)
    target_pivots = _source_pivots(shared, target_vectors)
    matches = tuple(
        CycleMatch(
            prediction_index=prediction_index,
            target_index=target_index,
            shared_birth=shared_birth,
            shared_cycle_edges=_bits_to_edges(shared_vector, edge_order),
        )
        for (shared_birth, shared_vector), prediction_index, target_index
        in zip(shared, prediction_pivots, target_pivots)
    )
    matched_prediction = set(prediction_pivots)
    matched_target = set(target_pivots)

    image_sum = _GF2Basis()
    for vector in (*prediction_vectors, *target_vectors):
        image_sum.add(vector)
    union_only = []
    for vector in comparison_vectors:
        independent, _ = image_sum.add(vector)
        if independent:
            union_only.append(_bits_to_edges(vector, edge_order))

    return CycleSpaceMatching(
        prediction_classes=prediction_classes,
        target_classes=target_classes,
        comparison_classes=comparison_classes,
        matches=matches,
        unmatched_prediction_indices=tuple(
            index
            for index in range(len(prediction_classes))
            if index not in matched_prediction
        ),
        unmatched_target_indices=tuple(
            index
            for index in range(len(target_classes))
            if index not in matched_target
        ),
        union_only_cycles=tuple(union_only),
    )


def cycle_space_matching_loss(
    edge_probabilities: torch.Tensor,
    candidate_edges: torch.Tensor,
    true_edges: torch.Tensor,
    *,
    num_vertices: int,
    terminal_value: float = 1.0,
    false_positive_weight: float = 1.0,
    false_negative_weight: float = 1.0,
    diagonal_factor: float = 0.5,
    normalize: bool = True,
) -> Tuple[torch.Tensor, CycleSpaceMatching]:
    """Return differentiable H1 loss and detached matching metadata."""
    if edge_probabilities.ndim != 1:
        raise ValueError("edge_probabilities must have shape [num_edges].")
    if candidate_edges.ndim != 2 or candidate_edges.shape[1] != 2:
        raise ValueError("candidate_edges must have shape [num_edges, 2].")
    if len(edge_probabilities) != len(candidate_edges):
        raise ValueError("candidate_edges and edge_probabilities must agree.")

    edge_tuples = tuple(
        _edge(*edge)
        for edge in candidate_edges.detach().cpu().tolist()
    )
    true_edge_tuples = tuple(
        _edge(*edge)
        for edge in true_edges.detach().cpu().reshape(-1, 2).tolist()
    )
    matching = compute_cycle_space_matching(
        edge_probabilities.detach().cpu().tolist(),
        edge_tuples,
        true_edge_tuples,
        num_vertices=num_vertices,
        terminal_value=terminal_value,
    )

    edge_filtration = terminal_value * (1.0 - edge_probabilities)
    loss = edge_probabilities.sum() * 0.0
    for match in matching.matches:
        prediction = matching.prediction_classes[match.prediction_index]
        target = matching.target_classes[match.target_index]
        loss = loss + (
            edge_filtration[prediction.birth_edge_index] - target.birth
        ).pow(2)
    for index in matching.unmatched_prediction_indices:
        item = matching.prediction_classes[index]
        persistence = terminal_value - edge_filtration[item.birth_edge_index]
        loss = (
            loss
            + diagonal_factor
            * false_positive_weight
            * persistence.pow(2)
        )

    missed_constant = sum(
        (
            terminal_value
            - matching.target_classes[index].birth
        )
        ** 2
        for index in matching.unmatched_target_indices
    )
    loss = loss + edge_probabilities.new_tensor(
        diagonal_factor * false_negative_weight * missed_constant
    )
    if normalize:
        feature_count = (
            matching.shared_rank
            + matching.false_prediction_rank
            + matching.missed_target_rank
        )
        loss = loss / max(1, feature_count)
    return loss, matching


__all__ = [
    "CycleClass",
    "CycleMatch",
    "CycleSpaceMatching",
    "compute_cycle_space_matching",
    "cycle_space_matching_loss",
]
