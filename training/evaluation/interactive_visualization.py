"""Parsing and graph helpers for portable SyntheticMRI visualizations.

Coordinates in this module are canonical normalized D/H/W until an explicit
call to :func:`normalized_dhw_to_plot_xyz` converts them for display.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from data.loaders.common import SamplePaths
from data.loaders.discovery import discover_synthetic_mri
from training.evaluation.metrics import canonical_edges
from training.evaluation.visualization import normalized_dhw_to_plot_xyz


@dataclass(frozen=True)
class FilteredPrediction:
    nodes_dhw: np.ndarray
    node_scores: np.ndarray
    edges: np.ndarray
    edge_scores: np.ndarray
    original_node_indices: np.ndarray


@dataclass(frozen=True)
class ErrorClassification:
    predicted_to_gt: Mapping[int, int]
    unmatched_predicted_nodes: tuple[int, ...]
    unmatched_gt_nodes: tuple[int, ...]
    matched_predicted_edges: tuple[tuple[int, int], ...]
    incident_to_unmatched_edges: tuple[tuple[int, int], ...]
    false_positive_edges: tuple[tuple[int, int], ...]
    missing_gt_edges: tuple[tuple[int, int], ...]


def load_prediction_records(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction export not found: {path}. Run evaluate.py with prediction "
            "export enabled before using this visualizer."
        )
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def resolve_prediction_record(
    records: Sequence[Mapping],
    *,
    sample_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
) -> Mapping:
    """Resolve source and evaluation identifiers without conflating them."""

    if bool(sample_id) == bool(evaluation_id):
        raise ValueError("Provide exactly one of sample_id or evaluation_id")
    field, wanted = (
        ("source_sample_id", sample_id)
        if sample_id
        else ("sample_id", evaluation_id)
    )
    matches = [record for record in records if str(record.get(field)) == wanted]
    if not matches:
        raise KeyError(f"No prediction has {field}={wanted!r}")
    if len(matches) != 1:
        raise ValueError(f"Prediction export has duplicate {field}={wanted!r}")
    record = matches[0]
    if not record.get("source_sample_id"):
        raise ValueError(
            f"Prediction {record.get('sample_id')!r} has no source_sample_id; "
            "it cannot be mapped safely to a dataset patch"
        )
    return record


def find_patch_triplet(
    dataset_root: Path | str, split: str, source_sample_id: str
) -> SamplePaths:
    matches = [
        record
        for record in discover_synthetic_mri(Path(dataset_root), split)
        if record.sample_id == source_sample_id
    ]
    if not matches:
        raise FileNotFoundError(
            f"No complete {split!r} patch triplet for source sample "
            f"{source_sample_id!r} below {dataset_root}"
        )
    if len(matches) != 1:
        raise ValueError(f"Duplicate patch triplet for {source_sample_id!r}")
    return matches[0]


def validate_graph_endpoints(edges, node_count: int, *, label: str = "graph") -> np.ndarray:
    array = np.asarray(edges, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    try:
        array = array.reshape(-1, 2)
    except ValueError as error:
        raise ValueError(f"{label} edges must contain endpoint pairs") from error
    invalid = np.logical_or(array < 0, array >= int(node_count))
    if invalid.any():
        row = int(np.flatnonzero(invalid.any(axis=1))[0])
        raise ValueError(
            f"{label} edge {array[row].tolist()} references a node outside "
            f"[0, {int(node_count)})"
        )
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError(f"{label} contains a self-loop")
    return array


def filter_prediction(
    record: Mapping, *, node_threshold: float = 0.0, edge_threshold: float = 0.0
) -> FilteredPrediction:
    nodes = np.asarray(record.get("nodes_dhw", []), dtype=np.float32).reshape(-1, 3)
    node_scores = np.asarray(record.get("node_scores", []), dtype=np.float32).reshape(-1)
    edges = validate_graph_endpoints(record.get("edges", []), len(nodes), label="prediction")
    edge_scores = np.asarray(record.get("edge_scores", []), dtype=np.float32).reshape(-1)
    if len(nodes) != len(node_scores):
        raise ValueError("Prediction must contain one node score per node")
    if len(edges) != len(edge_scores):
        raise ValueError("Prediction must contain one edge score per edge")
    if not 0.0 <= float(node_threshold) <= 1.0:
        raise ValueError("node_threshold must lie in [0, 1]")
    if not 0.0 <= float(edge_threshold) <= 1.0:
        raise ValueError("edge_threshold must lie in [0, 1]")

    kept_nodes = np.flatnonzero(node_scores >= float(node_threshold))
    old_to_new = np.full(len(nodes), -1, dtype=np.int64)
    old_to_new[kept_nodes] = np.arange(len(kept_nodes))
    edge_mask = edge_scores >= float(edge_threshold)
    if len(edges):
        edge_mask &= np.logical_and(
            old_to_new[edges[:, 0]] >= 0, old_to_new[edges[:, 1]] >= 0
        )
        filtered_edges = old_to_new[edges[edge_mask]]
    else:
        filtered_edges = np.empty((0, 2), dtype=np.int64)
    return FilteredPrediction(
        nodes_dhw=nodes[kept_nodes],
        node_scores=node_scores[kept_nodes],
        edges=filtered_edges,
        edge_scores=edge_scores[edge_mask],
        original_node_indices=kept_nodes,
    )


def load_patch_provenance(dataset_root: Path | str, sample_id: str) -> dict:
    path = Path(dataset_root) / "patch_index.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("sample_id") == sample_id]
    if len(matches) > 1:
        raise ValueError(f"patch_index.csv contains duplicate sample_id={sample_id!r}")
    return matches[0] if matches else {}


def classify_visualization_errors(
    predicted_nodes_dhw,
    predicted_edges,
    gt_nodes_dhw,
    gt_edges,
    *,
    max_distance: float = 0.1,
) -> ErrorClassification:
    """Classify likely graph errors using distance-gated one-to-one node matches.

    This is intentionally a visualization-time diagnostic, not the official
    IoU-based AP/F1 protocol or the training Hungarian assignment.
    """

    predicted = np.asarray(predicted_nodes_dhw, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(gt_nodes_dhw, dtype=np.float32).reshape(-1, 3)
    pred_edges = canonical_edges(
        validate_graph_endpoints(predicted_edges, len(predicted), label="prediction"),
        len(predicted),
    )
    target_edges = canonical_edges(
        validate_graph_endpoints(gt_edges, len(target), label="ground truth"),
        len(target),
    )
    if float(max_distance) < 0:
        raise ValueError("max_distance must be non-negative")

    mapping: dict[int, int] = {}
    if len(predicted) and len(target):
        from scipy.optimize import linear_sum_assignment

        distances = np.linalg.norm(predicted[:, None, :] - target[None, :, :], axis=2)
        pred_indices, target_indices = linear_sum_assignment(distances)
        mapping = {
            int(pred): int(gt)
            for pred, gt in zip(pred_indices, target_indices)
            if distances[pred, gt] <= float(max_distance)
        }

    target_edge_set = set(target_edges)
    matched_edges = []
    incident_edges = []
    false_positive_edges = []
    represented_target_edges = set()
    for edge in pred_edges:
        left, right = edge
        if left not in mapping or right not in mapping:
            incident_edges.append(edge)
            continue
        mapped = tuple(sorted((mapping[left], mapping[right])))
        if mapped in target_edge_set:
            matched_edges.append(edge)
            represented_target_edges.add(mapped)
        else:
            false_positive_edges.append(edge)
    return ErrorClassification(
        predicted_to_gt=mapping,
        unmatched_predicted_nodes=tuple(sorted(set(range(len(predicted))) - set(mapping))),
        unmatched_gt_nodes=tuple(sorted(set(range(len(target))) - set(mapping.values()))),
        matched_predicted_edges=tuple(matched_edges),
        incident_to_unmatched_edges=tuple(incident_edges),
        false_positive_edges=tuple(false_positive_edges),
        missing_gt_edges=tuple(sorted(target_edge_set - represented_target_edges)),
    )


__all__ = [
    "ErrorClassification",
    "FilteredPrediction",
    "classify_visualization_errors",
    "filter_prediction",
    "find_patch_triplet",
    "load_patch_provenance",
    "load_prediction_records",
    "normalized_dhw_to_plot_xyz",
    "resolve_prediction_record",
    "validate_graph_endpoints",
]
