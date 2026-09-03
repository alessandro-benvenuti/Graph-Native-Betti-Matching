#!/usr/bin/env python3
"""Audit SyntheticMRI source volumes and candidate patches without writing data.

The audit has two grid modes:

``full``
    A deterministic, boundary-complete grid suitable for deployment-style
    validation and testing.

``legacy``
    The inherited random-offset grid and cyclic random starting position from
    ``Vascular-Graph-Extraction/3d/data/generate_synth_data.py``.  The audit
    enumerates the whole legacy candidate grid; it does not stop at a patch
    quota, because the exact historical invocation is not recorded here.

Ground-truth segmentation and graph data are used only to describe candidates
and to reproduce the inherited rejection decision.  They never decide which
candidate rows are included in the audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy.stats import median_abs_deviation


PATCH_SIZE = (64, 64, 64)
PAD = (5, 5, 5)
DEFAULT_OVERLAP = 0.25
DEFAULT_SEED = 42
SPLITS = ("train", "val", "test")


def _natural_id_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def complete_axis_starts(axis_size: int, crop_size: int, stride: int) -> list[int]:
    """Return starts covering an axis from zero through its final boundary."""

    if axis_size < crop_size:
        raise ValueError(
            f"Axis size {axis_size} is smaller than crop size {crop_size}"
        )
    if stride <= 0:
        raise ValueError("stride must be positive")
    maximum = axis_size - crop_size
    starts = list(range(0, maximum + 1, stride))
    if starts[-1] != maximum:
        starts.append(maximum)
    return starts


def complete_grid_positions(
    shape: Sequence[int], crop_size: Sequence[int], stride: Sequence[int]
) -> list[tuple[int, int, int]]:
    axes = [
        complete_axis_starts(int(size), int(crop), int(step))
        for size, crop, step in zip(shape, crop_size, stride)
    ]
    return [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]


def legacy_grid_positions(
    shape: Sequence[int],
    crop_size: Sequence[int],
    stride: Sequence[int],
    *,
    patient_id: str,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Reproduce the inherited random-offset grid and cyclic start."""

    try:
        numeric_id = int(patient_id)
    except ValueError as error:
        raise ValueError(
            "Legacy grid reproduction requires integer patient IDs; "
            f"received {patient_id!r}"
        ) from error

    rng_seed = hash((int(seed), numeric_id)) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)
    offsets = [int(rng.integers(0, int(step))) for step in stride]
    axes = []
    for size, crop, step, offset in zip(shape, crop_size, stride, offsets):
        maximum = int(size) - int(crop)
        if maximum < 0:
            raise ValueError(
                f"Axis size {size} is smaller than crop size {crop}"
            )
        axes.append(list(range(offset, maximum + 1, int(step))))
    positions = [(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]]
    if not positions:
        positions = [(0, 0, 0)]
    start = int(rng.integers(0, len(positions)))
    return positions[start:] + positions[:start]


def _parse_int(value: object) -> int:
    return int(float(str(value)))


def _parse_position(value: Mapping[str, object]) -> np.ndarray:
    return np.asarray(
        [float(value["pos_x"]), float(value["pos_y"]), float(value["pos_z"])],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CenterlineEdge:
    node1: int
    node2: int
    positions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CroppedGraph:
    positions: np.ndarray
    edges: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(len(self.edges))


class SourceGraph:
    """Count the graph produced by the inherited centerline crop policy."""

    def __init__(
        self,
        nodes: Mapping[int, np.ndarray],
        edges: Sequence[tuple[int, int]],
        centerlines: Sequence[CenterlineEdge],
    ) -> None:
        self.nodes = dict(nodes)
        self.edges = tuple(edges)
        self.centerlines = tuple(centerlines)
        self.node_ids = np.asarray(list(self.nodes), dtype=np.int64)
        self.node_positions = np.asarray(
            [self.nodes[node] for node in self.node_ids], dtype=np.float64
        ).reshape(-1, 3)
        self.edge_geometries = tuple(self._edge_geometries())
        self.edge_polylines = tuple(
            self._oriented_polyline(edge) for edge in self.edge_geometries
        )
        self.edge_minima = np.asarray(
            [polyline.min(axis=0) for polyline in self.edge_polylines],
            dtype=np.float64,
        ).reshape(-1, 3)
        self.edge_maxima = np.asarray(
            [polyline.max(axis=0) for polyline in self.edge_polylines],
            dtype=np.float64,
        ).reshape(-1, 3)

    @classmethod
    def from_directory(cls, directory: Path) -> "SourceGraph":
        nodes_path = directory / "nodes.csv"
        edges_path = directory / "edges.csv"
        vvg_path = directory / "graph.vvg"
        missing = [path.name for path in (nodes_path, edges_path, vvg_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing graph files below {directory}: {', '.join(missing)}"
            )

        with nodes_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            nodes = {_parse_int(row["id"]): _parse_position(row) for row in reader}

        with edges_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            edges = [
                (_parse_int(row["node1id"]), _parse_int(row["node2id"]))
                for row in reader
            ]

        opener = gzip.open if vvg_path.suffix == ".gz" else open
        with opener(vvg_path, "rt") as handle:
            payload = json.load(handle)
        centerlines = []
        for edge in payload["graph"]["edges"]:
            positions = tuple(
                np.asarray(item["pos"], dtype=np.float64)
                for item in edge.get("skeletonVoxels", ())
            )
            centerlines.append(
                CenterlineEdge(
                    node1=_parse_int(edge["node1"]),
                    node2=_parse_int(edge["node2"]),
                    positions=positions,
                )
            )
        return cls(nodes, edges, centerlines)

    @staticmethod
    def _inside(position: np.ndarray, bounds: np.ndarray) -> bool:
        return bool(np.all(position >= bounds[:, 0]) and np.all(position <= bounds[:, 1]))

    @classmethod
    def _first_exit(
        cls, positions: Sequence[np.ndarray], bounds: np.ndarray
    ) -> np.ndarray | None:
        if len(positions) < 2:
            return None
        previous = positions[0]
        previous_inside = cls._inside(previous, bounds)
        for position in positions[1:]:
            inside = cls._inside(position, bounds)
            if not inside and previous_inside:
                return previous
            previous = position
            previous_inside = inside
        return None

    @classmethod
    def _inside_segments(
        cls, positions: Sequence[np.ndarray], bounds: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not positions:
            return []
        previous = positions[0]
        previous_inside = cls._inside(previous, bounds)
        active = previous_inside
        start = previous if active else None
        segments = []
        for position in positions[1:]:
            inside = cls._inside(position, bounds)
            if inside and not previous_inside:
                active = True
                start = position
            if not inside and previous_inside and active:
                segments.append((start, previous))
                active = False
                start = None
            previous = position
            previous_inside = inside
        if active and start is not None:
            segments.append((start, previous))
        return segments

    def crop_inherited(self, bounds: np.ndarray) -> CroppedGraph:
        """Reproduce the historical sample-based crop policy for auditing."""

        if bounds.shape != (3, 2):
            raise ValueError(f"Expected bounds [3,2], received {bounds.shape}")
        if len(self.node_ids):
            inside_mask = np.logical_and(
                self.node_positions >= bounds[:, 0],
                self.node_positions <= bounds[:, 1],
            ).all(axis=1)
            kept_ids = set(self.node_ids[inside_mask].tolist())
            positions = [position.copy() for position in self.node_positions[inside_mask]]
            kept_indices = {
                int(node_id): index
                for index, node_id in enumerate(self.node_ids[inside_mask].tolist())
            }
        else:
            kept_ids = set()
            positions = []
            kept_indices = {}

        edges = [
            (kept_indices[left], kept_indices[right])
            for left, right in self.edges
            if left in kept_ids and right in kept_ids
        ]

        candidate_indices = set(self._candidate_edge_indices(bounds).tolist())
        for edge_index, edge in enumerate(self.centerlines):
            if edge_index not in candidate_indices:
                continue
            node1_inside = edge.node1 in kept_ids
            node2_inside = edge.node2 in kept_ids
            if node1_inside and not node2_inside:
                boundary = self._first_exit(edge.positions, bounds)
                if boundary is not None:
                    boundary_index = len(positions)
                    positions.append(boundary.copy())
                    edges.append((boundary_index, kept_indices[edge.node1]))
            elif node2_inside and not node1_inside:
                boundary = self._first_exit(tuple(reversed(edge.positions)), bounds)
                if boundary is not None:
                    boundary_index = len(positions)
                    positions.append(boundary.copy())
                    edges.append((boundary_index, kept_indices[edge.node2]))
            elif not node1_inside and not node2_inside:
                for start, end in self._inside_segments(edge.positions, bounds):
                    start_index = len(positions)
                    positions.extend((start.copy(), end.copy()))
                    edges.append((start_index, start_index + 1))

        array = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        return CroppedGraph(positions=array, edges=edge_array)

    @staticmethod
    def _clip_segment(
        start: np.ndarray,
        end: np.ndarray,
        bounds: np.ndarray,
        tolerance: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the exact portion of a segment inside an axis-aligned box."""

        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        direction = end - start
        lower_t, upper_t = 0.0, 1.0
        for axis in range(3):
            if abs(float(direction[axis])) <= tolerance:
                if start[axis] < bounds[axis, 0] or start[axis] > bounds[axis, 1]:
                    return None
                continue
            first = (bounds[axis, 0] - start[axis]) / direction[axis]
            second = (bounds[axis, 1] - start[axis]) / direction[axis]
            entering, leaving = sorted((float(first), float(second)))
            lower_t = max(lower_t, entering)
            upper_t = min(upper_t, leaving)
            if lower_t > upper_t + tolerance:
                return None
        return start + lower_t * direction, start + upper_t * direction

    def _oriented_polyline(self, edge: CenterlineEdge) -> np.ndarray:
        """Return node1 -> centerline samples -> node2, correcting sample order."""

        node1 = np.asarray(self.nodes[edge.node1], dtype=np.float64)
        node2 = np.asarray(self.nodes[edge.node2], dtype=np.float64)
        samples = np.asarray(edge.positions, dtype=np.float64).reshape(-1, 3)
        if len(samples):
            forward_cost = np.linalg.norm(samples[0] - node1) + np.linalg.norm(
                samples[-1] - node2
            )
            reverse_cost = np.linalg.norm(samples[0] - node2) + np.linalg.norm(
                samples[-1] - node1
            )
            if reverse_cost < forward_cost:
                samples = samples[::-1]
        points = [node1, *samples, node2]
        deduplicated = [points[0]]
        for point in points[1:]:
            if not np.allclose(point, deduplicated[-1], atol=1e-9, rtol=0.0):
                deduplicated.append(point)
        return np.asarray(deduplicated, dtype=np.float64)

    @classmethod
    def _clip_polyline(
        cls, points: np.ndarray, bounds: np.ndarray, tolerance: float = 1e-8
    ) -> list[np.ndarray]:
        """Return connected, positive-length portions of a polyline in a box."""

        components: list[list[np.ndarray]] = []
        active: list[np.ndarray] = []
        for start, end in zip(points[:-1], points[1:]):
            clipped = cls._clip_segment(start, end, bounds)
            if clipped is None:
                if active:
                    components.append(active)
                    active = []
                continue
            clipped_start, clipped_end = clipped
            if np.linalg.norm(clipped_end - clipped_start) <= tolerance:
                continue
            if not active:
                active = [clipped_start, clipped_end]
            elif np.allclose(active[-1], clipped_start, atol=tolerance, rtol=0.0):
                if not np.allclose(active[-1], clipped_end, atol=tolerance, rtol=0.0):
                    active.append(clipped_end)
            else:
                components.append(active)
                active = [clipped_start, clipped_end]
        if active:
            components.append(active)
        return [np.asarray(component) for component in components]

    def _edge_geometries(self) -> list[CenterlineEdge]:
        """Require a one-to-one correspondence between VVG and CSV edges."""

        remaining: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for left, right in self.edges:
            remaining.setdefault(tuple(sorted((left, right))), []).append((left, right))
        result = []
        for centerline in self.centerlines:
            key = tuple(sorted((centerline.node1, centerline.node2)))
            matches = remaining.get(key)
            if not matches:
                raise ValueError(
                    "VVG centerline has no matching CSV edge: "
                    f"({centerline.node1}, {centerline.node2})"
                )
            matches.pop()
            result.append(centerline)
        unmatched = [edge for matches in remaining.values() for edge in matches]
        if unmatched:
            raise ValueError(
                f"CSV edges have no matching VVG centerline ({len(unmatched)} total): "
                f"{unmatched[:5]}"
            )
        return result

    def _candidate_edge_indices(self, bounds: np.ndarray) -> np.ndarray:
        if not len(self.edge_polylines):
            return np.empty((0,), dtype=np.int64)
        overlap = np.logical_and(
            self.edge_maxima >= bounds[:, 0], self.edge_minima <= bounds[:, 1]
        ).all(axis=1)
        return np.flatnonzero(overlap)

    def crop(self, bounds: np.ndarray) -> CroppedGraph:
        """Clip complete endpoint-aware edge polylines exactly to ``bounds``."""

        if bounds.shape != (3, 2):
            raise ValueError(f"Expected bounds [3,2], received {bounds.shape}")
        if len(self.node_ids):
            inside_mask = np.logical_and(
                self.node_positions >= bounds[:, 0],
                self.node_positions <= bounds[:, 1],
            ).all(axis=1)
            inside_ids = self.node_ids[inside_mask].tolist()
            positions = [position.copy() for position in self.node_positions[inside_mask]]
            kept_indices = {
                int(node_id): index for index, node_id in enumerate(inside_ids)
            }
        else:
            positions = []
            kept_indices = {}

        def component_endpoint_index(
            point: np.ndarray, source_edge: CenterlineEdge
        ) -> int:
            for node_id in (source_edge.node1, source_edge.node2):
                if node_id in kept_indices and np.allclose(
                    point, self.nodes[node_id], atol=1e-7, rtol=0.0
                ):
                    return kept_indices[node_id]
            index = len(positions)
            positions.append(point.copy())
            return index

        edges = []
        for edge_index in self._candidate_edge_indices(bounds):
            source_edge = self.edge_geometries[int(edge_index)]
            polyline = self.edge_polylines[int(edge_index)]
            for component in self._clip_polyline(polyline, bounds):
                left = component_endpoint_index(component[0], source_edge)
                right = component_endpoint_index(component[-1], source_edge)
                if left != right:
                    edges.append((left, right))

        array = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        return CroppedGraph(positions=array, edges=edge_array)


def _stem_id(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported NIfTI name: {path}")


def discover_sources(root: Path) -> dict[str, tuple[Path, Path, Path]]:
    raw_paths = sorted((root / "raw").glob("*.nii*"))
    seg_paths = sorted((root / "seg").glob("*.nii*"))
    raw = {_stem_id(path): path for path in raw_paths}
    seg = {_stem_id(path): path for path in seg_paths}
    graphs = {
        path.name: path
        for path in (root / "graphs").iterdir()
        if path.is_dir()
    }
    union = set(raw) | set(seg) | set(graphs)
    mismatches = []
    for patient_id in sorted(union, key=_natural_id_key):
        missing = [
            name
            for name, mapping in (("raw", raw), ("seg", seg), ("graphs", graphs))
            if patient_id not in mapping
        ]
        if missing:
            mismatches.append(f"{patient_id}: {','.join(missing)}")
    if mismatches:
        raise ValueError("Source ID mismatch: " + "; ".join(mismatches))
    if not union:
        raise ValueError(f"No source volumes found below {root}")
    return {
        patient_id: (raw[patient_id], seg[patient_id], graphs[patient_id])
        for patient_id in sorted(union, key=_natural_id_key)
    }


def read_splits(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "split"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain patient_id and split columns")
        result = {}
        for row in reader:
            patient_id = str(row["patient_id"]).strip()
            split = str(row["split"]).strip().lower()
            if split not in SPLITS:
                raise ValueError(f"Unsupported split {split!r} for patient {patient_id}")
            if patient_id in result:
                raise ValueError(f"Duplicate patient ID in split CSV: {patient_id}")
            result[patient_id] = split
    return result


def voxel_to_world(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (homogeneous @ affine.T)[:, :3]


def world_to_voxel(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (homogeneous @ np.linalg.inv(affine).T)[:, :3]


def patch_world_bounds(
    start: Sequence[int], crop_size: Sequence[int], affine: np.ndarray
) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = start_array + np.asarray(crop_size, dtype=np.float64) - 1.0
    corners = voxel_to_world(np.stack((start_array, end_array)), affine)
    return np.sort(corners.T, axis=1)


def coordinate_range(
    positions: np.ndarray,
    affine: np.ndarray,
    start: Sequence[int],
    pad: Sequence[int],
    patch_size: Sequence[int],
) -> tuple[float | None, float | None, bool]:
    if not len(positions):
        return None, None, True
    voxel = world_to_voxel(positions, affine)
    local = voxel - np.asarray(start) + np.asarray(pad)
    normalized = local / np.asarray(patch_size)
    minimum = float(normalized.min())
    maximum = float(normalized.max())
    return minimum, maximum, minimum >= -0.01 and maximum <= 1.01


def normalize_like_legacy(image: np.ndarray) -> tuple[np.ndarray, float]:
    integer_image = np.array(image, dtype=np.int32, copy=True)
    threshold = float(
        median_abs_deviation(integer_image.reshape(-1), scale="normal") * 4.0
        + np.median(integer_image)
    )
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"Legacy normalization threshold is invalid: {threshold}")
    integer_image[integer_image > threshold] = threshold
    return integer_image.astype(np.float32) / threshold, threshold


def inherited_rejection_reason(
    *,
    foreground_count: int,
    background_count: int,
    snr: float | None,
    foreground_ratio: float,
    segmentation_sum: float,
    node_count: int,
    coordinate_valid: bool,
) -> str:
    if foreground_count == 0:
        return "no_foreground"
    if background_count == 0:
        return "no_background"
    if snr is None or not math.isfinite(snr) or snr < 1.2:
        return "snr_below_1.2"
    if foreground_ratio > 0.5:
        return "foreground_ratio_above_0.5"
    if segmentation_sum <= 10:
        return "segmentation_sum_at_most_10"
    if node_count < 3:
        return "fewer_than_3_nodes"
    if not coordinate_valid:
        return "coordinates_out_of_bounds"
    return "accepted"


SUMMARY_FLAGS = (
    "segmentation_empty",
    "graph_empty",
    "under_3_nodes",
    "foreground_graph_empty",
    "graph_segmentation_empty",
    "old_accepted",
)


def _empty_summary() -> dict[str, int]:
    return {"candidates": 0, **{name: 0 for name in SUMMARY_FLAGS}}


def update_summary(summary: dict[str, int], row: Mapping[str, object]) -> None:
    summary["candidates"] += 1
    for name in SUMMARY_FLAGS:
        summary[name] += int(bool(row[name]))
    reason = str(row["old_rejection_reason"])
    key = f"reason:{reason}"
    summary[key] = summary.get(key, 0) + 1


CANDIDATE_FIELDS = (
    "patient_id",
    "split",
    "grid",
    "start_d",
    "start_h",
    "start_w",
    "foreground_voxels",
    "segmentation_sum",
    "foreground_ratio",
    "node_count",
    "edge_count",
    "segmentation_empty",
    "graph_empty",
    "under_3_nodes",
    "foreground_graph_empty",
    "graph_segmentation_empty",
    "image_mean",
    "image_std",
    "image_max",
    "gt_snr",
    "coordinate_min",
    "coordinate_max",
    "coordinate_valid",
    "old_accepted",
    "old_rejection_reason",
)


INVENTORY_FIELDS = (
    "patient_id",
    "split",
    "shape_d",
    "shape_h",
    "shape_w",
    "spacing_d",
    "spacing_h",
    "spacing_w",
    "raw_dtype",
    "seg_dtype",
    "seg_affine_matches_raw",
    "affine_determinant",
    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_std",
    "segmentation_foreground_voxels",
    "segmentation_foreground_ratio",
    "full_graph_nodes",
    "full_graph_edges",
    "legacy_normalization_threshold",
)


def _candidate_row(
    *,
    patient_id: str,
    split: str,
    grid: str,
    start: tuple[int, int, int],
    crop_size: Sequence[int],
    patch_size: Sequence[int],
    pad: Sequence[int],
    image: np.ndarray,
    segmentation: np.ndarray,
    graph: SourceGraph,
    affine: np.ndarray,
) -> dict[str, object]:
    slices = tuple(slice(origin, origin + size) for origin, size in zip(start, crop_size))
    image_patch = image[slices]
    segmentation_patch = segmentation[slices]
    foreground_mask = segmentation_patch > 0
    background_mask = segmentation_patch == 0
    foreground_count = int(foreground_mask.sum())
    core_background_count = int(background_mask.sum())
    padded_voxels = int(np.prod(patch_size) - np.prod(crop_size))
    background_count = core_background_count + padded_voxels
    segmentation_sum = float(segmentation_patch.sum(dtype=np.float64))
    foreground_ratio = foreground_count / float(np.prod(patch_size))

    if foreground_count and background_count:
        foreground = image_patch[foreground_mask].astype(np.float64, copy=False)
        core_background = image_patch[background_mask].astype(np.float64, copy=False)
        background_sum = float(core_background.sum())
        background_square_sum = float(np.square(core_background).sum())
        background_mean = background_sum / background_count
        background_variance = max(
            0.0, background_square_sum / background_count - background_mean**2
        )
        snr = float(
            (float(foreground.mean()) - background_mean)
            / (math.sqrt(background_variance) + 1e-8)
        )
    else:
        snr = None

    bounds = patch_world_bounds(start, crop_size, affine)
    # This report intentionally reproduces the historical selection policy.
    cropped = graph.crop_inherited(bounds)
    coordinate_min, coordinate_max, coordinate_valid = coordinate_range(
        cropped.positions, affine, start, pad, patch_size
    )
    node_count = int(len(cropped.positions))
    edge_count = int(cropped.edge_count)
    reason = inherited_rejection_reason(
        foreground_count=foreground_count,
        background_count=background_count,
        snr=snr,
        foreground_ratio=foreground_ratio,
        segmentation_sum=segmentation_sum,
        node_count=node_count,
        coordinate_valid=coordinate_valid,
    )
    segmentation_empty = foreground_count == 0
    graph_empty = node_count == 0 and edge_count == 0
    return {
        "patient_id": patient_id,
        "split": split,
        "grid": grid,
        "start_d": start[0],
        "start_h": start[1],
        "start_w": start[2],
        "foreground_voxels": foreground_count,
        "segmentation_sum": segmentation_sum,
        "foreground_ratio": foreground_ratio,
        "node_count": node_count,
        "edge_count": edge_count,
        "segmentation_empty": segmentation_empty,
        "graph_empty": graph_empty,
        "under_3_nodes": node_count < 3,
        "foreground_graph_empty": foreground_count > 0 and graph_empty,
        "graph_segmentation_empty": node_count > 0 and segmentation_empty,
        "image_mean": float(image_patch.mean()),
        "image_std": float(image_patch.std()),
        "image_max": float(image_patch.max()),
        "gt_snr": snr,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "coordinate_valid": coordinate_valid,
        "old_accepted": reason == "accepted",
        "old_rejection_reason": reason,
    }


def _write_summary_csv(
    path: Path, summaries: Mapping[tuple[str, str], Mapping[str, int]]
) -> None:
    reason_names = sorted(
        {
            key.removeprefix("reason:")
            for summary in summaries.values()
            for key in summary
            if key.startswith("reason:")
        }
    )
    count_fields = ["candidates", *SUMMARY_FLAGS, *[f"reason_{name}" for name in reason_names]]
    percentage_fields = [f"{field}_pct" for field in SUMMARY_FLAGS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grid", "split", *count_fields, *percentage_fields],
        )
        writer.writeheader()
        for (grid, split), summary in sorted(summaries.items()):
            candidates = int(summary["candidates"])
            row = {"grid": grid, "split": split, "candidates": candidates}
            for field in SUMMARY_FLAGS:
                row[field] = int(summary[field])
                row[f"{field}_pct"] = (
                    100.0 * int(summary[field]) / candidates if candidates else 0.0
                )
            for reason in reason_names:
                row[f"reason_{reason}"] = int(summary.get(f"reason:{reason}", 0))
            writer.writerow(row)


def run_audit(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    splits_path = args.splits.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite only for reports."
        )
    output.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(root)
    splits = read_splits(splits_path)
    if set(sources) != set(splits):
        missing_split = sorted(set(sources) - set(splits), key=_natural_id_key)
        missing_source = sorted(set(splits) - set(sources), key=_natural_id_key)
        raise ValueError(
            "Split/source ID mismatch: "
            f"without split={missing_split}; without source={missing_source}"
        )

    patient_ids = list(sources)
    if args.patient_id:
        requested = set(args.patient_id)
        unknown = requested - set(sources)
        if unknown:
            raise ValueError(f"Unknown patient IDs: {sorted(unknown)}")
        patient_ids = [item for item in patient_ids if item in requested]
    if args.max_patients is not None:
        patient_ids = patient_ids[: args.max_patients]

    patch_size = tuple(args.patch_size)
    pad = tuple(args.pad)
    crop_size = tuple(size - 2 * border for size, border in zip(patch_size, pad))
    if any(size <= 0 for size in crop_size):
        raise ValueError("Padding leaves a non-positive crop size")
    stride = tuple(
        max(1, min(crop, int(np.rint(crop * (1.0 - args.overlap)))))
        for crop in crop_size
    )
    grid_modes = ("full", "legacy") if args.grid == "both" else (args.grid,)

    inventory_path = output / "scan_inventory.csv"
    candidates_path = output / "candidate_audit.csv"
    summaries: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty_summary)

    with inventory_path.open("w", newline="") as inventory_handle, candidates_path.open(
        "w", newline=""
    ) as candidate_handle:
        inventory_writer = csv.DictWriter(inventory_handle, fieldnames=INVENTORY_FIELDS)
        candidate_writer = csv.DictWriter(candidate_handle, fieldnames=CANDIDATE_FIELDS)
        inventory_writer.writeheader()
        candidate_writer.writeheader()

        for index, patient_id in enumerate(patient_ids, start=1):
            raw_path, seg_path, graph_dir = sources[patient_id]
            split = splits[patient_id]
            print(
                f"[{index}/{len(patient_ids)}] patient={patient_id} split={split}",
                flush=True,
            )
            raw_image = nib.load(str(raw_path))
            seg_image = nib.load(str(seg_path))
            if raw_image.shape[:3] != seg_image.shape[:3]:
                raise ValueError(
                    f"Shape mismatch for {patient_id}: raw={raw_image.shape}, seg={seg_image.shape}"
                )
            if len(raw_image.shape) != 3 or len(seg_image.shape) != 3:
                raise ValueError(
                    f"Expected 3D volumes for {patient_id}: raw={raw_image.shape}, seg={seg_image.shape}"
                )
            raw_array = np.asanyarray(raw_image.dataobj)
            seg_array = np.asarray(seg_image.dataobj)
            normalized, normalization_threshold = normalize_like_legacy(raw_array)
            source_graph = SourceGraph.from_directory(graph_dir)
            foreground_total = int((seg_array > 0).sum())
            inventory_writer.writerow(
                {
                    "patient_id": patient_id,
                    "split": split,
                    "shape_d": raw_image.shape[0],
                    "shape_h": raw_image.shape[1],
                    "shape_w": raw_image.shape[2],
                    "spacing_d": raw_image.header.get_zooms()[0],
                    "spacing_h": raw_image.header.get_zooms()[1],
                    "spacing_w": raw_image.header.get_zooms()[2],
                    "raw_dtype": str(raw_image.get_data_dtype()),
                    "seg_dtype": str(seg_image.get_data_dtype()),
                    "seg_affine_matches_raw": bool(
                        np.allclose(raw_image.affine, seg_image.affine)
                    ),
                    "affine_determinant": float(np.linalg.det(raw_image.affine[:3, :3])),
                    "raw_min": float(raw_array.min()),
                    "raw_max": float(raw_array.max()),
                    "raw_mean": float(raw_array.mean()),
                    "raw_std": float(raw_array.std()),
                    "segmentation_foreground_voxels": foreground_total,
                    "segmentation_foreground_ratio": foreground_total / float(seg_array.size),
                    "full_graph_nodes": len(source_graph.nodes),
                    "full_graph_edges": len(source_graph.edges),
                    "legacy_normalization_threshold": normalization_threshold,
                }
            )
            inventory_handle.flush()

            for grid in grid_modes:
                if grid == "full":
                    positions = complete_grid_positions(raw_image.shape, crop_size, stride)
                else:
                    positions = legacy_grid_positions(
                        raw_image.shape,
                        crop_size,
                        stride,
                        patient_id=patient_id,
                        seed=args.seed,
                    )
                print(f"  grid={grid} candidates={len(positions)}", flush=True)
                for start in positions:
                    row = _candidate_row(
                        patient_id=patient_id,
                        split=split,
                        grid=grid,
                        start=start,
                        crop_size=crop_size,
                        patch_size=patch_size,
                        pad=pad,
                        image=normalized,
                        segmentation=seg_array,
                        graph=source_graph,
                        affine=raw_image.affine,
                    )
                    candidate_writer.writerow(row)
                    update_summary(summaries[(grid, split)], row)
                candidate_handle.flush()

            del raw_array, normalized, seg_array, source_graph

    summary_csv = output / "audit_summary_by_split.csv"
    _write_summary_csv(summary_csv, summaries)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "splits": str(splits_path),
        "patient_count": len(patient_ids),
        "patients": patient_ids,
        "grid_modes": list(grid_modes),
        "patch_size": list(patch_size),
        "pad": list(pad),
        "crop_size": list(crop_size),
        "overlap": args.overlap,
        "stride": list(stride),
        "legacy_grid_seed": args.seed,
        "notes": [
            "No source or patch files were written or modified.",
            "GT fields describe every candidate and do not control audit inclusion.",
            "Legacy quota stopping is intentionally not simulated because the exact historical CLI is unverified.",
        ],
        "summaries": {
            f"{grid}:{split}": dict(summary)
            for (grid, split), summary in sorted(summaries.items())
        },
    }
    with (output / "audit_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {inventory_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {output / 'audit_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of SyntheticMRI scan dimensions and candidate-patch "
            "segmentation/graph content."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", choices=("full", "legacy", "both"), default="full")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--patch-size", type=int, nargs=3, default=PATCH_SIZE)
    parser.add_argument("--pad", type=int, nargs=3, default=PAD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument("--max-patients", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing report files in a non-empty output directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must lie in [0,1)")
    if args.max_patients is not None and args.max_patients <= 0:
        parser.error("--max-patients must be positive")
    try:
        run_audit(args)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
