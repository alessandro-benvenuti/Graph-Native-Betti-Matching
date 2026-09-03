#!/usr/bin/env python3
"""Diagnose graph/crop boundary inconsistencies for SyntheticMRI patches.

The command is read-only with respect to source volumes and generated patches.
It compares the inherited sample-based crop policy with exact clipping of every
source centerline segment against the patch box.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import (
    CenterlineEdge,
    SourceGraph,
    patch_world_bounds,
    world_to_voxel,
)
from scripts.generate_synthetic_mri_dataset import write_csv_rows


DEFAULT_SAMPLES = (
    "sample_000038_0518",
    "sample_000056_0493",
    "sample_000060_0216",
    "sample_000131_0216",
    "sample_000036_0419",
    "sample_000125_0880",
    "sample_000025_0167",
    "sample_000125_0657",
    "sample_000121_0167",
    "sample_000051_0877",
)

SUMMARY_FIELDS = (
    "sample_id",
    "patient_id",
    "split",
    "patch_index",
    "start_d",
    "start_h",
    "start_w",
    "saved_nodes",
    "saved_edges",
    "source_edges_intersecting",
    "exact_clipped_components",
    "tangent_contacts",
    "current_clipped_components",
    "missing_components",
    "extra_components",
    "reversed_centerlines",
    "empty_centerlines",
    "boundary_nodes_compared",
    "displaced_boundary_nodes",
    "max_boundary_displacement_voxels",
    "seg_face_d0",
    "seg_face_d1",
    "seg_face_h0",
    "seg_face_h1",
    "seg_face_w0",
    "seg_face_w1",
)

EDGE_FIELDS = (
    "sample_id",
    "patient_id",
    "source_node1",
    "source_node2",
    "node1_inside",
    "node2_inside",
    "centerline_points",
    "orientation",
    "forward_endpoint_cost",
    "reverse_endpoint_cost",
    "max_polyline_step_voxels",
    "exact_components",
    "tangent_contacts",
    "current_components",
    "missing_components",
    "extra_components",
    "current_boundary_nodes",
    "exact_boundary_nodes",
    "max_boundary_displacement_voxels",
    "status",
)


def _inside(point: np.ndarray, bounds: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(
        np.all(point >= bounds[:, 0] - tolerance)
        and np.all(point <= bounds[:, 1] + tolerance)
    )


def clip_segment_to_box(
    start: np.ndarray, end: np.ndarray, bounds: np.ndarray, tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the exact part of a 3-D segment inside an axis-aligned box."""

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


def orient_edge_polyline(
    edge: CenterlineEdge, nodes: Mapping[int, np.ndarray]
) -> tuple[np.ndarray, str, float, float]:
    """Orient samples from node1 to node2 and include both true endpoints."""

    node1 = np.asarray(nodes[edge.node1], dtype=np.float64)
    node2 = np.asarray(nodes[edge.node2], dtype=np.float64)
    samples = np.asarray(edge.positions, dtype=np.float64).reshape(-1, 3)
    if not len(samples):
        return np.stack((node1, node2)), "empty", float("nan"), float("nan")

    forward_cost = float(
        np.linalg.norm(samples[0] - node1) + np.linalg.norm(samples[-1] - node2)
    )
    reverse_cost = float(
        np.linalg.norm(samples[0] - node2) + np.linalg.norm(samples[-1] - node1)
    )
    scale = max(1.0, forward_cost, reverse_cost)
    if reverse_cost + 1e-9 * scale < forward_cost:
        samples = samples[::-1]
        orientation = "reversed"
    elif abs(forward_cost - reverse_cost) <= 1e-9 * scale:
        orientation = "ambiguous"
    else:
        orientation = "forward"

    points = [node1]
    points.extend(samples)
    points.append(node2)
    deduplicated = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, deduplicated[-1], atol=1e-9, rtol=0.0):
            deduplicated.append(point)
    return np.asarray(deduplicated), orientation, forward_cost, reverse_cost


def clip_polyline_to_box(
    points: np.ndarray, bounds: np.ndarray, tolerance: float = 1e-8
) -> list[np.ndarray]:
    """Use the production cropper's closed-box component definition."""

    return SourceGraph._clip_polyline(points, bounds, tolerance=tolerance)


def _on_boundary(point: np.ndarray, bounds: np.ndarray, tolerance: float = 1e-7) -> bool:
    return bool(
        np.any(np.isclose(point, bounds[:, 0], atol=tolerance, rtol=0.0))
        or np.any(np.isclose(point, bounds[:, 1], atol=tolerance, rtol=0.0))
    )


def exact_boundary_nodes(components: Sequence[np.ndarray], bounds: np.ndarray) -> list[np.ndarray]:
    result = []
    for component in components:
        for point in (component[0], component[-1]):
            if _on_boundary(point, bounds) and not any(
                np.allclose(point, present, atol=1e-7, rtol=0.0) for present in result
            ):
                result.append(point)
    return result


def inherited_edge_crop(
    graph: SourceGraph, edge: CenterlineEdge, bounds: np.ndarray
) -> tuple[int, list[np.ndarray]]:
    """Describe what the current SourceGraph.crop policy contributes for one edge."""

    node1_inside = _inside(graph.nodes[edge.node1], bounds)
    node2_inside = _inside(graph.nodes[edge.node2], bounds)
    if node1_inside and node2_inside:
        return 1, []
    if node1_inside:
        boundary = graph._first_exit(edge.positions, bounds)
        return (1, [boundary]) if boundary is not None else (0, [])
    if node2_inside:
        boundary = graph._first_exit(tuple(reversed(edge.positions)), bounds)
        return (1, [boundary]) if boundary is not None else (0, [])
    segments = graph._inside_segments(edge.positions, bounds)
    return len(segments), [point for segment in segments for point in segment]


def _points_json(points: Sequence[np.ndarray]) -> str:
    return json.dumps([[round(float(value), 6) for value in point] for point in points])


def _boundary_displacements(
    current: Sequence[np.ndarray], exact: Sequence[np.ndarray], affine: np.ndarray
) -> list[float]:
    if not current or not exact:
        return []
    exact_voxel = world_to_voxel(np.asarray(exact), affine)
    current_voxel = world_to_voxel(np.asarray(current), affine)
    return [
        float(np.linalg.norm(exact_voxel - point, axis=1).min())
        for point in current_voxel
    ]


def _read_patch_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in result:
            raise ValueError(f"Duplicate sample ID in {path}: {sample_id}")
        result[sample_id] = row
    return result


def _face_foreground(segmentation: np.ndarray) -> tuple[int, int, int, int, int, int]:
    mask = segmentation > 0
    return tuple(
        int(np.count_nonzero(face))
        for face in (mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1])
    )


def diagnose_sample(
    root: Path,
    row: Mapping[str, str],
    patch_size: Sequence[int],
    pad: Sequence[int],
    displacement_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    patient_id = row["patient_id"]
    sample_id = row["sample_id"]
    start = np.asarray(
        [int(row["start_d"]), int(row["start_h"]), int(row["start_w"])],
        dtype=np.int64,
    )
    crop_size = np.asarray(patch_size, dtype=np.int64) - 2 * np.asarray(pad, dtype=np.int64)
    raw_image = nib.load(str(root / "raw" / f"{patient_id}.nii.gz"))
    segmentation_image = nib.load(str(root / "seg" / f"{patient_id}.nii.gz"))
    graph = SourceGraph.from_directory(root / "graphs" / patient_id)
    bounds = patch_world_bounds(start, crop_size, raw_image.affine)
    slices = tuple(slice(int(origin), int(origin + size)) for origin, size in zip(start, crop_size))
    segmentation = np.asarray(segmentation_image.dataobj)[slices]

    edge_rows = []
    intersecting = exact_total = current_total = 0
    reversed_count = empty_count = compared = displaced = tangent_total = 0
    missing_total = extra_total = 0
    maximum_displacement = 0.0

    for edge in graph.centerlines:
        if edge.node1 not in graph.nodes or edge.node2 not in graph.nodes:
            continue
        polyline, orientation, forward_cost, reverse_cost = orient_edge_polyline(edge, graph.nodes)
        exact_components = clip_polyline_to_box(polyline, bounds)
        if not exact_components:
            continue
        intersecting += 1
        exact_count = len(exact_components)
        tangent_contacts = sum(len(component) == 1 for component in exact_components)
        current_count, current_boundary = inherited_edge_crop(graph, edge, bounds)
        exact_boundary = exact_boundary_nodes(exact_components, bounds)
        displacements = _boundary_displacements(current_boundary, exact_boundary, raw_image.affine)
        edge_max_displacement = max(displacements, default=0.0)
        edge_displaced = sum(value > displacement_tolerance for value in displacements)
        missing = max(0, exact_count - current_count)
        extra = max(0, current_count - exact_count)
        node1_inside = _inside(graph.nodes[edge.node1], bounds)
        node2_inside = _inside(graph.nodes[edge.node2], bounds)

        exact_total += exact_count
        tangent_total += tangent_contacts
        current_total += current_count
        missing_total += missing
        extra_total += extra
        reversed_count += orientation == "reversed"
        empty_count += orientation == "empty"
        compared += len(displacements)
        displaced += edge_displaced
        maximum_displacement = max(maximum_displacement, edge_max_displacement)

        statuses = []
        if missing:
            statuses.append("missing_component")
        if extra:
            statuses.append("extra_component")
        if edge_displaced:
            statuses.append("displaced_boundary")
        if tangent_contacts:
            statuses.append("tangent_contact")
        if orientation == "reversed":
            statuses.append("reversed_samples")
        if orientation == "empty":
            statuses.append("empty_samples")
        if not statuses:
            statuses.append("ok")
        polyline_voxel = world_to_voxel(polyline, raw_image.affine)
        maximum_step = max(
            (float(np.linalg.norm(end - begin)) for begin, end in zip(polyline_voxel[:-1], polyline_voxel[1:])),
            default=0.0,
        )
        edge_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "source_node1": edge.node1,
                "source_node2": edge.node2,
                "node1_inside": node1_inside,
                "node2_inside": node2_inside,
                "centerline_points": len(edge.positions),
                "orientation": orientation,
                "forward_endpoint_cost": forward_cost,
                "reverse_endpoint_cost": reverse_cost,
                "max_polyline_step_voxels": maximum_step,
                "exact_components": exact_count,
                "tangent_contacts": tangent_contacts,
                "current_components": current_count,
                "missing_components": missing,
                "extra_components": extra,
                "current_boundary_nodes": _points_json(current_boundary),
                "exact_boundary_nodes": _points_json(exact_boundary),
                "max_boundary_displacement_voxels": edge_max_displacement,
                "status": ";".join(statuses),
            }
        )

    faces = _face_foreground(segmentation)
    summary = {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "split": row["split"],
        "patch_index": row["patch_index"],
        "start_d": int(start[0]),
        "start_h": int(start[1]),
        "start_w": int(start[2]),
        "saved_nodes": int(row["node_count"]),
        "saved_edges": int(row["edge_count"]),
        "source_edges_intersecting": intersecting,
        "exact_clipped_components": exact_total,
        "tangent_contacts": tangent_total,
        "current_clipped_components": current_total,
        "missing_components": missing_total,
        "extra_components": extra_total,
        "reversed_centerlines": reversed_count,
        "empty_centerlines": empty_count,
        "boundary_nodes_compared": compared,
        "displaced_boundary_nodes": displaced,
        "max_boundary_displacement_voxels": maximum_displacement,
        "seg_face_d0": faces[0],
        "seg_face_d1": faces[1],
        "seg_face_h0": faces[2],
        "seg_face_h1": faces[3],
        "seg_face_w0": faces[4],
        "seg_face_w1": faces[5],
    }
    return summary, edge_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--pad", type=int, nargs=3, default=(5, 5, 5))
    parser.add_argument("--displacement-tolerance", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    patch_root = (args.patch_root or root / "new_patches").resolve()
    output = (args.output_dir or root / "results" / "patch_boundary_diagnostics").resolve()
    index = _read_patch_index(patch_root / "patch_index.csv")
    missing = [sample for sample in args.samples if sample not in index]
    if missing:
        raise KeyError(f"Samples absent from {patch_root / 'patch_index.csv'}: {missing}")

    summaries = []
    edge_rows = []
    for sample in args.samples:
        summary, sample_edges = diagnose_sample(
            root,
            index[sample],
            args.patch_size,
            args.pad,
            args.displacement_tolerance,
        )
        summaries.append(summary)
        edge_rows.extend(sample_edges)
        print(
            f"{sample}: saved={summary['saved_nodes']}N/{summary['saved_edges']}E "
            f"missing={summary['missing_components']} "
            f"displaced={summary['displaced_boundary_nodes']} "
            f"reversed={summary['reversed_centerlines']} "
            f"max_shift={summary['max_boundary_displacement_voxels']:.3f} vox"
        )

    write_csv_rows(output / "patch_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv_rows(output / "edge_diagnostics.csv", EDGE_FIELDS, edge_rows)
    suspicious = [row for row in edge_rows if row["status"] != "ok"]
    write_csv_rows(output / "suspicious_edges.csv", EDGE_FIELDS, suspicious)
    print(f"Wrote {output / 'patch_summary.csv'}")
    print(f"Wrote {output / 'edge_diagnostics.csv'}")
    print(f"Wrote {output / 'suspicious_edges.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
