#!/usr/bin/env python3
"""Generate the versioned SyntheticMRI patch dataset without GT selection.

The generator creates ``new_split.csv`` and ``new_patches/`` below the source
root by default. It never modifies ``patches/`` or ``splits.csv``. Candidate
locations are fixed from MRI geometry alone using an endpoint-distributed grid;
every candidate is saved, including empty or sparse targets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_synthetic_mri_grid import (
    SourceGraph,
    discover_sources,
    normalize_like_legacy,
    patch_world_bounds,
    read_splits,
    world_to_voxel,
)


SPLITS = ("train", "val", "test")
PATCH_INDEX_FIELDS = (
    "sample_id",
    "patient_id",
    "split",
    "patch_index",
    "start_d",
    "start_h",
    "start_w",
    "node_count",
    "edge_count",
    "legacy_node_count",
    "legacy_edge_count",
    "graph_crop_changed",
    "image_mean",
    "image_std",
    "foreground_voxels",
    "foreground_fraction",
)
FEATURE_NAMES = (
    "foreground_fraction",
    "node_count",
    "edge_count",
    "bifurcation_count",
    "betti_0",
    "betti_1",
)


def natural_id_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def endpoint_axis_starts(
    axis_size: int, crop_size: int, maximum_stride: int
) -> list[int]:
    """Cover both endpoints with integer starts distributed as evenly as possible."""

    axis_size = int(axis_size)
    crop_size = int(crop_size)
    maximum_stride = int(maximum_stride)
    if axis_size < crop_size:
        raise ValueError(
            f"Axis size {axis_size} is smaller than crop size {crop_size}"
        )
    if maximum_stride <= 0:
        raise ValueError("maximum_stride must be positive")
    last = axis_size - crop_size
    if last == 0:
        return [0]
    intervals = int(math.ceil(last / maximum_stride))
    starts = np.rint(np.linspace(0, last, intervals + 1)).astype(np.int64).tolist()
    if starts[0] != 0 or starts[-1] != last:
        raise AssertionError("Endpoint grid failed to include both boundaries")
    if len(set(starts)) != len(starts):
        raise AssertionError(f"Endpoint grid contains duplicate starts: {starts}")
    if max(np.diff(starts)) > maximum_stride:
        raise AssertionError(f"Endpoint grid exceeds maximum stride: {starts}")
    return [int(value) for value in starts]


def endpoint_grid_positions(
    shape: Sequence[int], crop_size: Sequence[int], maximum_stride: Sequence[int]
) -> list[tuple[int, int, int]]:
    axes = [
        endpoint_axis_starts(size, crop, stride)
        for size, crop, stride in zip(shape, crop_size, maximum_stride)
    ]
    return [(d, h, w) for d in axes[0] for h in axes[1] for w in axes[2]]


@dataclass(frozen=True)
class PatientFeatures:
    patient_id: str
    foreground_fraction: float
    node_count: int
    edge_count: int
    bifurcation_count: int
    betti_0: int
    betti_1: int

    def vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in FEATURE_NAMES], dtype=np.float64
        )


def graph_topology_features(graph: SourceGraph) -> tuple[int, int, int]:
    """Return bifurcations, beta0, and beta1 for an undirected multigraph."""

    node_ids = list(graph.nodes)
    parent = {node_id: node_id for node_id in node_ids}
    degree = Counter({node_id: 0 for node_id in node_ids})

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in graph.edges:
        if left not in parent or right not in parent:
            raise ValueError(f"Graph edge references an unknown node: {(left, right)}")
        degree[left] += 1
        degree[right] += 1
        union(left, right)

    components = len({find(node_id) for node_id in node_ids}) if node_ids else 0
    beta_1 = len(graph.edges) - len(node_ids) + components
    if beta_1 < 0:
        raise ValueError(f"Invalid graph cycle rank: {beta_1}")
    bifurcations = sum(value >= 3 for value in degree.values())
    return int(bifurcations), int(components), int(beta_1)


def collect_patient_features(
    sources: Mapping[str, tuple[Path, Path, Path]]
) -> list[PatientFeatures]:
    features = []
    total = len(sources)
    for index, patient_id in enumerate(sources, start=1):
        _, segmentation_path, graph_directory = sources[patient_id]
        print(f"[split {index}/{total}] patient={patient_id}", flush=True)
        segmentation_image = nib.load(str(segmentation_path))
        if len(segmentation_image.shape) != 3:
            raise ValueError(
                f"Expected a 3D segmentation for {patient_id}, "
                f"received {segmentation_image.shape}"
            )
        segmentation = np.asarray(segmentation_image.dataobj)
        foreground_fraction = float(np.count_nonzero(segmentation > 0) / segmentation.size)
        graph = SourceGraph.from_directory(graph_directory)
        bifurcations, beta_0, beta_1 = graph_topology_features(graph)
        features.append(
            PatientFeatures(
                patient_id=patient_id,
                foreground_fraction=foreground_fraction,
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                bifurcation_count=bifurcations,
                betti_0=beta_0,
                betti_1=beta_1,
            )
        )
    return features


def split_sizes(patient_count: int) -> dict[str, int]:
    if patient_count < 3:
        raise ValueError("At least three patients are required for train/val/test")
    train = int(math.floor(patient_count * 0.70))
    val = int(math.floor(patient_count * 0.15))
    test = patient_count - train - val
    return {"train": train, "val": val, "test": test}


def balanced_patient_split(
    features: Sequence[PatientFeatures], *, seed: int, trials: int
) -> tuple[dict[str, str], float]:
    """Select the lowest-imbalance exact-size split from deterministic trials."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    ordered = sorted(features, key=lambda item: natural_id_key(item.patient_id))
    sizes = split_sizes(len(ordered))
    matrix = np.stack([item.vector() for item in ordered])
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    active = scales > 0
    standardized = np.zeros_like(matrix)
    standardized[:, active] = (matrix[:, active] - means[active]) / scales[active]

    rng = np.random.default_rng(seed)
    best_score = math.inf
    best_permutation = None
    train_end = sizes["train"]
    val_end = train_end + sizes["val"]

    for _ in range(trials):
        permutation = rng.permutation(len(ordered))
        groups = (
            permutation[:train_end],
            permutation[train_end:val_end],
            permutation[val_end:],
        )
        score = 0.0
        for indices in groups:
            group = standardized[indices][:, active]
            if group.size:
                score += float(np.mean(np.abs(group.mean(axis=0))))
                score += 0.25 * float(np.mean(np.abs(group.std(axis=0) - 1.0)))
        if score < best_score:
            best_score = score
            best_permutation = permutation.copy()

    assert best_permutation is not None
    result = {}
    for split, indices in (
        ("train", best_permutation[:train_end]),
        ("val", best_permutation[train_end:val_end]),
        ("test", best_permutation[val_end:]),
    ):
        for index in indices:
            result[ordered[int(index)].patient_id] = split
    return result, float(best_score)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_split_csv(path: Path, split_map: Mapping[str, str]) -> None:
    rows = ["patient_id,split"]
    rows.extend(
        f"{patient_id},{split_map[patient_id]}"
        for patient_id in sorted(split_map, key=natural_id_key)
    )
    atomic_write_text(path, "\n".join(rows) + "\n")


def read_patient_features(path: Path) -> list[PatientFeatures]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        PatientFeatures(
            patient_id=str(row["patient_id"]),
            foreground_fraction=float(row["foreground_fraction"]),
            node_count=int(row["node_count"]),
            edge_count=int(row["edge_count"]),
            bifurcation_count=int(row["bifurcation_count"]),
            betti_0=int(row["betti_0"]),
            betti_1=int(row["betti_1"]),
        )
        for row in rows
    ]


def split_diagnostics(
    features: Sequence[PatientFeatures], split_map: Mapping[str, str]
) -> list[dict[str, object]]:
    feature_by_id = {item.patient_id: item for item in features}
    rows = []
    for split in (*SPLITS, "all"):
        patient_ids = (
            list(feature_by_id)
            if split == "all"
            else [patient_id for patient_id, value in split_map.items() if value == split]
        )
        matrix = np.stack([feature_by_id[patient_id].vector() for patient_id in patient_ids])
        for column, feature_name in enumerate(FEATURE_NAMES):
            values = matrix[:, column]
            rows.append(
                {
                    "split": split,
                    "patients": len(patient_ids),
                    "feature": feature_name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configuration_fingerprint(configuration: Mapping[str, object]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_nifti(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(temporary))
    os.replace(temporary, path)


def hardlink_patch(source: Path, destination: Path) -> None:
    """Materialize an unchanged patch without duplicating its file payload."""

    if not source.is_file():
        raise FileNotFoundError(f"Reusable patch is missing: {source}")
    try:
        os.link(source, destination)
    except OSError as error:
        raise OSError(
            f"Could not hard-link {source} to {destination}. Both datasets must be "
            "on the same filesystem; no implicit full-data copy was attempted."
        ) from error


def write_vtp_graph(path: Path, positions: np.ndarray, edges: np.ndarray) -> None:
    """Write a minimal ASCII VTK PolyData graph readable by PyVista/VTK."""

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= len(positions)):
        raise ValueError("Graph edge index is outside the point array")
    point_values = " ".join(f"{float(value):.9g}" for value in positions.reshape(-1))
    connectivity = " ".join(str(int(value)) for value in edges.reshape(-1))
    offsets = " ".join(str(2 * (index + 1)) for index in range(len(edges)))
    payload = f"""<?xml version=\"1.0\"?>
<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">
  <PolyData>
    <Piece NumberOfPoints=\"{len(positions)}\" NumberOfVerts=\"0\" NumberOfLines=\"{len(edges)}\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">
      <Points>
        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">{point_values}</DataArray>
      </Points>
      <Lines>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">{connectivity}</DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">{offsets}</DataArray>
      </Lines>
    </Piece>
  </PolyData>
</VTKFile>
"""
    atomic_write_text(path, payload)


def graph_geometry_signature(graph) -> tuple[tuple, tuple]:
    """Return an index-independent, tolerance-stable geometric graph signature."""

    points = [tuple(np.round(position, decimals=6)) for position in graph.positions]
    nodes = tuple(sorted(points))
    edges = []
    for left, right in graph.edges:
        endpoints = sorted((points[int(left)], points[int(right)]))
        edges.append(tuple(endpoints))
    return nodes, tuple(sorted(edges))


def patient_token(patient_id: str) -> str:
    try:
        return f"{int(patient_id):06d}"
    except ValueError as error:
        raise ValueError(f"Patch filenames require numeric patient IDs: {patient_id}") from error


def remove_incomplete_patient(output: Path, split: str, patient_id: str) -> None:
    token = patient_token(patient_id)
    patterns = {
        "raw": f"sample_{token}_*_data.nii.gz",
        "seg": f"sample_{token}_*_seg.nii.gz",
        "vtp": f"sample_{token}_*_graph.vtp",
    }
    for folder, pattern in patterns.items():
        for path in (output / split / folder).glob(pattern):
            path.unlink()
    manifest = output / ".manifests" / f"{token}.csv"
    if manifest.exists():
        manifest.unlink()


def _generate_patient(task: Mapping[str, object]) -> dict[str, object]:
    patient_id = str(task["patient_id"])
    split = str(task["split"])
    raw_path = Path(str(task["raw_path"]))
    segmentation_path = Path(str(task["segmentation_path"]))
    graph_directory = Path(str(task["graph_directory"]))
    output = Path(str(task["output"]))
    patch_size = tuple(int(value) for value in task["patch_size"])
    pad = tuple(int(value) for value in task["pad"])
    crop_size = tuple(size - 2 * border for size, border in zip(patch_size, pad))
    maximum_stride = tuple(int(value) for value in task["maximum_stride"])
    fingerprint = str(task["fingerprint"])
    reuse_patch_root = (
        Path(str(task["reuse_patch_root"])) if task.get("reuse_patch_root") else None
    )

    marker = output / ".complete" / f"{patient_token(patient_id)}.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"Completion marker configuration mismatch for {patient_id}")
        return {**payload, "skipped": True}

    remove_incomplete_patient(output, split, patient_id)
    raw_image = nib.load(str(raw_path))
    segmentation_image = nib.load(str(segmentation_path))
    if len(raw_image.shape) != 3 or len(segmentation_image.shape) != 3:
        raise ValueError(
            f"Expected 3D volumes for {patient_id}: "
            f"raw={raw_image.shape}, seg={segmentation_image.shape}"
        )
    if raw_image.shape != segmentation_image.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id}: "
            f"raw={raw_image.shape}, seg={segmentation_image.shape}"
        )
    if not np.allclose(raw_image.affine, segmentation_image.affine):
        raise ValueError(f"Raw/segmentation affine mismatch for {patient_id}")

    reuse_rows = task.get("reuse_rows")
    if reuse_rows is None:
        raw = np.asanyarray(raw_image.dataobj)
        segmentation = np.asarray(segmentation_image.dataobj)
        normalized, threshold = normalize_like_legacy(raw)
    else:
        segmentation = None
        normalized = None
        threshold = float(task["normalization_threshold"])
    graph = SourceGraph.from_directory(graph_directory)
    positions = endpoint_grid_positions(raw_image.shape, crop_size, maximum_stride)
    if reuse_rows is not None and len(reuse_rows) != len(positions):
        raise ValueError(
            f"Reusable patch count mismatch for {patient_id}: "
            f"received={len(reuse_rows)}, expected={len(positions)}"
        )
    rows = []
    token = patient_token(patient_id)

    for patch_index, start in enumerate(positions):
        slices = tuple(
            slice(origin, origin + size) for origin, size in zip(start, crop_size)
        )
        if reuse_rows is None:
            image_patch = np.pad(
                normalized[slices], tuple((border, border) for border in pad)
            ).astype(np.float32, copy=False)
            segmentation_patch = np.pad(
                segmentation[slices], tuple((border, border) for border in pad)
            )
            image_mean = float(image_patch.mean())
            image_std = float(image_patch.std())
            foreground_voxels = int(np.count_nonzero(segmentation_patch > 0))
            foreground_fraction = float(foreground_voxels / segmentation_patch.size)
        else:
            reused = reuse_rows[patch_index]
            expected_sample_id = f"sample_{token}_{patch_index:04d}"
            if reused["sample_id"] != expected_sample_id:
                raise ValueError(
                    f"Reusable sample ID mismatch: received={reused['sample_id']}, "
                    f"expected={expected_sample_id}"
                )
            expected_start = tuple(
                int(reused[field]) for field in ("start_d", "start_h", "start_w")
            )
            if expected_start != tuple(start):
                raise ValueError(
                    f"Reusable grid mismatch for {patient_id} patch {patch_index}: "
                    f"received={expected_start}, expected={start}"
                )
            image_mean = float(reused["image_mean"])
            image_std = float(reused["image_std"])
            foreground_voxels = int(reused["foreground_voxels"])
            foreground_fraction = float(reused["foreground_fraction"])
        bounds = patch_world_bounds(start, crop_size, raw_image.affine)
        inherited_crop = graph.crop_inherited(bounds)
        cropped = graph.crop(bounds)
        if len(cropped.positions):
            voxel_positions = world_to_voxel(cropped.positions, raw_image.affine)
            local_positions = (
                voxel_positions - np.asarray(start, dtype=np.float64) + np.asarray(pad)
            )
            normalized_positions = local_positions / np.asarray(patch_size)
            minimum = float(normalized_positions.min())
            maximum = float(normalized_positions.max())
            if minimum < -0.01 or maximum > 1.01:
                raise ValueError(
                    f"Coordinates outside patch for patient {patient_id}, start={start}: "
                    f"min={minimum}, max={maximum}"
                )
        else:
            normalized_positions = np.empty((0, 3), dtype=np.float32)

        sample_id = f"sample_{token}_{patch_index:04d}"
        raw_destination = output / split / "raw" / f"{sample_id}_data.nii.gz"
        segmentation_destination = output / split / "seg" / f"{sample_id}_seg.nii.gz"
        if reuse_patch_root is None:
            write_nifti(raw_destination, image_patch)
            write_nifti(segmentation_destination, segmentation_patch)
        else:
            hardlink_patch(
                reuse_patch_root / split / "raw" / raw_destination.name,
                raw_destination,
            )
            hardlink_patch(
                reuse_patch_root / split / "seg" / segmentation_destination.name,
                segmentation_destination,
            )
        write_vtp_graph(
            output / split / "vtp" / f"{sample_id}_graph.vtp",
            normalized_positions,
            cropped.edges,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "split": split,
                "patch_index": patch_index,
                "start_d": start[0],
                "start_h": start[1],
                "start_w": start[2],
                "node_count": len(cropped.positions),
                "edge_count": cropped.edge_count,
                "legacy_node_count": len(inherited_crop.positions),
                "legacy_edge_count": inherited_crop.edge_count,
                "graph_crop_changed": (
                    graph_geometry_signature(cropped)
                    != graph_geometry_signature(inherited_crop)
                ),
                "image_mean": image_mean,
                "image_std": image_std,
                "foreground_voxels": foreground_voxels,
                "foreground_fraction": foreground_fraction,
            }
        )

    manifest_path = output / ".manifests" / f"{token}.csv"
    write_csv_rows(manifest_path, PATCH_INDEX_FIELDS, rows)
    payload = {
        "patient_id": patient_id,
        "split": split,
        "patches": len(rows),
        "normalization_threshold": threshold,
        "fingerprint": fingerprint,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(marker, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "skipped": False}


def combine_patient_manifests(output: Path) -> tuple[int, int, dict[str, dict[str, float]]]:
    rows = []
    manifests = sorted((output / ".manifests").glob("*.csv"))
    for path in manifests:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(
        key=lambda row: (
            SPLITS.index(str(row["split"])),
            natural_id_key(str(row["patient_id"])),
            int(row["patch_index"]),
        )
    )
    write_csv_rows(output / "patch_index.csv", PATCH_INDEX_FIELDS, rows)
    statistics = {}
    for split in (*SPLITS, "all"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        if selected:
            statistics[split] = {
                "patches": len(selected),
                "image_mean": float(np.mean([float(row["image_mean"]) for row in selected])),
                "foreground_fraction": float(
                    np.mean([float(row["foreground_fraction"]) for row in selected])
                ),
                "graph_crop_changed_patches": sum(
                    row["graph_crop_changed"] == "True" for row in selected
                ),
                "graph_crop_changed_fraction": float(
                    np.mean([row["graph_crop_changed"] == "True" for row in selected])
                ),
                "node_count_delta": sum(
                    int(row["node_count"]) - int(row["legacy_node_count"])
                    for row in selected
                ),
                "edge_count_delta": sum(
                    int(row["edge_count"]) - int(row["legacy_edge_count"])
                    for row in selected
                ),
            }
    return len(manifests), len(rows), statistics


def validate_split_map(
    split_map: Mapping[str, str], sources: Mapping[str, tuple[Path, Path, Path]]
) -> None:
    if set(split_map) != set(sources):
        raise ValueError(
            "Split/source mismatch: "
            f"without split={sorted(set(sources) - set(split_map), key=natural_id_key)}; "
            f"without source={sorted(set(split_map) - set(sources), key=natural_id_key)}"
        )
    counts = Counter(split_map.values())
    expected = split_sizes(len(sources))
    if dict(counts) != expected:
        raise ValueError(f"Unexpected split sizes: received={counts}, expected={expected}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a complete, patient-split SyntheticMRI patch dataset"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument(
        "--reuse-patches-from",
        type=Path,
        help="Hard-link unchanged raw/seg patches from a compatible generated dataset",
    )
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--pad", type=int, nargs=3, default=(5, 5, 5))
    parser.add_argument("--maximum-stride", type=int, nargs=3, default=(40, 40, 40))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument("--max-patients", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output = (args.output_dir or (root / "new_patches")).resolve()
    split_output = (args.split_output or (root / "new_split.csv")).resolve()
    reuse_patch_root = (
        args.reuse_patches_from.resolve() if args.reuse_patches_from else None
    )
    if reuse_patch_root == output:
        raise ValueError("--reuse-patches-from must differ from --output-dir")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.split_trials <= 0:
        raise ValueError("--split-trials must be positive")
    if args.max_patients is not None and args.max_patients <= 0:
        raise ValueError("--max-patients must be positive")
    patch_size = tuple(int(value) for value in args.patch_size)
    pad = tuple(int(value) for value in args.pad)
    crop_size = tuple(size - 2 * border for size, border in zip(patch_size, pad))
    if any(size <= 0 for size in crop_size):
        raise ValueError("Padding leaves a non-positive effective crop")
    if any(value <= 0 for value in args.maximum_stride):
        raise ValueError("Maximum strides must be positive")

    sources = discover_sources(root)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --resume only for this run."
        )
    features = None
    split_score = None
    if split_output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Split already exists: {split_output}. Use --resume to reuse it."
            )
        split_map = read_splits(split_output)
        validate_split_map(split_map, sources)
    else:
        features = collect_patient_features(sources)
        split_map, split_score = balanced_patient_split(
            features, seed=args.seed, trials=args.split_trials
        )
        validate_split_map(split_map, sources)
        write_split_csv(split_output, split_map)

    for split in SPLITS:
        for folder in ("raw", "seg", "vtp"):
            (output / split / folder).mkdir(parents=True, exist_ok=True)
    (output / ".complete").mkdir(parents=True, exist_ok=True)
    (output / ".manifests").mkdir(parents=True, exist_ok=True)

    feature_path = output / "patient_features.csv"
    if features is None and feature_path.exists():
        features = read_patient_features(feature_path)
        if {item.patient_id for item in features} != set(sources):
            raise ValueError(f"Patient feature/source mismatch: {feature_path}")
    if features is None and reuse_patch_root is not None:
        reusable_features = reuse_patch_root / "patient_features.csv"
        if not reusable_features.is_file():
            raise FileNotFoundError(
                f"Reusable patient features are missing: {reusable_features}"
            )
        features = read_patient_features(reusable_features)
        if {item.patient_id for item in features} != set(sources):
            raise ValueError(
                f"Reusable patient feature/source mismatch: {reusable_features}"
            )
    if features is None:
        features = collect_patient_features(sources)
    write_csv_rows(
        feature_path,
        ("patient_id", *FEATURE_NAMES),
        (asdict(item) for item in sorted(features, key=lambda item: natural_id_key(item.patient_id))),
    )
    diagnostic_rows = split_diagnostics(features, split_map)
    write_csv_rows(
        output / "split_balance.csv",
        ("split", "patients", "feature", "mean", "std", "minimum", "maximum"),
        diagnostic_rows,
    )

    split_bytes = split_output.read_bytes()
    configuration = {
        "format_version": 2,
        "source_root": str(root),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "patch_size": list(patch_size),
        "pad": list(pad),
        "crop_size": list(crop_size),
        "maximum_stride": list(args.maximum_stride),
        "grid": "endpoint_distributed_v1",
        "normalization": "legacy_mad_clip_v1",
        "graph_crop": "endpoint_aware_exact_polyline_box_v2",
        "selection_filter": None,
        "raw_seg_materialization": (
            "hardlink_from_existing" if reuse_patch_root is not None else "generated"
        ),
    }
    reuse_rows_by_patient = None
    reuse_thresholds = None
    if reuse_patch_root is not None:
        reuse_configuration_path = reuse_patch_root / "generation_config.json"
        if not reuse_configuration_path.is_file():
            raise FileNotFoundError(
                f"Reusable dataset has no generation configuration: {reuse_configuration_path}"
            )
        reuse_configuration = json.loads(reuse_configuration_path.read_text())
        compatible_fields = (
            "split_sha256",
            "patch_size",
            "pad",
            "crop_size",
            "maximum_stride",
            "grid",
            "normalization",
        )
        mismatches = [
            field
            for field in compatible_fields
            if reuse_configuration.get(field) != configuration.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Reusable dataset is incompatible for hard-linking raw/seg patches; "
                f"different fields: {mismatches}"
            )
        reuse_rows_by_patient = {patient_id: [] for patient_id in sources}
        with (reuse_patch_root / "patch_index.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                patient_id = row["patient_id"]
                if patient_id not in reuse_rows_by_patient:
                    raise ValueError(
                        f"Reusable patch index contains unknown patient: {patient_id}"
                    )
                if row["split"] != split_map[patient_id]:
                    raise ValueError(
                        f"Reusable split mismatch for patient {patient_id}: {row['split']}"
                    )
                reuse_rows_by_patient[patient_id].append(row)
        for rows in reuse_rows_by_patient.values():
            rows.sort(key=lambda row: int(row["patch_index"]))
            if [int(row["patch_index"]) for row in rows] != list(range(len(rows))):
                raise ValueError("Reusable patch indices must be contiguous from zero")
        reuse_thresholds = {}
        for patient_id in sources:
            marker = reuse_patch_root / ".complete" / f"{patient_token(patient_id)}.json"
            if not marker.is_file():
                raise FileNotFoundError(f"Reusable patient marker is missing: {marker}")
            reuse_thresholds[patient_id] = float(
                json.loads(marker.read_text())["normalization_threshold"]
            )
    fingerprint = configuration_fingerprint(configuration)
    manifest_path = output / "generation_config.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Existing generation configuration does not match: {manifest_path}"
            )
    else:
        atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    **configuration,
                    "fingerprint": fingerprint,
                    "seed": args.seed,
                    "split_trials": args.split_trials,
                    "split_score": split_score,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    counts = Counter(split_map.values())
    candidate_counts = {}
    for patient_id, (raw_path, _, _) in sources.items():
        shape = nib.load(str(raw_path)).shape
        candidate_counts[patient_id] = len(
            endpoint_grid_positions(shape, crop_size, args.maximum_stride)
        )
    total_candidates = sum(candidate_counts.values())
    candidates_by_split = {
        split: sum(
            candidate_counts[patient_id]
            for patient_id, assigned in split_map.items()
            if assigned == split
        )
        for split in SPLITS
    }
    print(f"Split: {dict(counts)}", flush=True)
    print(
        f"Grid: patch={patch_size}, crop={crop_size}, maximum_stride={tuple(args.maximum_stride)}",
        flush=True,
    )
    print(
        "Candidates per volume: "
        f"min={min(candidate_counts.values())}, max={max(candidate_counts.values())}",
        flush=True,
    )
    print(f"Candidates by split: {candidates_by_split}", flush=True)
    print(
        f"Estimated total candidates: {total_candidates}",
        flush=True,
    )
    uncompressed_bytes = total_candidates * (
        int(np.prod(patch_size)) * (np.dtype(np.float32).itemsize + 1)
    )
    print(
        "Approximate raw+seg uncompressed payload: "
        f"{uncompressed_bytes / 1024**3:.1f} GiB (NIfTI compression may reduce this)",
        flush=True,
    )
    if args.plan_only:
        print("Plan written; no patches generated (--plan-only).", flush=True)
        return 0

    patient_ids = list(sources)
    if args.patient_id:
        requested = set(args.patient_id)
        unknown = requested - set(sources)
        if unknown:
            raise ValueError(f"Unknown patient IDs: {sorted(unknown, key=natural_id_key)}")
        patient_ids = [patient_id for patient_id in patient_ids if patient_id in requested]
    if args.max_patients is not None:
        patient_ids = patient_ids[: args.max_patients]

    tasks = []
    for patient_id in patient_ids:
        raw_path, segmentation_path, graph_directory = sources[patient_id]
        tasks.append(
            {
                "patient_id": patient_id,
                "split": split_map[patient_id],
                "raw_path": str(raw_path),
                "segmentation_path": str(segmentation_path),
                "graph_directory": str(graph_directory),
                "output": str(output),
                "patch_size": patch_size,
                "pad": pad,
                "maximum_stride": tuple(args.maximum_stride),
                "fingerprint": fingerprint,
                "reuse_patch_root": (
                    str(reuse_patch_root) if reuse_patch_root is not None else None
                ),
                "reuse_rows": (
                    reuse_rows_by_patient[patient_id]
                    if reuse_rows_by_patient is not None
                    else None
                ),
                "normalization_threshold": (
                    reuse_thresholds[patient_id]
                    if reuse_thresholds is not None
                    else None
                ),
            }
        )

    results = []
    if args.workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = _generate_patient(task)
            results.append(result)
            action = "skipped" if result["skipped"] else "completed"
            print(
                f"[{index}/{len(tasks)}] patient={result['patient_id']} "
                f"split={result['split']} patches={result['patches']} {action}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_generate_patient, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                action = "skipped" if result["skipped"] else "completed"
                print(
                    f"[{index}/{len(tasks)}] patient={result['patient_id']} "
                    f"split={result['split']} patches={result['patches']} {action}",
                    flush=True,
                )

    completed_patients, patch_count, dataset_statistics = combine_patient_manifests(output)
    expected_patients = len(sources)
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completed_patients": completed_patients,
        "expected_patients": expected_patients,
        "patches_indexed": patch_count,
        "complete": completed_patients == expected_patients,
        "fingerprint": fingerprint,
        "dataset_statistics": dataset_statistics,
    }
    atomic_write_text(
        output / "generation_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
