"""Optional integration test against existing Plants PNG/VTP patches.

Set ``PLANTS_DATASET`` to a dataset root containing either
``train/{raw,seg,vtp}`` or directly ``{raw,seg,vtp}``. The test uses the
stored normalized graph convention from ``generate_plants_data.py`` and
exercises the active graph-safe 3D geometry without changing production code.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import torch
import torch.nn.functional as F

from data.augmentations import (
    AugmentationParameters,
    apply_augmentation,
    embed_2d_coordinates,
    pad_coordinates,
    project_2d_to_3d,
)
from tests.integration_helpers import (
    node_foreground_hit_rate,
    node_foreground_neighbourhood_hit_rate,
    select_evenly_spaced,
)


DATASET_ROOT = os.environ.get("PLANTS_DATASET")


def _resolve_leaf(root: Path) -> Path:
    candidates = (root / "train", root)
    for candidate in candidates:
        if all((candidate / folder).is_dir() for folder in ("raw", "seg", "vtp")):
            return candidate
    raise FileNotFoundError(
        f"Could not find raw/seg/vtp below {root} or {root / 'train'}"
    )


def _selected_triplets(
    leaf: Path,
    sample_count: int,
) -> list[tuple[Path, Path, Path]]:
    raw_files = sorted((leaf / "raw").glob("*_data.png"))
    if not raw_files:
        raise FileNotFoundError(f"No *_data.png files in {leaf / 'raw'}")

    triplets = []
    for raw_path in select_evenly_spaced(raw_files, sample_count):
        stem = raw_path.name[: -len("_data.png")]
        seg_path = leaf / "seg" / f"{stem}_seg.png"
        graph_path = leaf / "vtp" / f"{stem}_graph.vtp"
        if not seg_path.exists() or not graph_path.exists():
            raise FileNotFoundError(
                f"Could not pair {raw_path.name} with "
                f"{seg_path.name} and {graph_path.name}"
            )
        triplets.append((raw_path, seg_path, graph_path))
    return triplets


def _symmetric_pad_3d(
    volume: torch.Tensor,
    padding: int,
    *,
    value: float,
) -> torch.Tensor:
    return F.pad(
        volume,
        (padding, padding, padding, padding, padding, padding),
        mode="constant",
        value=value,
    )


def _stored_points_to_xy(
    stored_points: torch.Tensor,
    *,
    coordinate_order: str,
    coordinate_storage: str,
    height: int,
    width: int,
) -> torch.Tensor:
    points_xy = (
        stored_points[:, [1, 0]]
        if coordinate_order == "yx"
        else stored_points.clone()
    )
    if coordinate_storage == "voxel":
        points_xy[:, 0] /= float(width)
        points_xy[:, 1] /= float(height)
    return points_xy


class PlantsCoordinateConventionTests(unittest.TestCase):
    def test_generator_yx_storage_projects_to_the_expected_foreground(self) -> None:
        segmentation_2d = torch.zeros((1, 5, 5), dtype=torch.float32)
        segmentation_2d[0, 1, 3] = 1.0
        stored_yx = torch.tensor([[1 / 5, 3 / 5]], dtype=torch.float32)
        points_xy = _stored_points_to_xy(
            stored_yx,
            coordinate_order="yx",
            coordinate_storage="normalized",
            height=5,
            width=5,
        )
        segmentation_3d = project_2d_to_3d(
            segmentation_2d,
            z_position=0.5,
            thickness=5,
            depth=5,
        )
        nodes = embed_2d_coordinates(points_xy, z_position=0.5)
        self.assertEqual(node_foreground_hit_rate(segmentation_3d, nodes), 1.0)


@unittest.skipUnless(
    DATASET_ROOT,
    "Set PLANTS_DATASET to run integration checks on existing Plants patches",
)
class ExistingPlantsDatasetTests(unittest.TestCase):
    def test_graph_alignment_survives_active_geometric_pipeline(self) -> None:
        leaf = _resolve_leaf(Path(DATASET_ROOT))
        sample_count = int(os.environ.get("AUGMENTATION_DATASET_SAMPLES", "8"))
        triplets = _selected_triplets(leaf, sample_count)

        try:
            import numpy as np
            from PIL import Image
            import pyvista
        except ImportError as error:
            self.skipTest(f"Optional dataset dependencies unavailable: {error}")

        coordinate_storage = os.environ.get(
            "PLANTS_COORDINATES", "normalized"
        ).strip().lower()
        if coordinate_storage not in {"normalized", "voxel"}:
            raise ValueError("PLANTS_COORDINATES must be 'normalized' or 'voxel'")
        coordinate_order = os.environ.get(
            "PLANTS_COORDINATE_ORDER", "yx"
        ).strip().lower()
        if coordinate_order not in {"yx", "xy"}:
            raise ValueError("PLANTS_COORDINATE_ORDER must be 'yx' or 'xy'")

        minimum_hit_rate = float(
            os.environ.get("PLANTS_MIN_NODE_HIT_RATE", "0.80")
        )
        padding = int(os.environ.get("PLANTS_TEST_PADDING", "5"))
        if padding < 0:
            raise ValueError("PLANTS_TEST_PADDING must be non-negative")

        measurements = []
        print(f"\nPlants integration: {len(triplets)} evenly spaced patches")
        for raw_path, seg_path, graph_path in triplets:
            with self.subTest(sample=raw_path.name):
                # PIL conversion matches the legacy loader's grayscale handling.
                with Image.open(raw_path) as raw_image:
                    image_array = np.array(raw_image.convert("L"), copy=True)
                with Image.open(seg_path) as seg_image:
                    segmentation_array = np.array(seg_image.convert("L"), copy=True)

                image_2d = torch.as_tensor(
                    image_array, dtype=torch.float32
                ).unsqueeze(0) / 255.0
                segmentation_2d = torch.as_tensor(
                    segmentation_array, dtype=torch.float32
                ).unsqueeze(0) / 255.0
                segmentation_2d = (segmentation_2d >= 0.3).float()

                stored_points = torch.as_tensor(
                    np.asarray(pyvista.read(graph_path).points)[:, :2].copy(),
                    dtype=torch.float32,
                )
                height, width = (int(value) for value in segmentation_2d.shape[-2:])
                # generate_plants_data.py converts source (x,y) annotations to
                # (y,x) before clipping and saving VTP patches. Convert the
                # declared on-disk order back to (x,y) for the shared embedding
                # helper, which then emits model order (D,H,W)=(y,x,z).
                points_xy = _stored_points_to_xy(
                    stored_points,
                    coordinate_order=coordinate_order,
                    coordinate_storage=coordinate_storage,
                    height=height,
                    width=width,
                )

                z_position = 0.5
                image_3d = project_2d_to_3d(
                    image_2d,
                    z_position=z_position,
                    thickness=5,
                    depth=width,
                )
                segmentation_3d = project_2d_to_3d(
                    segmentation_2d,
                    z_position=z_position,
                    thickness=5,
                    depth=width,
                )
                nodes = embed_2d_coordinates(
                    points_xy,
                    z_position=z_position,
                )

                baseline_hit_rate = node_foreground_hit_rate(
                    segmentation_3d, nodes
                )
                baseline_neighbourhood_hit_rate = (
                    node_foreground_neighbourhood_hit_rate(
                        segmentation_3d,
                        nodes,
                        radius_voxels=1,
                    )
                )

                transformed = apply_augmentation(
                    image_3d,
                    segmentation_3d,
                    nodes,
                    AugmentationParameters(
                        quarter_turns=(1, 2, 3),
                        flip_axes=(True, True, True),
                    ),
                )
                rigid_hit_rate = node_foreground_hit_rate(
                    transformed.segmentation, transformed.nodes
                )
                rigid_neighbourhood_hit_rate = (
                    node_foreground_neighbourhood_hit_rate(
                        transformed.segmentation,
                        transformed.nodes,
                        radius_voxels=1,
                    )
                )

                source_shape = tuple(
                    int(value) for value in transformed.segmentation.shape[-3:]
                )
                target_shape = tuple(value + 2 * padding for value in source_shape)
                padded_segmentation = _symmetric_pad_3d(
                    transformed.segmentation,
                    padding,
                    value=-0.5,
                )
                padded_nodes = pad_coordinates(
                    transformed.nodes,
                    source_shape,
                    target_shape,
                )
                padded_hit_rate = node_foreground_hit_rate(
                    padded_segmentation, padded_nodes
                )
                padded_neighbourhood_hit_rate = (
                    node_foreground_neighbourhood_hit_rate(
                        padded_segmentation,
                        padded_nodes,
                        radius_voxels=1,
                    )
                )
                measurements.append(
                    (
                        baseline_hit_rate,
                        baseline_neighbourhood_hit_rate,
                        rigid_hit_rate,
                        rigid_neighbourhood_hit_rate,
                        padded_hit_rate,
                        padded_neighbourhood_hit_rate,
                    )
                )
                print(
                    f"  {raw_path.name}: baseline_exact={baseline_hit_rate:.3f}, "
                    f"baseline_within_1_voxel="
                    f"{baseline_neighbourhood_hit_rate:.3f}, "
                    f"rigid_exact={rigid_hit_rate:.3f}, "
                    f"rigid_within_1_voxel="
                    f"{rigid_neighbourhood_hit_rate:.3f}, "
                    f"padded_exact={padded_hit_rate:.3f}, "
                    f"padded_within_1_voxel="
                    f"{padded_neighbourhood_hit_rate:.3f}"
                )

                # Real graph nodes are continuous and often lie exactly between
                # raster pixels. At such a tie, discrete rounding need not commute
                # with a reflection, rotation, or normalization after padding.
                # Exact rates are therefore diagnostic; the local-neighbourhood
                # rates are the invariant alignment check for this raster dataset.
                self.assertGreaterEqual(
                    baseline_neighbourhood_hit_rate,
                    minimum_hit_rate,
                    "Plants graph nodes lie outside the local foreground "
                    "neighbourhood before augmentation",
                )
                self.assertGreaterEqual(
                    rigid_neighbourhood_hit_rate,
                    baseline_neighbourhood_hit_rate - 1.0e-6,
                    "Plants rotations/flips moved graph nodes outside their "
                    "original foreground neighbourhood",
                )
                self.assertGreaterEqual(
                    padded_neighbourhood_hit_rate,
                    rigid_neighbourhood_hit_rate - 1.0e-6,
                    "Symmetric padding moved Plants graph nodes outside their "
                    "original foreground neighbourhood",
                )

        if measurements:
            minima = [min(values) for values in zip(*measurements)]
            print(
                "Plants minima: "
                f"baseline_exact={minima[0]:.3f}, "
                f"baseline_within_1_voxel={minima[1]:.3f}, "
                f"rigid_exact={minima[2]:.3f}, "
                f"rigid_within_1_voxel={minima[3]:.3f}, "
                f"padded_exact={minima[4]:.3f}, "
                f"padded_within_1_voxel={minima[5]:.3f}"
            )


if __name__ == "__main__":
    unittest.main()
