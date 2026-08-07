"""Optional integration test against existing syntheticMRI patch data.

Set ``SYNTHETIC_MRI_DATASET`` to a dataset root containing either
``train/{raw,seg,vtp}`` or directly ``{raw,seg,vtp}``.  This test is skipped
when the variable is absent, so the exact geometry suite remains lightweight.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

# PyVista imports Matplotlib even though this test never renders. Force a
# non-interactive backend before that import on headless cluster nodes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import torch

from data.augmentations import (
    AugmentationParameters,
    apply_augmentation,
    normalize_voxel_coordinates,
)
from tests.integration_helpers import (
    node_foreground_hit_rate,
    node_foreground_neighbourhood_hit_rate,
    select_evenly_spaced,
)


DATASET_ROOT = os.environ.get("SYNTHETIC_MRI_DATASET")


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
    raw_files = sorted((leaf / "raw").glob("*_data.nii*"))
    if not raw_files:
        raise FileNotFoundError(f"No *_data.nii or *_data.nii.gz files in {leaf / 'raw'}")
    triplets = []
    for raw_path in select_evenly_spaced(raw_files, sample_count):
        name = raw_path.name
        seg_name = name.replace("_data.nii.gz", "_seg.nii.gz").replace(
            "_data.nii", "_seg.nii"
        )
        graph_name = name.replace("_data.nii.gz", "_graph.vtp").replace(
            "_data.nii", "_graph.vtp"
        )
        seg_path = leaf / "seg" / seg_name
        graph_path = leaf / "vtp" / graph_name
        if not seg_path.exists() or not graph_path.exists():
            raise FileNotFoundError(
                f"Could not pair {raw_path.name} with "
                f"{seg_path.name} and {graph_path.name}"
            )
        triplets.append((raw_path, seg_path, graph_path))
    return triplets


@unittest.skipUnless(
    DATASET_ROOT,
    "Set SYNTHETIC_MRI_DATASET to run integration checks on existing patches",
)
class ExistingSyntheticDatasetTests(unittest.TestCase):
    def test_graph_alignment_survives_active_geometric_pipeline(self) -> None:
        # Validate the path before checking optional readers, so a bad dataset
        # root is never disguised as a missing-package skip.
        leaf = _resolve_leaf(Path(DATASET_ROOT))
        sample_count = int(os.environ.get("AUGMENTATION_DATASET_SAMPLES", "8"))
        triplets = _selected_triplets(leaf, sample_count)

        try:
            import numpy as np
            import pyvista
        except ImportError as error:
            self.skipTest(f"Optional dataset dependencies unavailable: {error}")

        try:
            import nibabel as nib

            def load_volume(path: Path):
                return np.asarray(nib.load(str(path)).get_fdata())

            volume_reader = "nibabel"
        except ImportError:
            try:
                from medpy.io import load as medpy_load
            except ImportError as error:
                self.skipTest(
                    "A NIfTI reader is required; install nibabel or medpy "
                    f"({error})"
                )

            def load_volume(path: Path):
                volume, _ = medpy_load(str(path))
                return np.asarray(volume)

            volume_reader = "medpy"

        coordinate_storage = os.environ.get(
            "SYNTHETIC_MRI_COORDINATES", "normalized"
        ).strip().lower()
        if coordinate_storage not in {"normalized", "voxel"}:
            raise ValueError(
                "SYNTHETIC_MRI_COORDINATES must be 'normalized' or 'voxel'"
            )
        minimum_hit_rate = float(
            os.environ.get("AUGMENTATION_MIN_NODE_HIT_RATE", "0.80")
        )
        measurements = []
        print(
            f"\nSyntheticMRI integration: {len(triplets)} evenly spaced patches "
            f"(NIfTI reader: {volume_reader})"
        )
        for raw_path, seg_path, graph_path in triplets:
            with self.subTest(sample=raw_path.name):
                image = torch.as_tensor(
                    load_volume(raw_path), dtype=torch.float32
                ).unsqueeze(0)
                segmentation = torch.as_tensor(
                    load_volume(seg_path), dtype=torch.float32
                ).unsqueeze(0)
                nodes = torch.as_tensor(
                    np.asarray(pyvista.read(graph_path).points),
                    dtype=torch.float32,
                )
                shape = tuple(int(value) for value in segmentation.shape[-3:])
                if coordinate_storage == "voxel":
                    nodes = normalize_voxel_coordinates(nodes, shape)

                baseline_hit_rate = node_foreground_hit_rate(segmentation, nodes)
                self.assertGreaterEqual(
                    baseline_hit_rate,
                    minimum_hit_rate,
                    "The unaugmented patch is already misaligned; "
                    "inspect its coordinate convention",
                )

                rigid = apply_augmentation(
                    image,
                    segmentation,
                    nodes,
                    AugmentationParameters(
                        quarter_turns=(1, 2, 3),
                        flip_axes=(True, False, True),
                    ),
                )
                rigid_hit_rate = node_foreground_hit_rate(
                    rigid.segmentation, rigid.nodes
                )
                self.assertGreaterEqual(
                    rigid_hit_rate,
                    baseline_hit_rate - 1.0e-6,
                    "Exact rotations/flips moved graph nodes away from "
                    "their segmentation",
                )

                transformed = apply_augmentation(
                    image,
                    segmentation,
                    nodes,
                    AugmentationParameters(
                        quarter_turns=(1, 2, 3),
                        zoom_factor=0.8,
                        flip_axes=(True, False, True),
                    ),
                )
                zoom_exact = node_foreground_hit_rate(
                    transformed.segmentation, transformed.nodes
                )
                zoom_near = node_foreground_neighbourhood_hit_rate(
                    transformed.segmentation,
                    transformed.nodes,
                    radius_voxels=1,
                )
                self.assertGreaterEqual(
                    zoom_near,
                    baseline_hit_rate - 0.05,
                    "Zoom moved graph nodes outside the local segmentation "
                    "neighbourhood",
                )
                measurements.append(
                    (baseline_hit_rate, rigid_hit_rate, zoom_exact, zoom_near)
                )
                print(
                    f"  {raw_path.name}: baseline={baseline_hit_rate:.3f}, "
                    f"rigid_exact={rigid_hit_rate:.3f}, "
                    f"zoom_exact={zoom_exact:.3f}, "
                    f"zoom_within_1_voxel={zoom_near:.3f}"
                )

        minima = [min(values) for values in zip(*measurements)]
        print(
            "SyntheticMRI minima: "
            f"baseline={minima[0]:.3f}, rigid_exact={minima[1]:.3f}, "
            f"zoom_exact={minima[2]:.3f}, "
            f"zoom_within_1_voxel={minima[3]:.3f}"
        )


if __name__ == "__main__":
    unittest.main()
