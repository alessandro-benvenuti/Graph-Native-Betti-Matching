"""Optional end-to-end checks against the real Magnolia datasets."""

from __future__ import annotations

import os
from pathlib import Path
import random
import unittest

import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from data.loaders import (
    PlantsDataset,
    SyntheticMRIDataset,
    discover_plants,
    discover_synthetic_mri,
)
from data.loaders.io import read_vtp_graph
from tests.integration_helpers import (
    node_foreground_neighbourhood_hit_rate,
    select_evenly_spaced,
)


def _sample_count() -> int:
    value = int(os.environ.get("AUGMENTATION_DATASET_SAMPLES", "8"))
    if value <= 0:
        raise ValueError("AUGMENTATION_DATASET_SAMPLES must be positive")
    return value


@unittest.skipUnless(
    os.environ.get("SYNTHETIC_MRI_DATASET"),
    "Set SYNTHETIC_MRI_DATASET to run the SyntheticMRI loader integration test",
)
class SyntheticMRILoaderIntegrationTests(unittest.TestCase):
    def test_loading_and_active_augmentation_preserve_local_alignment(self) -> None:
        root = Path(os.environ["SYNTHETIC_MRI_DATASET"])
        records = select_evenly_spaced(
            discover_synthetic_mri(root, "train"), _sample_count()
        )
        coordinate_space = os.environ.get(
            "SYNTHETIC_MRI_COORDINATES", "normalized"
        )
        minimum = float(os.environ.get("AUGMENTATION_MIN_NODE_HIT_RATE", "0.80"))
        rates = []
        try:
            dataset = SyntheticMRIDataset(
                records,
                coordinate_space=coordinate_space,
                foreground_mean=0.33335259556770325,
                augment=True,
            )
            for sample in dataset:
                segmentation, nodes = sample[1][0], sample[2][0]
                rates.append(
                    node_foreground_neighbourhood_hit_rate(
                        segmentation, nodes, radius_voxels=1
                    )
                )
        except ImportError as error:
            self.skipTest(f"Optional dataset dependencies unavailable: {error}")
        print(f"\nSyntheticMRI loader minimum within one voxel: {min(rates):.3f}")
        self.assertGreaterEqual(min(rates), minimum - 0.05)


@unittest.skipUnless(
    os.environ.get("PLANTS_DATASET"),
    "Set PLANTS_DATASET to run the Plants loader integration test",
)
class PlantsLoaderIntegrationTests(unittest.TestCase):
    def test_loading_projection_augmentation_and_padding_preserve_alignment(self) -> None:
        root = Path(os.environ["PLANTS_DATASET"])
        # Exercise the same filesystem order used by production sample caps.
        records = select_evenly_spaced(
            discover_plants(root, "train"), _sample_count()
        )
        minimum = float(os.environ.get("PLANTS_MIN_NODE_HIT_RATE", "0.80"))
        baseline_rates = []
        augmented_rates = []
        try:
            baseline_dataset = PlantsDataset(
                records,
                augment=False,
                rng=random.Random(0),
            )
            augmented_dataset = PlantsDataset(
                records,
                augment=True,
                rng=random.Random(0),
            )
            for record, baseline, augmented in zip(
                records, baseline_dataset, augmented_dataset
            ):
                baseline_seg, baseline_nodes = baseline[1][0], baseline[2][0]
                augmented_seg, augmented_nodes = augmented[1][0], augmented[2][0]
                baseline_rate = node_foreground_neighbourhood_hit_rate(
                    baseline_seg, baseline_nodes, radius_voxels=1
                )
                augmented_rate = node_foreground_neighbourhood_hit_rate(
                    augmented_seg, augmented_nodes, radius_voxels=1
                )
                baseline_rates.append(baseline_rate)
                augmented_rates.append(augmented_rate)
                if baseline_rate < minimum or augmented_rate < baseline_rate - 1.0e-6:
                    stored_points, _ = read_vtp_graph(record.graph)
                    z_position = float(baseline[4][0])
                    no_swap_nodes = torch.cat(
                        (
                            stored_points[:, :2].float(),
                            stored_points.new_full(
                                (stored_points.shape[0], 1), z_position
                            ),
                        ),
                        dim=1,
                    )
                    no_swap_nodes = (
                        no_swap_nodes
                        * (baseline_dataset.size - 2 * baseline_dataset.padding)
                        + baseline_dataset.padding
                    ) / baseline_dataset.size
                    no_swap_rate = node_foreground_neighbourhood_hit_rate(
                        baseline_seg, no_swap_nodes, radius_voxels=1
                    )

                    # Determine whether alignment is lost by MedPy decoding or
                    # by the established MONAI resize + 0.3 threshold stage.
                    native_seg_2d = (
                        baseline_dataset.image_reader(record.segmentation).float()
                        / 255.0
                    ).unsqueeze(0)
                    native_seg_2d = (native_seg_2d >= 0.3).float()
                    diagnostic_z = 0.5
                    native_seg_3d = baseline_dataset._project_image_3d(
                        native_seg_2d, diagnostic_z
                    )
                    swapped_native_nodes = torch.cat(
                        (
                            stored_points[:, [1, 0]].float(),
                            stored_points.new_full(
                                (stored_points.shape[0], 1), diagnostic_z
                            ),
                        ),
                        dim=1,
                    )
                    native_rate = node_foreground_neighbourhood_hit_rate(
                        native_seg_3d,
                        swapped_native_nodes,
                        radius_voxels=1,
                    )
                    no_swap_native_nodes = torch.cat(
                        (
                            stored_points[:, :2].float(),
                            stored_points.new_full(
                                (stored_points.shape[0], 1), diagnostic_z
                            ),
                        ),
                        dim=1,
                    )
                    native_no_swap_rate = node_foreground_neighbourhood_hit_rate(
                        native_seg_3d,
                        no_swap_native_nodes,
                        radius_voxels=1,
                    )

                    resized_seg_2d = (
                        baseline_dataset.image_reader(record.segmentation).float()
                        / 255.0
                    )
                    resized_seg_2d = (resized_seg_2d >= 0.3).float().unsqueeze(0)
                    resized_seg_2d = baseline_dataset.resize_segmentation(
                        resized_seg_2d
                    )
                    resized_seg_3d = baseline_dataset._project_image_3d(
                        resized_seg_2d, diagnostic_z
                    )
                    resized_rate = node_foreground_neighbourhood_hit_rate(
                        resized_seg_3d,
                        swapped_native_nodes,
                        radius_voxels=1,
                    )
                    print(
                        f"\n  {record.image.name}: baseline={baseline_rate:.3f}, "
                        f"augmented={augmented_rate:.3f}, "
                        f"alternative_without_yx_swap={no_swap_rate:.3f}, "
                        f"medpy_before_resize={native_rate:.3f}, "
                        f"medpy_before_resize_without_yx_swap="
                        f"{native_no_swap_rate:.3f}, "
                        f"after_resize_threshold={resized_rate:.3f}"
                    )
        except ImportError as error:
            self.skipTest(f"Optional dataset dependencies unavailable: {error}")
        print(
            "\nPlants loader minima within one voxel: "
            f"baseline={min(baseline_rates):.3f}, "
            f"augmented={min(augmented_rates):.3f}"
        )
        for record, baseline_rate, augmented_rate in zip(
            records, baseline_rates, augmented_rates
        ):
            self.assertGreaterEqual(
                baseline_rate,
                minimum,
                f"Loader preprocessing misaligned {record.image.name}",
            )
            self.assertGreaterEqual(
                augmented_rate,
                baseline_rate - 1.0e-6,
                f"Augmentation worsened graph alignment for {record.image.name}",
            )


if __name__ == "__main__":
    unittest.main()
