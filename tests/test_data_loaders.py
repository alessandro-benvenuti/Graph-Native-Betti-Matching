"""Dependency-light tests for configuration, discovery, datasets, and batching."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs import ConfigError, load_config
from data.augmentations import coordinates_to_voxel_indices, scale_intensity
from data.loaders import (
    DistributedWeightedSampler,
    PlantsDataset,
    SamplePaths,
    SyntheticMRIDataset,
    build_datasets,
    build_evaluation_loader,
    build_synthetic_mri_dataset,
    compose_source_target,
    dataset_sample_manifest,
    discover_plants,
    discover_synthetic_mri,
    image_graph_collate,
)
from data.loaders.mixed import _supports_keyword
from tests.integration_helpers import node_foreground_neighbourhood_hit_rate


def _pad_volume(volume: torch.Tensor, shape, value: float = -0.5) -> torch.Tensor:
    source = volume.shape[-3:]
    padding = []
    for current, target in reversed(tuple(zip(source, shape))):
        total = target - current
        padding.extend((total // 2, total - total // 2))
    return F.pad(volume, tuple(padding), value=value)


def _foreground_hit(segmentation: torch.Tensor, nodes: torch.Tensor) -> float:
    indices = coordinates_to_voxel_indices(nodes, segmentation.shape[-3:])
    values = segmentation[0, indices[:, 0], indices[:, 1], indices[:, 2]]
    return float((values > 0).float().mean()) if values.numel() else 1.0


def _resize_area(shape):
    return lambda image: F.interpolate(
        image.unsqueeze(0), size=shape, mode="area"
    ).squeeze(0)


def _test_scale(image: torch.Tensor) -> torch.Tensor:
    return scale_intensity(image, (-0.5, 0.5))


def _test_pad(shape):
    return lambda volume: _pad_volume(volume, shape)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_environment_and_overrides_are_resolved(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs" / "pretrain_mixed.yaml",
            environment={
                "GNBM_OUTPUT_DIR": "/outputs",
                "PLANTS_DATASET": "/plants",
                "SYNTHETIC_MRI_DATASET": "/synthetic",
            },
        )
        self.assertEqual(config["experiment"]["name"], "pretrain_mixed")
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertEqual(config["model"]["decoder"]["object_queries"], 120)
        self.assertEqual(config["data"]["datasets"]["plants"]["root"], "/plants")
        self.assertTrue(
            config["data"]["mixed_sampling"]["balance_source_target"]
        )
        self.assertTrue(config["loss"]["supervise_target_graphs"])
        self.assertNotIn("domain_adaptation", config)
        self.assertNotIn("domain_lr", config["training"]["optimizer"])
        self.assertEqual(
            config["evaluation"]["protocol"]["iou_thresholds"],
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        )

    def test_missing_environment_variable_is_reported_at_its_location(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ConfigError, "PLANTS_DATASET"):
            load_config(
                root / "configs" / "pretrain_mixed.yaml",
                environment={
                    "GNBM_OUTPUT_DIR": "/outputs",
                    "SYNTHETIC_MRI_DATASET": "/synthetic",
                },
            )


class DataLoaderCompatibilityTests(unittest.TestCase):
    def test_collation_removes_tensor_subclass_metadata(self) -> None:
        class MetadataTensor(torch.Tensor):
            @staticmethod
            def wrap(value):
                return torch.Tensor._make_subclass(
                    MetadataTensor, value, value.requires_grad
                )

            def as_tensor(self):
                return self.as_subclass(torch.Tensor)

        image = MetadataTensor.wrap(torch.ones((1, 3, 3, 3)))
        item = (
            [image],
            [image.clone()],
            [torch.zeros((1, 3))],
            [torch.empty((0, 2), dtype=torch.long)],
            [None],
            [1],
        )

        images, segmentations, *_ = image_graph_collate([item])

        self.assertIs(type(images), torch.Tensor)
        self.assertIs(type(segmentations), torch.Tensor)

    def test_generator_is_only_passed_to_versions_that_support_it(self) -> None:
        class LegacyLoader:
            def __init__(self, dataset, batch_size=1):
                pass

        class ModernLoader:
            def __init__(self, dataset, batch_size=1, generator=None):
                pass

        self.assertFalse(_supports_keyword(LegacyLoader.__init__, "generator"))
        self.assertTrue(_supports_keyword(ModernLoader.__init__, "generator"))

    def test_distributed_weighted_sampler_shards_one_global_draw(self) -> None:
        samplers = [
            DistributedWeightedSampler(
                [1.0, 2.0, 3.0, 4.0],
                7,
                num_replicas=2,
                rank=rank,
                seed=19,
            )
            for rank in range(2)
        ]
        shards = [list(sampler) for sampler in samplers]
        self.assertEqual([len(shard) for shard in shards], [4, 4])
        interleaved = [value for pair in zip(*shards) for value in pair]

        reference = torch.multinomial(
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double),
            7,
            replacement=True,
            generator=torch.Generator().manual_seed(19),
        ).tolist()
        self.assertEqual(interleaved[:7], reference)

        for sampler in samplers:
            sampler.set_epoch(1)
        self.assertNotEqual(shards, [list(sampler) for sampler in samplers])


class DiscoveryTests(unittest.TestCase):
    @staticmethod
    def _folders(root: Path, split: str = "train") -> Path:
        leaf = root / split
        for name in ("raw", "seg", "vtp"):
            (leaf / name).mkdir(parents=True, exist_ok=True)
        return leaf

    def test_synthetic_discovery_supports_both_extensions_and_requires_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            leaf = self._folders(Path(directory))
            for sample_id, extension in (("sample_b", ".nii"), ("sample_a", ".nii.gz")):
                (leaf / "raw" / f"{sample_id}_data{extension}").touch()
                (leaf / "seg" / f"{sample_id}_seg{extension}").touch()
                (leaf / "vtp" / f"{sample_id}_graph.vtp").touch()
            records = discover_synthetic_mri(Path(directory), "train")
            self.assertEqual([record.sample_id for record in records], ["sample_a", "sample_b"])
            (leaf / "vtp" / "sample_b_graph.vtp").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete sample"):
                discover_synthetic_mri(Path(directory), "train")

    def test_configured_split_discovery_does_not_reuse_a_direct_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("raw", "seg", "vtp"):
                (root / name).mkdir()
            (root / "raw" / "sample_data.png").touch()
            (root / "seg" / "sample_seg.png").touch()
            (root / "vtp" / "sample_graph.vtp").touch()
            self.assertEqual(len(discover_plants(root, "val")), 1)
            with self.assertRaises(FileNotFoundError):
                discover_plants(root, "val", allow_direct=False)

    def test_seeded_mri_cap_has_stable_random_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = self._folders(root)
            sample_ids = [f"sample_{index:03d}" for index in range(20)]
            for sample_id in sample_ids:
                (leaf / "raw" / f"{sample_id}_data.nii.gz").touch()
                (leaf / "seg" / f"{sample_id}_seg.nii.gz").touch()
                (leaf / "vtp" / f"{sample_id}_graph.vtp").touch()

            options = dict(
                root=root,
                split="train",
                max_samples=5,
                sample_cap_selection="seeded_random",
                sample_cap_seed=364505,
                augment=False,
            )
            first = build_synthetic_mri_dataset(**options)
            second = build_synthetic_mri_dataset(**options)
            selected = [record.sample_id for record in first.records]
            self.assertEqual(
                selected, [record.sample_id for record in second.records]
            )

            def key(sample_id):
                payload = f"364505\0train\0{sample_id}".encode("utf-8")
                return (hashlib.sha256(payload).digest(), sample_id)

            expected = sorted(sorted(sample_ids, key=key)[:5])
            self.assertEqual(selected, expected)
            self.assertNotEqual(selected, sample_ids[:5])
            manifest = dataset_sample_manifest(first)
            self.assertEqual(manifest[0]["sample_ids"], expected)


class SyntheticMRILoaderTests(unittest.TestCase):
    @staticmethod
    def _record() -> SamplePaths:
        return SamplePaths(Path("image"), Path("seg"), Path("graph"), "synthetic")

    def test_voxel_coordinates_and_augmentation_stay_aligned(self) -> None:
        image = torch.zeros((5, 5, 5))
        segmentation = torch.zeros((5, 5, 5))
        image[1, 2, 3] = 1.0
        segmentation[1, 2, 3] = 1.0

        def volume_reader(path: Path) -> torch.Tensor:
            return segmentation.clone() if path.name == "seg" else image.clone()

        rng = Mock()
        rng.randint.side_effect = [1, 2, 3]
        rng.uniform.return_value = 0.8
        dataset = SyntheticMRIDataset(
            [self._record()],
            image_size=(5, 5, 5),
            coordinate_space="voxel",
            augment=True,
            domain_label=1,
            volume_reader=volume_reader,
            graph_reader=lambda path: (
                torch.tensor([[1.0, 2.0, 3.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
            intensity_transform=lambda value: value,
            rng=rng,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample[0][0].shape), (1, 5, 5, 5))
        self.assertEqual(_foreground_hit(sample[1][0], sample[2][0]), 1.0)
        self.assertEqual(sample[5], [1])

    def test_validation_is_identity_and_does_not_clamp(self) -> None:
        volume = torch.ones((3, 3, 3)) * 2.0
        dataset = SyntheticMRIDataset(
            [self._record()],
            image_size=(3, 3, 3),
            foreground_mean=0.5,
            augment=False,
            volume_reader=lambda path: volume.clone(),
            graph_reader=lambda path: (
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
        )
        self.assertAlmostEqual(float(dataset[0][0][0].max()), 1.5)


class PlantsLoaderTests(unittest.TestCase):
    @staticmethod
    def _record() -> SamplePaths:
        return SamplePaths(Path("image"), Path("seg"), Path("graph"), "plant")

    def test_yx_storage_projects_augments_and_pads_correctly(self) -> None:
        image = torch.zeros((5, 5))
        segmentation = torch.zeros((5, 5))
        image[1, 3] = segmentation[1, 3] = 255.0

        def image_reader(path: Path) -> torch.Tensor:
            return segmentation.clone() if path.name == "seg" else image.clone()

        rng = Mock()
        rng.randint.side_effect = [1, 2, 3]
        rng.random.side_effect = [0.0, 0.0, 0.0]
        dataset = PlantsDataset(
            [self._record()],
            size=7,
            padding=1,
            augment=True,
            image_reader=image_reader,
            graph_reader=lambda path: (
                torch.tensor([[3.0 / 5.0, 1.0 / 5.0, 0.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
            resize_transform=lambda value: value,
            scale_transform=_test_scale,
            pad_transform=_test_pad((7, 7, 7)),
            rng=rng,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample[0][0].shape), (1, 7, 7, 7))
        self.assertEqual(_foreground_hit(sample[1][0], sample[2][0]), 1.0)

    def test_validation_does_not_sample_random_geometry(self) -> None:
        image = torch.zeros((5, 5))
        rng = Mock()
        dataset = PlantsDataset(
            [self._record()],
            size=7,
            padding=1,
            augment=False,
            image_reader=lambda path: image.clone(),
            graph_reader=lambda path: (
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
            resize_transform=lambda value: value,
            scale_transform=_test_scale,
            pad_transform=_test_pad((7, 7, 7)),
            rng=rng,
        )
        dataset[0]
        rng.randint.assert_not_called()
        rng.random.assert_not_called()

    def test_occupancy_resize_preserves_foreground_erased_by_area_threshold(self) -> None:
        image = torch.zeros((12, 12))
        image[1, 7] = 255.0
        area_resized = _resize_area((6, 6))((image / 255.0).unsqueeze(0))
        self.assertEqual(int((area_resized >= 0.3).sum()), 0)
        dataset = PlantsDataset(
            [self._record()],
            size=8,
            padding=1,
            augment=False,
            image_reader=lambda path: image.clone(),
            graph_reader=lambda path: (
                torch.tensor([[7.0 / 12.0, 1.0 / 12.0, 0.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
            resize_transform=_resize_area((6, 6)),
            scale_transform=_test_scale,
            pad_transform=_test_pad((8, 8, 8)),
        )
        sample = dataset[0]
        self.assertEqual(
            node_foreground_neighbourhood_hit_rate(
                sample[1][0], sample[2][0], radius_voxels=1
            ),
            1.0,
        )


class CompositionTests(unittest.TestCase):
    class _SizedDataset(Dataset):
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return index

    def test_target_balancing_preserves_the_original_epoch_size(self) -> None:
        dataset, sampler = compose_source_target(
            self._SizedDataset(10),
            self._SizedDataset(4),
            balance_source_target=True,
        )
        self.assertEqual(len(dataset), 14)
        self.assertEqual(sampler.num_samples, 20)
        self.assertTrue(torch.allclose(sampler.weights[10:], torch.full((4,), 2.5, dtype=torch.double)))

    def test_collate_preserves_six_part_model_contract(self) -> None:
        record = SamplePaths(Path("image"), Path("seg"), Path("graph"), "one")
        dataset = SyntheticMRIDataset(
            [record, record],
            image_size=(2, 2, 2),
            augment=False,
            volume_reader=lambda path: torch.ones((2, 2, 2)),
            graph_reader=lambda path: (
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.empty((0, 2), dtype=torch.long),
            ),
        )
        batch = image_graph_collate([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch[0].shape), (2, 1, 2, 2, 2))
        self.assertEqual(batch[4], [None, None])
        self.assertEqual(tuple(batch[5].shape), (2,))

    def test_real_yaml_builds_strict_split_aware_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plants_root = root / "plants"
            synthetic_root = root / "synthetic"
            for dataset_root, extension in ((plants_root, "png"), (synthetic_root, "nii.gz")):
                for split in ("train", "val"):
                    for folder in ("raw", "seg", "vtp"):
                        (dataset_root / split / folder).mkdir(parents=True)
                    for index in range(6):
                        sample = f"sample_{index:06d}"
                        (dataset_root / split / "raw" / f"{sample}_data.{extension}").touch()
                        (dataset_root / split / "seg" / f"{sample}_seg.{extension}").touch()
                        (dataset_root / split / "vtp" / f"{sample}_graph.vtp").touch()

            repository = Path(__file__).resolve().parents[1]
            config = load_config(
                repository / "configs" / "pretrain_mixed.yaml",
                environment={
                    "GNBM_OUTPUT_DIR": str(root / "output"),
                    "PLANTS_DATASET": str(plants_root),
                    "SYNTHETIC_MRI_DATASET": str(synthetic_root),
                },
            )
            plants = config["data"]["datasets"]["plants"]
            synthetic = config["data"]["datasets"]["synthetic_mri"]
            plants.update(train_samples=4, validation_samples=2)
            synthetic.update(train_samples=3, validation_samples=2)
            with patch(
                "data.loaders.plants._build_monai_transforms",
                return_value=(lambda x: x, lambda x: x, lambda x: x),
            ), patch(
                "data.loaders.synthetic_mri._build_training_intensity_transform",
                return_value=lambda x: x,
            ):
                train, validation, sampler = build_datasets(config)
            self.assertEqual((len(train), len(validation), sampler.num_samples), (7, 4, 8))
            self.assertTrue(train.datasets[0].datasets[0].augment)
            self.assertTrue(train.datasets[1].datasets[0].augment)
            self.assertFalse(validation.datasets[0].datasets[0].augment)
            self.assertFalse(validation.datasets[1].datasets[0].augment)

    def test_null_sample_caps_use_every_discovered_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plants_root = root / "plants"
            synthetic_root = root / "synthetic"
            for dataset_root, extension in (
                (plants_root, "png"),
                (synthetic_root, "nii.gz"),
            ):
                for split in ("train", "val"):
                    for folder in ("raw", "seg", "vtp"):
                        (dataset_root / split / folder).mkdir(parents=True)
                    for index in range(6):
                        sample = f"sample_{index:06d}"
                        (dataset_root / split / "raw" / f"{sample}_data.{extension}").touch()
                        (dataset_root / split / "seg" / f"{sample}_seg.{extension}").touch()
                        (dataset_root / split / "vtp" / f"{sample}_graph.vtp").touch()

            repository = Path(__file__).resolve().parents[1]
            config = load_config(
                repository
                / "configs"
                / "experiments"
                / "focal_matrix_600"
                / "pretrain_baseline.yaml",
                environment={
                    "GNBM_OUTPUT_DIR": str(root / "output"),
                    "PLANTS_DATASET": str(plants_root),
                    "SYNTHETIC_MRI_DATASET": str(synthetic_root),
                },
            )
            with patch(
                "data.loaders.plants._build_monai_transforms",
                return_value=(lambda x: x, lambda x: x, lambda x: x),
            ), patch(
                "data.loaders.synthetic_mri._build_training_intensity_transform",
                return_value=lambda x: x,
            ):
                train, validation, sampler = build_datasets(config)
            self.assertEqual(
                (len(train), len(validation), sampler.num_samples),
                (12, 12, 12),
            )

    def test_evaluation_loader_uses_requested_split_cap_and_no_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "synthetic"
            for folder in ("raw", "seg", "vtp"):
                (dataset_root / "test" / folder).mkdir(parents=True)
            for index in range(3):
                sample = f"sample_{index:06d}"
                (dataset_root / "test" / "raw" / f"{sample}_data.nii.gz").touch()
                (dataset_root / "test" / "seg" / f"{sample}_seg.nii.gz").touch()
                (dataset_root / "test" / "vtp" / f"{sample}_graph.vtp").touch()
            repository = Path(__file__).resolve().parents[1]
            config = load_config(
                repository / "configs" / "finetune_synthetic_mri.yaml",
                environment={
                    "GNBM_OUTPUT_DIR": str(root / "output"),
                    "SYNTHETIC_MRI_DATASET": str(dataset_root),
                },
            )
            config["runtime"]["workers"] = 0
            loader = build_evaluation_loader(
                config,
                dataset_name="synthetic_mri",
                split="test",
                max_samples=2,
            )
            self.assertEqual(len(loader.dataset), 2)
            self.assertFalse(loader.dataset.augment)
            self.assertEqual(
                [record.sample_id for record in loader.dataset.records],
                ["sample_000000", "sample_000001"],
            )


if __name__ == "__main__":
    unittest.main()
