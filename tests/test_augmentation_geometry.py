from __future__ import annotations

import itertools
import random
import unittest

import torch

from data.augmentations import (
    EVALUATION_POLICY,
    PLANTS_TRAIN_POLICY,
    SYNTHETIC_MRI_TRAIN_POLICY,
    AugmentationParameters,
    AugmentationPolicy,
    add_gaussian_noise,
    apply_augmentation,
    coordinates_to_voxel_indices,
    degrees_to_quarter_turns,
    embed_2d_coordinates,
    flip_coordinates,
    flip_volume,
    normalize_voxel_coordinates,
    pad_coordinates,
    policy_for_split,
    project_2d_to_3d,
    rotate_coordinates,
    rotate_volume,
    sample_augmentation,
    scale_intensity,
    zoom_coordinates,
    zoom_volume,
)


def _landmark_sample(
    shape: tuple[int, int, int],
    voxel: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.zeros((1, *shape), dtype=torch.float32)
    segmentation = torch.zeros_like(image)
    image[(0, *voxel)] = 1.0
    segmentation[(0, *voxel)] = 1.0
    voxel_tensor = torch.tensor([voxel], dtype=torch.float32)
    nodes = normalize_voxel_coordinates(voxel_tensor, shape)
    return image, segmentation, nodes


def _single_foreground_index(volume: torch.Tensor) -> torch.Tensor:
    locations = (volume[0] > 0.5).nonzero(as_tuple=False)
    if locations.shape[0] != 1:
        raise AssertionError(f"Expected one landmark, found {locations.tolist()}")
    return locations[0]


class CoordinateConventionTests(unittest.TestCase):
    def test_index_over_size_round_trip(self) -> None:
        shape = (7, 9, 11)
        voxels = torch.tensor([[0, 0, 0], [6, 8, 10], [3, 4, 5]], dtype=torch.float32)
        coordinates = normalize_voxel_coordinates(voxels, shape)
        recovered = coordinates_to_voxel_indices(coordinates, shape)
        self.assertTrue(torch.equal(recovered, voxels.long()))
        self.assertTrue(
            torch.allclose(
                coordinates[1], torch.tensor([6 / 7, 8 / 9, 10 / 11])
            )
        )

    def test_degrees_reject_fake_continuous_rotation(self) -> None:
        self.assertEqual(degrees_to_quarter_turns((90, 180, 270)), (1, 2, 3))
        with self.assertRaises(ValueError):
            degrees_to_quarter_turns((45, 0, 0))


class ExactGeometryTests(unittest.TestCase):
    def test_every_composed_rotation_keeps_landmark_and_node_aligned(self) -> None:
        shape = (7, 7, 7)
        image, segmentation, nodes = _landmark_sample(shape, (1, 3, 5))
        for turns in itertools.product(range(4), repeat=3):
            with self.subTest(turns=turns):
                rotated_image = rotate_volume(image, turns)
                rotated_segmentation = rotate_volume(segmentation, turns)
                rotated_nodes = rotate_coordinates(nodes, turns, shape)
                expected = _single_foreground_index(rotated_segmentation)
                node_index = coordinates_to_voxel_indices(
                    rotated_nodes, rotated_segmentation.shape[-3:]
                )[0]
                self.assertTrue(torch.equal(expected, node_index))
                self.assertTrue(torch.equal(rotated_image, rotated_segmentation))

    def test_rotation_supports_non_cubic_volumes(self) -> None:
        shape = (5, 7, 9)
        image, segmentation, nodes = _landmark_sample(shape, (1, 4, 7))
        turns = (1, 3, 1)
        rotated = rotate_volume(segmentation, turns)
        rotated_nodes = rotate_coordinates(nodes, turns, shape)
        self.assertTrue(
            torch.equal(
                _single_foreground_index(rotated),
                coordinates_to_voxel_indices(
                    rotated_nodes, rotated.shape[-3:]
                )[0],
            )
        )

    def test_each_flip_keeps_landmark_and_node_aligned(self) -> None:
        shape = (7, 9, 11)
        _, segmentation, nodes = _landmark_sample(shape, (1, 4, 8))
        for axes in itertools.product((False, True), repeat=3):
            with self.subTest(axes=axes):
                flipped = flip_volume(segmentation, axes)
                flipped_nodes = flip_coordinates(nodes, axes, shape)
                self.assertTrue(
                    torch.equal(
                        _single_foreground_index(flipped),
                        coordinates_to_voxel_indices(flipped_nodes, shape)[0],
                    )
                )

    def test_zoom_keeps_node_inside_transformed_foreground(self) -> None:
        shape = (17, 17, 17)
        centre = (5, 8, 11)
        segmentation = torch.zeros((1, *shape), dtype=torch.float32)
        segmentation[0, 4:7, 7:10, 10:13] = 1.0
        nodes = normalize_voxel_coordinates(
            torch.tensor([centre], dtype=torch.float32), shape
        )
        for factor in (0.6, 0.75, 0.9, 1.0):
            with self.subTest(factor=factor):
                zoomed = zoom_volume(segmentation, factor, mode="nearest")
                zoomed_nodes = zoom_coordinates(nodes, factor, shape)
                index = coordinates_to_voxel_indices(zoomed_nodes, shape)[0]
                self.assertGreater(float(zoomed[(0, *index.tolist())]), 0.5)
                self.assertTrue(
                    set(torch.unique(zoomed).tolist()).issubset({0.0, 1.0})
                )

    def test_composed_pipeline_keeps_geometry_aligned(self) -> None:
        shape = (17, 17, 17)
        centre = (5, 8, 11)
        image = torch.zeros((1, *shape), dtype=torch.float32)
        segmentation = torch.zeros_like(image)
        image[0, 4:7, 7:10, 10:13] = 1.0
        segmentation.copy_(image)
        nodes = normalize_voxel_coordinates(
            torch.tensor([centre], dtype=torch.float32), shape
        )
        params = AugmentationParameters(
            quarter_turns=(1, 2, 3),
            zoom_factor=0.75,
            flip_axes=(True, False, True),
        )
        result = apply_augmentation(image, segmentation, nodes, params)
        index = coordinates_to_voxel_indices(
            result.nodes, result.segmentation.shape[-3:]
        )[0]
        self.assertGreater(float(result.segmentation[(0, *index.tolist())]), 0.5)


class PolicyCompatibilityTests(unittest.TestCase):
    def test_canonical_policies_preserve_active_ranges(self) -> None:
        self.assertTrue(SYNTHETIC_MRI_TRAIN_POLICY.rotate_90)
        self.assertEqual(SYNTHETIC_MRI_TRAIN_POLICY.zoom_range, (0.6, 1.0))
        self.assertEqual(
            SYNTHETIC_MRI_TRAIN_POLICY.gaussian_noise_probability, 0.35
        )
        self.assertEqual(SYNTHETIC_MRI_TRAIN_POLICY.gaussian_noise_max_std, 0.015)
        self.assertEqual(SYNTHETIC_MRI_TRAIN_POLICY.clamp_range, (-0.5, 0.5))
        self.assertTrue(PLANTS_TRAIN_POLICY.rotate_90)
        self.assertEqual(PLANTS_TRAIN_POLICY.flip_probability, (0.5, 0.5, 0.5))

    def test_validation_and_test_policies_are_identity(self) -> None:
        for dataset in ("plants", "syntheticMRI"):
            for split in ("val", "validation", "test"):
                with self.subTest(dataset=dataset, split=split):
                    policy = policy_for_split(dataset, split)
                    params = sample_augmentation(policy, random.Random(123))
                    self.assertIs(policy, EVALUATION_POLICY)
                    self.assertTrue(params.is_identity)

    def test_sampling_is_reproducible(self) -> None:
        left = sample_augmentation(
            SYNTHETIC_MRI_TRAIN_POLICY, random.Random(364505)
        )
        right = sample_augmentation(
            SYNTHETIC_MRI_TRAIN_POLICY, random.Random(364505)
        )
        self.assertEqual(left, right)
        self.assertTrue(0.6 <= left.zoom_factor <= 1.0)
        self.assertTrue(0.0 <= left.noise_std <= 0.015)

    def test_disabled_policy_fields_do_not_advance_rng(self) -> None:
        observed_rng = random.Random(19)
        sample_augmentation(EVALUATION_POLICY, observed_rng)
        observed = observed_rng.random()
        expected = random.Random(19).random()
        self.assertEqual(observed, expected)

    def test_noise_is_reproducible_and_clamped(self) -> None:
        image = torch.full((1, 5, 5, 5), 0.49)
        first = add_gaussian_noise(
            image,
            mean=0.0,
            std=0.05,
            clamp_range=(-0.5, 0.5),
            generator=torch.Generator().manual_seed(10),
        )
        second = add_gaussian_noise(
            image,
            mean=0.0,
            std=0.05,
            clamp_range=(-0.5, 0.5),
            generator=torch.Generator().manual_seed(10),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertLessEqual(float(first.max()), 0.5)
        self.assertGreaterEqual(float(first.min()), -0.5)

    def test_training_clamp_runs_when_noise_is_not_selected(self) -> None:
        image = torch.tensor([[[[-0.75, 0.0, 0.75]]]], dtype=torch.float32)
        segmentation = torch.zeros_like(image)
        nodes = torch.empty((0, 3), dtype=torch.float32)
        parameters = AugmentationParameters(clamp_range=(-0.5, 0.5))

        result = apply_augmentation(image, segmentation, nodes, parameters)

        self.assertTrue(
            torch.equal(
                result.image,
                torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=torch.float32),
            )
        )
        self.assertFalse(parameters.is_identity)

    def test_policy_and_parameter_ranges_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            AugmentationPolicy(flip_probability=(0.5, 0.5))
        with self.assertRaises(ValueError):
            AugmentationPolicy(clamp_range=(0.5, -0.5))
        with self.assertRaises(ValueError):
            AugmentationParameters(flip_axes=(True, False))
        with self.assertRaises(ValueError):
            AugmentationParameters(zoom_factor=0.0)
        with self.assertRaises(ValueError):
            AugmentationParameters(noise_std=-0.1)

    def test_plant_projection_padding_and_scaling(self) -> None:
        image_2d = torch.zeros((1, 6, 6), dtype=torch.float32)
        image_2d[0, 2, 4] = 1.0
        projected = project_2d_to_3d(image_2d, depth=6, z_position=0.5)
        coordinates = embed_2d_coordinates(
            torch.tensor([[4 / 6, 2 / 6]], dtype=torch.float32),
            z_position=0.5,
        )
        padded_coordinates = pad_coordinates(coordinates, (6, 6, 6), (8, 8, 8))
        self.assertEqual(tuple(projected.shape), (1, 6, 6, 6))
        self.assertEqual(
            coordinates_to_voxel_indices(padded_coordinates, (8, 8, 8))[0].tolist(),
            [3, 5, 4],
        )
        scaled = scale_intensity(torch.tensor([0.0, 1.0, 2.0]))
        self.assertTrue(torch.allclose(scaled, torch.tensor([-0.5, 0.0, 0.5])))


if __name__ == "__main__":
    unittest.main()
