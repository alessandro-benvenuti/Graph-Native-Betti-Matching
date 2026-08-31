"""Regression checks for preserving MRI intensity information during training."""

from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

from data.augmentations import (
    AugmentationParameters,
    SYNTHETIC_MRI_TRAIN_POLICY,
    apply_augmentation,
)
from data.loaders import SamplePaths, SyntheticMRIDataset
from data.loaders.synthetic_mri import _build_training_intensity_transform


class MRIIntensityTests(unittest.TestCase):
    def test_standalone_pipeline_preserves_values_without_noise(self):
        image = torch.tensor([[[[-0.75, 0.1, 0.567, 0.667]]]])
        result = apply_augmentation(
            image, torch.zeros_like(image), torch.empty((0, 3)),
            AugmentationParameters(clamp_range=SYNTHETIC_MRI_TRAIN_POLICY.clamp_range),
        )
        self.assertTrue(torch.equal(result.image, image))

    def test_standalone_pipeline_preserves_noise_tails(self):
        image = torch.linspace(-0.7, 0.7, 128).reshape(1, 4, 4, 8)
        seed = 42
        noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(seed))
        expected = image + noise * 0.2
        result = apply_augmentation(
            image, torch.zeros_like(image), torch.empty((0, 3)),
            AugmentationParameters(
                add_noise=True, noise_std=0.2,
                clamp_range=SYNTHETIC_MRI_TRAIN_POLICY.clamp_range,
            ),
            noise_generator=torch.Generator().manual_seed(seed),
        )
        self.assertTrue(torch.equal(result.image, expected))
        self.assertGreater(float(result.image.max()), 0.5)
        self.assertLess(float(result.image.min()), -0.5)

    def test_loader_matches_evaluation_and_does_not_clip_noise(self):
        # Exercise the real loader/transform construction without requiring MONAI.
        # The noise substitute makes both selected and unselected cases exact.
        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms

            def __call__(self, image):
                for transform in self.transforms:
                    image = transform(image)
                return image

        class Noise:
            def __init__(self, prob, std, mean):
                self.prob = prob

            def __call__(self, image):
                return image + 0.25 if self.prob else image

        monai = ModuleType("monai")
        transforms = ModuleType("monai.transforms")
        transforms.Compose = Compose
        transforms.RandGaussianNoise = Noise
        transforms.Lambda = lambda function: function
        monai.transforms = transforms
        raw = torch.tensor([0.0, 0.8, 0.9, 1.0]).reshape(1, 1, 4)
        mean = 0.332761968904616
        common = dict(
            records=[SamplePaths(Path("raw"), Path("seg"), Path("graph"), "sample")],
            image_size=(1, 1, 4), foreground_mean=mean,
            rotate_90=False, zoom_range=None,
            volume_reader=lambda path: raw.clone(),
            graph_reader=lambda path: (torch.empty((0, 3)), torch.empty((0, 2), dtype=torch.long)),
        )
        evaluation = SyntheticMRIDataset(**common, augment=False)[0][0][0]
        with patch.dict(sys.modules, {"monai": monai, "monai.transforms": transforms}):
            for probability in (0.0, 1.0):
                with self.subTest(noise_probability=probability):
                    training = SyntheticMRIDataset(
                        **common, augment=True, gaussian_noise_probability=probability,
                    )[0][0][0]
                    expected = evaluation if probability == 0 else evaluation + 0.25
                    self.assertTrue(torch.equal(training, expected))
                    self.assertGreater(float(training.max()), 0.5)
                    self.assertLess(float(training.flatten()[-2]), float(training.flatten()[-1]))

    def test_real_monai_transform_contains_noise_only(self):
        try:
            from monai.transforms import Compose, RandGaussianNoise
        except ImportError:
            self.skipTest("MONAI unavailable; run this check in the training environment")
        image = torch.linspace(-0.7, 0.7, 128).reshape(1, 4, 4, 8)
        actual = _build_training_intensity_transform(1.0, 0.015, None)
        expected = Compose([RandGaussianNoise(prob=1.0, std=0.015, mean=0)])
        actual.set_random_state(seed=42)
        expected.set_random_state(seed=42)
        result = actual(image.clone())
        self.assertTrue(torch.equal(result, expected(image.clone())))
        self.assertGreater(float(result.max()), 0.5)
        self.assertLess(float(result.min()), -0.5)


if __name__ == "__main__":
    unittest.main()
