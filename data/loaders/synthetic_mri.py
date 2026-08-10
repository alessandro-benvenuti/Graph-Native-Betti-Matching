"""SyntheticMRI NIfTI/VTP dataset."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.augmentations import (
    normalize_voxel_coordinates,
    rotate_coordinates,
    rotate_volume,
    zoom_coordinates,
    zoom_volume,
)
from data.loaders.common import DatasetSample, SamplePaths
from data.loaders.discovery import discover_synthetic_mri
from data.loaders.io import read_nifti, read_vtp_graph


def _build_training_intensity_transform(
    noise_probability: float,
    noise_max_std: float,
    clamp_range,
):
    try:
        from monai.transforms import Compose, Lambda, RandGaussianNoise
    except ImportError as error:
        raise ImportError(
            "SyntheticMRI training requires MONAI Compose, RandGaussianNoise and Lambda"
        ) from error
    transforms = []
    if noise_probability > 0:
        transforms.append(
            RandGaussianNoise(prob=noise_probability, std=noise_max_std, mean=0)
        )
    if clamp_range is not None:
        low, high = (float(value) for value in clamp_range)
        transforms.append(Lambda(lambda image: image.clamp(low, high)))
    return Compose(transforms)


class SyntheticMRIDataset(Dataset):
    """Load native 3D image, segmentation, and graph triplets."""

    def __init__(
        self,
        records: Sequence[SamplePaths],
        *,
        image_size=(64, 64, 64),
        foreground_mean: float = 0.33335259556770325,
        coordinate_space: str = "normalized",
        augment: bool = False,
        rotate_90: bool = True,
        zoom_range=(0.6, 1.0),
        gaussian_noise_probability: float = 0.35,
        gaussian_noise_max_std: float = 0.015,
        clamp_range=(-0.5, 0.5),
        domain_label: int = 1,
        volume_reader=read_nifti,
        graph_reader=read_vtp_graph,
        intensity_transform=None,
        rng=None,
    ):
        if coordinate_space not in {"normalized", "voxel"}:
            raise ValueError("coordinate_space must be 'normalized' or 'voxel'")
        if len(image_size) != 3 or any(int(value) <= 0 for value in image_size):
            raise ValueError(f"Invalid image_size: {image_size}")
        self.records = list(records)
        self.image_size = tuple(int(value) for value in image_size)
        self.foreground_mean = float(foreground_mean)
        self.coordinate_space = coordinate_space
        self.augment = bool(augment)
        self.rotate_90 = bool(rotate_90)
        self.zoom_range = (
            None
            if zoom_range is None
            else (float(zoom_range[0]), float(zoom_range[1]))
        )
        if self.zoom_range is not None and (
            self.zoom_range[0] <= 0 or self.zoom_range[1] < self.zoom_range[0]
        ):
            raise ValueError(f"Invalid zoom_range: {self.zoom_range}")
        if not 0.0 <= float(gaussian_noise_probability) <= 1.0:
            raise ValueError("gaussian_noise_probability must lie in [0,1]")
        if float(gaussian_noise_max_std) < 0:
            raise ValueError("gaussian_noise_max_std must be non-negative")
        self.domain_label = int(domain_label)
        self.volume_reader = volume_reader
        self.graph_reader = graph_reader
        self.rng = rng if rng is not None else random
        self.intensity_transform = (
            intensity_transform
            if intensity_transform is not None or not self.augment
            else _build_training_intensity_transform(
                float(gaussian_noise_probability),
                float(gaussian_noise_max_std),
                clamp_range,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DatasetSample:
        record = self.records[index]
        raw_image = self.volume_reader(record.image).float()
        source_shape = tuple(int(value) for value in raw_image.shape[-3:])
        image = raw_image.unsqueeze(0) - self.foreground_mean
        segmentation = self.volume_reader(record.segmentation).float().unsqueeze(0) - 0.5
        nodes, edges = self.graph_reader(record.graph)

        if tuple(image.shape[-3:]) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=self.image_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        if tuple(segmentation.shape[-3:]) != self.image_size:
            segmentation = F.interpolate(
                segmentation.unsqueeze(0), size=self.image_size, mode="nearest"
            ).squeeze(0)

        nodes = nodes.float()
        if self.coordinate_space == "voxel":
            nodes = normalize_voxel_coordinates(nodes, source_shape)
        edges = edges.long()

        if self.augment:
            turns = (
                tuple(self.rng.randint(0, 3) for _ in range(3))
                if self.rotate_90
                else (0, 0, 0)
            )
            shape_before_rotation = tuple(int(value) for value in image.shape[-3:])
            image = rotate_volume(image, turns)
            segmentation = rotate_volume(segmentation, turns)
            nodes = rotate_coordinates(nodes, turns, shape_before_rotation)

            if self.zoom_range is not None:
                zoom_factor = self.rng.uniform(*self.zoom_range)
                image = zoom_volume(image, zoom_factor, mode="bilinear")
                segmentation = zoom_volume(segmentation, zoom_factor, mode="nearest")
                nodes = zoom_coordinates(nodes, zoom_factor, segmentation.shape[-3:])
            image = self.intensity_transform(image)

        return (
            [image],
            [segmentation],
            [nodes],
            [edges],
            [None],
            [self.domain_label],
        )


def build_synthetic_mri_dataset(
    root: Path | str,
    *,
    split: str,
    max_samples: Optional[int] = None,
    image_size=(64, 64, 64),
    foreground_mean: float = 0.33335259556770325,
    coordinate_space: str = "normalized",
    domain_label: int = 1,
    allow_direct_root: bool = True,
    augment: Optional[bool] = None,
    **kwargs,
) -> SyntheticMRIDataset:
    records = discover_synthetic_mri(
        Path(root), split, allow_direct=allow_direct_root
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        records = records[:max_samples]
    return SyntheticMRIDataset(
        records,
        image_size=image_size,
        foreground_mean=foreground_mean,
        coordinate_space=coordinate_space,
        augment=(split.strip().lower() == "train" if augment is None else augment),
        domain_label=domain_label,
        **kwargs,
    )


__all__ = ["SyntheticMRIDataset", "build_synthetic_mri_dataset"]
