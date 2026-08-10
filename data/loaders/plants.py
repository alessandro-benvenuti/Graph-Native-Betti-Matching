"""Plants PNG/VTP dataset with graph-safe 2D-to-3D augmentation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.augmentations import AugmentationParameters, apply_augmentation
from data.loaders.common import DatasetSample, SamplePaths
from data.loaders.discovery import discover_plants
from data.loaders.io import read_png_grayscale, read_vtp_graph


def _build_monai_transforms(size: int, padding: int):
    try:
        from monai.transforms import Resize, ScaleIntensity, SpatialPad
    except ImportError as error:
        raise ImportError(
            "The Plants dataset requires MONAI Resize, ScaleIntensity and SpatialPad"
        ) from error
    inner_size = size - 2 * padding
    return (
        Resize(spatial_size=(inner_size, inner_size)),
        ScaleIntensity(minv=-0.5, maxv=0.5),
        SpatialPad([size, size, size], value=-0.5),
    )


class PlantsDataset(Dataset):
    """Load 2D Plants patches and project them into five-slice 3D slabs."""

    def __init__(
        self,
        records: Sequence[SamplePaths],
        *,
        size: int = 64,
        padding: int = 5,
        projection_depth: int = 5,
        augment: bool = False,
        rotate_90: bool = True,
        flip_probability=(0.5, 0.5, 0.5),
        domain_label: int = 0,
        image_reader=read_png_grayscale,
        graph_reader=read_vtp_graph,
        resize_transform=None,
        scale_transform=None,
        pad_transform=None,
        rng=None,
    ):
        if size <= 0 or padding < 0 or 2 * padding >= size:
            raise ValueError(f"Invalid Plants size/padding: size={size}, padding={padding}")
        if projection_depth <= 0 or projection_depth % 2 == 0:
            raise ValueError("projection_depth must be a positive odd integer")
        if len(flip_probability) != 3 or any(
            not 0.0 <= float(value) <= 1.0 for value in flip_probability
        ):
            raise ValueError("flip_probability must contain three values in [0,1]")
        self.records = list(records)
        self.size = int(size)
        self.padding = int(padding)
        self.inner_size = self.size - 2 * self.padding
        self.projection_depth = int(projection_depth)
        self.augment = bool(augment)
        self.rotate_90 = bool(rotate_90)
        self.flip_probability = tuple(float(value) for value in flip_probability)
        self.domain_label = int(domain_label)
        self.image_reader = image_reader
        self.graph_reader = graph_reader
        self.rng = rng if rng is not None else random

        if resize_transform is None or scale_transform is None or pad_transform is None:
            resize, scale, pad = _build_monai_transforms(self.size, self.padding)
            resize_transform = resize_transform or resize
            scale_transform = scale_transform or scale
            pad_transform = pad_transform or pad
        self.resize = resize_transform
        self.scale = scale_transform
        self.pad = pad_transform

    def __len__(self) -> int:
        return len(self.records)

    def resize_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Resize a binary vessel mask without averaging thin branches away.

        Downsampling uses foreground occupancy: an output cell is foreground
        when any source pixel in its adaptive pooling region is foreground.
        This is the categorical-mask counterpart to the area interpolation used
        for images. Nearest interpolation is used only when an axis grows.
        """

        if segmentation.ndim != 3:
            raise ValueError(
                f"Expected a channel-first [C,H,W] mask, got {segmentation.shape}"
            )
        target = (self.inner_size, self.inner_size)
        source = tuple(int(value) for value in segmentation.shape[-2:])
        if source == target:
            return segmentation
        if all(target_size <= source_size for source_size, target_size in zip(source, target)):
            return F.adaptive_max_pool2d(segmentation.unsqueeze(0), target).squeeze(0)
        return F.interpolate(
            segmentation.unsqueeze(0), size=target, mode="nearest"
        ).squeeze(0)

    def _project_image_3d(self, image: torch.Tensor, z_position: float = 0.5) -> torch.Tensor:
        depth = int(image.shape[-1])
        centre = round(z_position * depth)
        radius = self.projection_depth // 2
        if centre - radius < 0 or centre + radius >= depth:
            raise ValueError(
                f"Cannot project {self.projection_depth} slices into depth {depth}"
            )
        projected = image.new_zeros((image.shape[0], image.shape[1], image.shape[2], depth))
        for offset in range(-radius, radius + 1):
            projected[..., centre + offset] = image
        return projected - 0.5

    def _augmentation_parameters(self) -> AugmentationParameters:
        if not self.augment:
            return AugmentationParameters()
        # Preserve the original RNG order: three rotations, then x/y/z flips.
        turns = (
            tuple(self.rng.randint(0, 3) for _ in range(3))
            if self.rotate_90
            else (0, 0, 0)
        )
        # Do not advance the random stream for disabled flips.
        sampled_flips = []
        for probability in self.flip_probability:
            if probability == 0.0:
                sampled_flips.append(False)
            elif probability == 1.0:
                sampled_flips.append(True)
            else:
                sampled_flips.append(self.rng.random() < probability)
        flip_x, flip_y, flip_z = sampled_flips
        return AugmentationParameters(
            quarter_turns=turns,
            flip_axes=(flip_y, flip_x, flip_z),
        )

    def __getitem__(self, index: int) -> DatasetSample:
        record = self.records[index]
        segmentation = self.image_reader(record.segmentation).float()
        image = self.image_reader(record.image).float()
        nodes, edges = self.graph_reader(record.graph)

        # Segmentation is categorical. Threshold it at native resolution and
        # use occupancy downsampling so one-pixel branches are not diluted by
        # image interpolation and then deleted by a second threshold.
        segmentation = ((segmentation / 255.0) >= 0.3).float().unsqueeze(0)
        segmentation = self.resize_segmentation(segmentation)
        image = self.resize((image / 255.0).unsqueeze(0))

        z_position = 0.5
        segmentation = self._project_image_3d(segmentation, z_position)
        image = self._project_image_3d(image, z_position)

        # The generator stores normalized (y,x); MedPy exposes the image axes
        # in (x,y), so swap once before appending the projected z coordinate.
        nodes = nodes.float()[:, :2]
        nodes = F.pad(nodes, (0, 1), "constant", z_position)[:, [1, 0, 2]]
        edges = edges.long()

        segmentation = torch.where(
            segmentation >= 0,
            segmentation.new_tensor(0.5),
            segmentation.new_tensor(-0.5),
        )
        transformed = apply_augmentation(
            image, segmentation, nodes, self._augmentation_parameters()
        )
        image = self.pad(self.scale(transformed.image))
        segmentation = self.pad(transformed.segmentation)
        nodes = (
            transformed.nodes * (self.size - 2 * self.padding) + self.padding
        ) / self.size

        return (
            [image],
            [segmentation],
            [nodes],
            [edges],
            [z_position],
            [self.domain_label],
        )


def build_plants_dataset(
    root: Path | str,
    *,
    split: str,
    max_samples: Optional[int] = None,
    size: int = 64,
    padding: int = 5,
    projection_depth: int = 5,
    domain_label: int = 0,
    allow_direct_root: bool = True,
    augment: Optional[bool] = None,
    **kwargs,
) -> PlantsDataset:
    records = discover_plants(
        Path(root), split, allow_direct=allow_direct_root
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        records = records[:max_samples]
    return PlantsDataset(
        records,
        size=size,
        padding=padding,
        projection_depth=projection_depth,
        augment=(split.strip().lower() == "train" if augment is None else augment),
        domain_label=domain_label,
        **kwargs,
    )


__all__ = ["PlantsDataset", "build_plants_dataset"]
