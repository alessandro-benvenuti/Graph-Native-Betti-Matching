"""Configuration-driven dataset composition and DataLoader construction."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data.loaders.common import image_graph_collate, seed_data_worker
from data.loaders.plants import build_plants_dataset
from data.loaders.synthetic_mri import build_synthetic_mri_dataset


def _supports_keyword(callable_object, keyword: str) -> bool:
    """Return whether a callable explicitly accepts a compatibility keyword."""

    try:
        return keyword in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False


def compose_source_target(
    source: Dataset,
    target: Dataset,
    *,
    balance_source_target: bool,
) -> Tuple[ConcatDataset, Optional[WeightedRandomSampler]]:
    """Concatenate source and target data and optionally equalize expected draws."""

    if len(source) == 0 or len(target) == 0:
        raise ValueError("Mixed training requires non-empty source and target datasets")
    dataset = ConcatDataset((source, target))
    if not balance_source_target:
        return dataset, None
    target_weight = float(len(source)) / float(len(target))
    weights = torch.cat(
        (
            torch.ones(len(source), dtype=torch.double),
            torch.full((len(target),), target_weight, dtype=torch.double),
        )
    )
    return dataset, WeightedRandomSampler(
        weights,
        num_samples=2 * len(source),
        replacement=True,
    )


def _dataset_for_split(
    name: str,
    settings: Mapping,
    config: Mapping,
    split: str,
):
    data = config["data"]
    augmentation = config["augmentation"]
    role = settings["role"]
    domain_label = 0 if role == "source" else 1
    max_samples = (
        settings["train_samples"]
        if split == "train"
        else settings.get("validation_samples")
    )
    common = dict(
        root=Path(settings["root"]),
        split=split,
        max_samples=max_samples,
        domain_label=domain_label,
        # Config-driven builds require actual root/{train,val} splits. This
        # prevents a direct leaf from being reused as both train and validation.
        allow_direct_root=False,
        augment=(split == "train" and bool(data["train_augmentation"])),
    )
    if name == "plants":
        image_size = data["image_size"]
        if len(set(image_size)) != 1:
            raise ValueError("Plants projection currently requires a cubic image_size")
        padding = int(augmentation["plants"].get("padding", 5))
        plants_augmentation = augmentation["plants"]
        flip = plants_augmentation["flip"]
        probability = float(flip["probability_per_axis"])
        return build_plants_dataset(
            size=int(image_size[0]),
            padding=padding,
            projection_depth=int(plants_augmentation["projection_depth"]),
            rotate_90=bool(plants_augmentation["rotate_90"]["enabled"]),
            flip_probability=(probability, probability, probability)
            if bool(flip["enabled"])
            else (0.0, 0.0, 0.0),
            **common,
        )
    if name == "synthetic_mri":
        mri_augmentation = augmentation["synthetic_mri"]
        zoom = mri_augmentation["zoom"]
        noise = mri_augmentation["gaussian_noise"]
        return build_synthetic_mri_dataset(
            image_size=data["image_size"],
            foreground_mean=float(settings["foreground_mean"]),
            coordinate_space=str(settings["coordinate_space_on_disk"]),
            rotate_90=bool(mri_augmentation["rotate_90"]["enabled"]),
            zoom_range=tuple(zoom["range"]) if bool(zoom["enabled"]) else None,
            gaussian_noise_probability=(
                float(noise["probability"]) if bool(noise["enabled"]) else 0.0
            ),
            gaussian_noise_max_std=float(noise["std_range"][1]),
            clamp_range=tuple(mri_augmentation["intensity_clamp"]),
            **common,
        )
    raise ValueError(f"Unsupported dataset: {name}")


def build_datasets(config: Mapping):
    """Build train/validation datasets and the optional domain sampler."""

    source_train = []
    source_validation = []
    target_train = []
    target_validation = []
    for name, settings in config["data"]["datasets"].items():
        train = _dataset_for_split(name, settings, config, "train")
        validation = _dataset_for_split(name, settings, config, "val")
        if settings["role"] == "source":
            source_train.append(train)
            source_validation.append(validation)
        else:
            target_train.append(train)
            target_validation.append(validation)

    if not target_train:
        raise ValueError("At least one target dataset is required")
    target_train_dataset = ConcatDataset(target_train)
    target_validation_dataset = ConcatDataset(target_validation)
    if not source_train:
        return target_train_dataset, target_validation_dataset, None

    source_train_dataset = ConcatDataset(source_train)
    source_validation_dataset = ConcatDataset(source_validation)
    train_dataset, sampler = compose_source_target(
        source_train_dataset,
        target_train_dataset,
        balance_source_target=bool(
            config["data"]["mixed_sampling"]["balance_source_target"]
        ),
    )
    validation_dataset = ConcatDataset(
        (source_validation_dataset, target_validation_dataset)
    )
    return train_dataset, validation_dataset, sampler


def build_data_loaders(config: Mapping):
    """Construct reproducibly seeded PyTorch train and validation loaders."""

    train_dataset, validation_dataset, sampler = build_datasets(config)
    data = config["data"]
    runtime = config["runtime"]
    seed = int(config["experiment"]["seed"])
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    common = dict(
        batch_size=int(data["batch_size"]),
        num_workers=int(runtime["workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        collate_fn=image_graph_collate,
        worker_init_fn=seed_data_worker,
    )
    train_options = dict(shuffle=sampler is None, sampler=sampler)
    validation_options = dict(shuffle=False)
    if _supports_keyword(DataLoader.__init__, "generator"):
        train_options["generator"] = train_generator
        validation_options["generator"] = validation_generator
    train_loader = DataLoader(train_dataset, **train_options, **common)
    validation_loader = DataLoader(
        validation_dataset, **validation_options, **common
    )
    return train_loader, validation_loader


__all__ = ["build_data_loaders", "build_datasets", "compose_source_target"]
