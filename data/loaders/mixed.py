"""Configuration-driven dataset composition and DataLoader construction."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    WeightedRandomSampler,
)

from data.loaders.common import image_graph_collate, seed_data_worker
from data.loaders.plants import build_plants_dataset
from data.loaders.synthetic_mri import (
    build_synthetic_mri_dataset,
    select_capped_records,
)


def _supports_keyword(callable_object, keyword: str) -> bool:
    """Return whether a callable explicitly accepts a compatibility keyword."""

    try:
        return keyword in inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False


class DistributedWeightedSampler(Sampler[int]):
    """Deterministically shard a global weighted sample across DDP ranks."""

    def __init__(
        self,
        weights,
        num_samples: int,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        replacement: bool = True,
    ):
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.global_num_samples = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0
        self.num_samples = (
            self.global_num_samples + self.num_replicas - 1
        ) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.global_num_samples,
            self.replacement,
            generator=generator,
        ).tolist()
        if len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


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
            clamp_range=(
                tuple(mri_augmentation["intensity_clamp"])
                if mri_augmentation["intensity_clamp"] is not None
                else None
            ),
            sample_cap_selection=str(
                settings.get("sample_cap_selection", "first")
            ),
            sample_cap_seed=int(settings.get("sample_cap_seed", 0)),
            **common,
        )
    raise ValueError(f"Unsupported dataset: {name}")


def dataset_sample_manifest(dataset: Dataset):
    """Return sample IDs for every concrete dataset below a concatenation."""
    leaves = []

    def visit(current):
        if isinstance(current, ConcatDataset):
            for child in current.datasets:
                visit(child)
            return
        records = getattr(current, "records", None)
        if records is None:
            return
        sample_ids = [str(record.sample_id) for record in records]
        leaves.append(
            {
                "dataset_type": type(current).__name__,
                "domain_label": int(getattr(current, "domain_label", -1)),
                "count": len(sample_ids),
                "sample_ids": sample_ids,
            }
        )

    visit(dataset)
    return leaves


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


def build_data_loaders(config: Mapping, *, rank: int = 0, world_size: int = 1):
    """Construct reproducibly seeded PyTorch train and validation loaders."""

    train_dataset, validation_dataset, sampler = build_datasets(config)
    data = config["data"]
    runtime = config["runtime"]
    seed = int(config["experiment"]["seed"])
    train_generator = torch.Generator().manual_seed(seed + rank)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    common = dict(
        num_workers=int(runtime["workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        collate_fn=image_graph_collate,
        worker_init_fn=seed_data_worker,
    )
    if world_size > 1:
        if sampler is None:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
        else:
            sampler = DistributedWeightedSampler(
                sampler.weights,
                sampler.num_samples,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                replacement=sampler.replacement,
            )
    train_options = dict(shuffle=sampler is None, sampler=sampler)
    validation_options = dict(shuffle=False)
    if _supports_keyword(DataLoader.__init__, "generator"):
        train_options["generator"] = train_generator
        validation_options["generator"] = validation_generator
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data["batch_size"]),
        **train_options,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(data.get("validation_batch_size", data["batch_size"])),
        **validation_options,
        **common,
    )
    return train_loader, validation_loader


def build_evaluation_loader(
    config: Mapping,
    *,
    dataset_name: str,
    split: str = "val",
    max_samples: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
):
    """Build an unshuffled, augmentation-free loader for one configured dataset."""

    split = split.strip().lower()
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'")
    datasets = config["data"]["datasets"]
    if dataset_name not in datasets:
        raise ValueError(
            "Unknown evaluation dataset {!r}; configured datasets are {}".format(
                dataset_name, sorted(datasets)
            )
        )
    settings = dict(datasets[dataset_name])
    # Discover the complete split before applying an evaluation-only cap. In
    # particular, Plants intentionally preserves legacy filesystem order for
    # training, which must not make a metric smoke-test subset nondeterministic.
    settings["validation_samples"] = None
    dataset = _dataset_for_split(dataset_name, settings, config, split)
    if hasattr(dataset, "records"):
        dataset.records.sort(key=lambda record: record.sample_id)
    if max_samples is not None and sample_ids is not None:
        raise ValueError("max_samples and sample_ids are mutually exclusive")
    if sample_ids is not None:
        requested = [str(sample_id).strip() for sample_id in sample_ids]
        if not requested or any(not sample_id for sample_id in requested):
            raise ValueError("sample_ids must contain at least one non-empty ID")
        if len(set(requested)) != len(requested):
            raise ValueError("sample_ids must not contain duplicates")
        records_by_id = {record.sample_id: record for record in dataset.records}
        missing = [sample_id for sample_id in requested if sample_id not in records_by_id]
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"Requested sample IDs are absent from the {split} split: "
                f"{preview}{suffix}"
            )
        # Preserve the caller's order so every model export follows the same
        # explicitly recorded patch manifest.
        dataset.records = [records_by_id[sample_id] for sample_id in requested]
    elif max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive or None")
        dataset.records = select_capped_records(
            dataset.records,
            int(max_samples),
            mode=str(settings.get("sample_cap_selection", "first")),
            seed=int(settings.get("sample_cap_seed", 0)),
            split=split,
        )
    data = config["data"]
    runtime = config["runtime"]
    options = dict(
        batch_size=int(data.get("validation_batch_size", data["batch_size"])),
        shuffle=False,
        num_workers=int(runtime["workers"]),
        pin_memory=bool(runtime["pin_memory"]),
        collate_fn=image_graph_collate,
        worker_init_fn=seed_data_worker,
    )
    if _supports_keyword(DataLoader.__init__, "generator"):
        options["generator"] = torch.Generator().manual_seed(
            int(config["experiment"]["seed"]) + 1
        )
    return DataLoader(dataset, **options)


__all__ = [
    "DistributedWeightedSampler",
    "build_data_loaders",
    "build_datasets",
    "build_evaluation_loader",
    "compose_source_target",
    "dataset_sample_manifest",
]
