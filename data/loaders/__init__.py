"""Supported graph-labelled datasets and their canonical builders."""

from data.loaders.common import DatasetSample, SamplePaths, image_graph_collate
from data.loaders.discovery import discover_plants, discover_synthetic_mri
from data.loaders.mixed import (
    DistributedWeightedSampler,
    build_data_loaders,
    build_datasets,
    build_evaluation_loader,
    compose_source_target,
    dataset_sample_manifest,
)
from data.loaders.plants import PlantsDataset, build_plants_dataset
from data.loaders.synthetic_mri import (
    SyntheticMRIDataset,
    build_synthetic_mri_dataset,
    select_capped_records,
)

__all__ = [
    "DatasetSample",
    "DistributedWeightedSampler",
    "PlantsDataset",
    "SamplePaths",
    "SyntheticMRIDataset",
    "build_data_loaders",
    "build_datasets",
    "build_evaluation_loader",
    "build_plants_dataset",
    "build_synthetic_mri_dataset",
    "compose_source_target",
    "dataset_sample_manifest",
    "discover_plants",
    "discover_synthetic_mri",
    "image_graph_collate",
    "select_capped_records",
]
