"""Deterministic discovery of graph-labelled dataset triplets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from data.loaders.common import SamplePaths


REQUIRED_FOLDERS = ("raw", "seg", "vtp")


def resolve_split_root(root: Path, split: str, *, allow_direct: bool = True) -> Path:
    """Resolve ``root/split/{raw,seg,vtp}``, optionally accepting a leaf root."""

    split_name = split.strip().lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    candidates = (root / split_name, root) if allow_direct else (root / split_name,)
    for candidate in candidates:
        if all((candidate / folder).is_dir() for folder in REQUIRED_FOLDERS):
            return candidate
    raise FileNotFoundError(
        f"Could not find {REQUIRED_FOLDERS} for split '{split_name}' below {root}"
    )


def discover_synthetic_mri(
    root: Path, split: str, *, allow_direct: bool = True
) -> List[SamplePaths]:
    leaf = resolve_split_root(root, split, allow_direct=allow_direct)
    images = sorted((leaf / "raw").glob("*_data.nii*"))
    records = []
    for image in images:
        suffix = "_data.nii.gz" if image.name.endswith("_data.nii.gz") else "_data.nii"
        sample_id = image.name[: -len(suffix)]
        extension = ".nii.gz" if suffix.endswith(".nii.gz") else ".nii"
        segmentation = leaf / "seg" / f"{sample_id}_seg{extension}"
        graph = leaf / "vtp" / f"{sample_id}_graph.vtp"
        _require_pair(image, segmentation, graph)
        records.append(SamplePaths(image, segmentation, graph, sample_id))
    if not records:
        raise FileNotFoundError(f"No *_data.nii or *_data.nii.gz files in {leaf / 'raw'}")
    return records


def discover_plants(
    root: Path, split: str, *, allow_direct: bool = True
) -> List[SamplePaths]:
    leaf = resolve_split_root(root, split, allow_direct=allow_direct)
    raw_directory = leaf / "raw"
    # Preserve os.listdir order because the sample cap is applied before
    # applying its sample cap, so sorting could select a different subset.
    images = [
        raw_directory / name
        for name in os.listdir(raw_directory)
        if name.endswith("_data.png")
    ]
    records = []
    for image in images:
        sample_id = image.name[: -len("_data.png")]
        segmentation = leaf / "seg" / f"{sample_id}_seg.png"
        graph = leaf / "vtp" / f"{sample_id}_graph.vtp"
        _require_pair(image, segmentation, graph)
        records.append(SamplePaths(image, segmentation, graph, sample_id))
    if not records:
        raise FileNotFoundError(f"No *_data.png files in {leaf / 'raw'}")
    return records


def _require_pair(image: Path, segmentation: Path, graph: Path) -> None:
    missing = [path for path in (segmentation, graph) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete sample for {image.name}; missing: "
            + ", ".join(str(path) for path in missing)
        )
