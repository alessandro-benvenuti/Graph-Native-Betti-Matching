#!/usr/bin/env python3
"""Validate configured datasets on a login node before allocating GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Executing this file by path sets sys.path[0] to cluster/jean_zay rather than
# the repository root.  Resolve imports independently of the caller's cwd so
# both manual preflight and submit_train.sh behave identically.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from configs import load_config, validate_config
from data.loaders.common import image_graph_collate
from data.loaders.mixed import build_datasets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def _check_item(dataset, index: int, label: str) -> None:
    item = dataset[index]
    images, segmentations, nodes, edges, _, _ = image_graph_collate([item])
    if type(images) is not torch.Tensor or type(segmentations) is not torch.Tensor:
        raise TypeError(f"{label}: collation retained a tensor subclass")
    if images.ndim != 5 or segmentations.shape != images.shape:
        raise ValueError(
            f"{label}: invalid dense shapes image={tuple(images.shape)} "
            f"segmentation={tuple(segmentations.shape)}"
        )
    if not torch.isfinite(images).all() or not torch.isfinite(segmentations).all():
        raise ValueError(f"{label}: image or segmentation contains non-finite values")
    graph_nodes = nodes[0]
    graph_edges = edges[0]
    if graph_nodes.ndim != 2 or graph_nodes.shape[1] < 3:
        raise ValueError(f"{label}: invalid node shape {tuple(graph_nodes.shape)}")
    if not torch.isfinite(graph_nodes).all():
        raise ValueError(f"{label}: nodes contain non-finite values")
    if graph_edges.ndim != 2 or graph_edges.shape[1] != 2:
        raise ValueError(f"{label}: invalid edge shape {tuple(graph_edges.shape)}")
    if graph_edges.numel() and (
        int(graph_edges.min()) < 0 or int(graph_edges.max()) >= len(graph_nodes)
    ):
        raise ValueError(f"{label}: edge endpoint is outside the node array")
    print(
        f"preflight {label}: image={tuple(images.shape)} "
        f"nodes={len(graph_nodes)} edges={len(graph_edges)}"
    )


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    validate_config(config)
    train, validation, sampler = build_datasets(config)
    print(
        f"preflight datasets: train={len(train)} validation={len(validation)} "
        f"sampling={'weighted' if sampler is not None else 'complete'}"
    )
    for name, dataset in (("train", train), ("val", validation)):
        if len(dataset) == 0:
            raise ValueError(f"{name}: dataset is empty")
        indices = sorted({0, len(dataset) - 1})
        for index in indices:
            _check_item(dataset, index, f"{name}[{index}]")


if __name__ == "__main__":
    main()
