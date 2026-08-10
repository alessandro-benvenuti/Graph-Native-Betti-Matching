"""Optional on-disk readers used by dataset loaders.

Heavy readers are imported only when an actual dataset item is read, keeping
discovery and unit tests usable in lightweight environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def read_medpy_image(path: Path) -> torch.Tensor:
    try:
        from medpy.io import load
    except ImportError as error:
        raise ImportError(
            "Reading dataset images requires medpy. Install the dataset extras."
        ) from error
    array, _ = load(str(path))
    array = np.asarray(array)
    return torch.from_numpy(array.copy())


def read_nifti(path: Path) -> torch.Tensor:
    """Read a NIfTI volume with MedPy."""

    return read_medpy_image(path)


def read_png_grayscale(path: Path) -> torch.Tensor:
    """Read Plants PNGs through MedPy and normalize the channel layout."""

    tensor = read_medpy_image(path)
    array = tensor.numpy()
    if array.ndim == 2:
        return tensor
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D/3D Plants image, got {array.shape}")
    if array.shape[0] <= 4 and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        return torch.from_numpy(array[..., 0].copy())
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError("RGB Plants images require Pillow for grayscale conversion") from error
    grayscale = np.asarray(Image.fromarray(array).convert("L"))
    return torch.from_numpy(grayscale.copy())


def _decode_vtk_lines(flat_lines: np.ndarray) -> torch.Tensor:
    values = np.asarray(flat_lines, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return torch.empty((0, 2), dtype=torch.long)
    if values.size % 3 != 0:
        raise ValueError("Expected VTK line connectivity triples")
    lines = values.reshape(-1, 3)
    if not np.all(lines[:, 0] == 2):
        raise ValueError("Expected every VTK line cell to contain two node indices")
    return torch.as_tensor(lines[:, 1:].copy(), dtype=torch.long)


def read_vtp_graph(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        import pyvista
    except ImportError as error:
        raise ImportError(
            "Reading VTP graphs requires pyvista and vtk. Install the dataset extras."
        ) from error
    graph = pyvista.read(path)
    points = torch.as_tensor(
        np.asarray(graph.points, dtype=np.float32).copy(), dtype=torch.float32
    )
    edges = _decode_vtk_lines(np.asarray(graph.lines))
    return points, edges
