"""Headless 3D plots for graph-extraction evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import canonical_edges


def normalized_dhw_to_plot_xyz(nodes, spatial_shape):
    """Map canonical normalized D/H/W coordinates to plot X/Y/Z voxels."""

    nodes = torch.as_tensor(nodes, dtype=torch.float32).reshape(-1, 3).cpu()
    shape = torch.as_tensor(spatial_shape, dtype=nodes.dtype)
    voxels = nodes * shape
    return torch.stack((voxels[:, 2], voxels[:, 1], voxels[:, 0]), dim=1).numpy()


def _draw_graph(axis, nodes_xyz, edges, color, label):
    if len(nodes_xyz):
        axis.scatter(
            nodes_xyz[:, 0], nodes_xyz[:, 1], nodes_xyz[:, 2],
            s=14, c=color, label=label + " nodes",
        )
    for left, right in canonical_edges(edges, len(nodes_xyz)):
        segment = nodes_xyz[[left, right]]
        axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=1.2)


def save_graph_comparison(
    segmentation,
    target_nodes,
    target_edges,
    prediction,
    path,
    *,
    title="",
    max_segmentation_points=5000,
):
    """Save a deterministic PNG with segmentation, target, and prediction."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Matplotlib versions installed on Magnolia do not register the ``3d``
    # projection until mplot3d is imported explicitly. Newer releases perform
    # this import implicitly, so keep the explicit import for both runtimes.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    volume = torch.as_tensor(segmentation).detach().cpu().squeeze().numpy()
    if volume.ndim != 3:
        raise ValueError("segmentation must reduce to a D/H/W volume")
    shape = volume.shape
    foreground = np.argwhere(volume > 0)
    if len(foreground) > int(max_segmentation_points):
        indices = np.linspace(
            0, len(foreground) - 1, int(max_segmentation_points), dtype=np.int64
        )
        foreground = foreground[indices]
    foreground_xyz = foreground[:, [2, 1, 0]] if len(foreground) else foreground
    target_xyz = normalized_dhw_to_plot_xyz(target_nodes, shape)
    predicted_xyz = normalized_dhw_to_plot_xyz(prediction["nodes"], shape)

    figure = plt.figure(figsize=(11, 5))
    for panel, (nodes, edges, color, name) in enumerate(
        (
            (target_xyz, target_edges, "tab:red", "Target"),
            (predicted_xyz, prediction["edges"], "tab:blue", "Prediction"),
        ),
        start=1,
    ):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        if len(foreground_xyz):
            axis.scatter(
                foreground_xyz[:, 0], foreground_xyz[:, 1], foreground_xyz[:, 2],
                s=0.4, c="0.65", alpha=0.12,
            )
        _draw_graph(axis, nodes, edges, color, name)
        axis.set_title(name)
        axis.set_xlim(0, shape[2])
        axis.set_ylim(0, shape[1])
        axis.set_zlim(0, shape[0])
        axis.set_xlabel("W / X")
        axis.set_ylabel("H / Y")
        axis.set_zlabel("D / Z")
        if hasattr(axis, "set_box_aspect"):
            axis.set_box_aspect((shape[2], shape[1], shape[0]))
    figure.suptitle(title)
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output), dpi=160)
    plt.close(figure)
    return output


__all__ = ["normalized_dhw_to_plot_xyz", "save_graph_comparison"]
