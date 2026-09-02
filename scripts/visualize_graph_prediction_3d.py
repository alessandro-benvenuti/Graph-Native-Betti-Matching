#!/usr/bin/env python3
"""Create a portable interactive 3D SyntheticMRI prediction report."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.loaders.io import read_nifti, read_vtp_graph  # noqa: E402
from training.evaluation.interactive_visualization import (  # noqa: E402
    classify_visualization_errors,
    filter_prediction,
    find_patch_triplet,
    load_patch_provenance,
    load_prediction_records,
    normalized_dhw_to_plot_xyz,
    resolve_prediction_record,
    validate_graph_endpoints,
)


def _visibility_option(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--show-{name}", dest=f"show_{name.replace('-', '_')}", action="store_true", help=help_text)
    group.add_argument(f"--hide-{name}", dest=f"show_{name.replace('-', '_')}", action="store_false")
    parser.set_defaults(**{f"show_{name.replace('-', '_')}": default})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Root containing split/raw, split/seg and split/vtp")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--predictions", type=Path, required=True, help="Existing evaluation predictions.json")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--sample-id", help="Dataset source_sample_id (for example sample_000109_0021)")
    identity.add_argument("--evaluation-id", help="Sequential evaluation sample_id (for example sample_000123)")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--node-threshold", type=float, default=0.0)
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    _visibility_option(parser, "mri", True, "Show three orthogonal MRI slices")
    _visibility_option(parser, "segmentation", True, "Show the segmentation surface")
    _visibility_option(parser, "ground-truth", True, "Show the ground-truth graph")
    _visibility_option(parser, "prediction", True, "Show the thresholded predicted graph")
    parser.add_argument("--html", action=argparse.BooleanOptionalAction, default=True, help="Write a self-contained HTML report (default: enabled)")
    parser.add_argument("--png", action="store_true", help="Also write a PNG via Plotly/Kaleido when available")
    parser.add_argument("--error-analysis", action="store_true", help="Overlay likely node/edge errors using visualization-time matching")
    parser.add_argument("--match-distance", type=float, default=0.1, help="Maximum normalized D/H/W distance for visualization-time node matching")
    return parser


def _source_default():
    scratch = os.environ.get("SCRATCH")
    return Path(scratch) / "datasets/syntheticMRI/new_patches" if scratch else None


def _list_records(records):
    print("No sample identifier supplied. Available predictions:")
    print(f"{'evaluation_id':<18} source_sample_id")
    for record in records:
        print(f"{str(record.get('sample_id', '')):<18} {record.get('source_sample_id', '')}")


def _add_mri_slices(figure, volume, *, visible):
    import plotly.graph_objects as go

    depth, height, width = volume.shape
    low, high = np.percentile(volume[np.isfinite(volume)], (2, 98))
    if low == high:
        high = low + 1.0
    d, h, w = depth // 2, height // 2, width // 2
    x_xy, y_xy = np.meshgrid(np.arange(width), np.arange(height))
    x_xz, z_xz = np.meshgrid(np.arange(width), np.arange(depth))
    y_yz, z_yz = np.meshgrid(np.arange(height), np.arange(depth))
    surfaces = (
        (x_xy, y_xy, np.full_like(x_xy, d), volume[d, :, :], "axial"),
        (x_xz, np.full_like(x_xz, h), z_xz, volume[:, h, :], "coronal"),
        (np.full_like(y_yz, w), y_yz, z_yz, volume[:, :, w], "sagittal"),
    )
    for index, (x, y, z, values, plane) in enumerate(surfaces):
        figure.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=values, colorscale="Gray",
            cmin=float(low), cmax=float(high), opacity=0.38, showscale=False,
            name="MRI slices", legendgroup="mri", showlegend=index == 0,
            visible=visible, hovertemplate=f"MRI {plane}<extra></extra>",
        ))


def _add_segmentation(figure, segmentation, *, visible):
    import plotly.graph_objects as go

    mask = np.asarray(segmentation) > 0.5
    if not mask.any() or mask.all():
        return
    from skimage.measure import marching_cubes

    vertices_dhw, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    vertices = vertices_dhw[:, [2, 1, 0]]
    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#77aa88", opacity=0.20, flatshading=True,
        name="Segmentation", legendgroup="segmentation", visible=visible,
        hoverinfo="skip",
    ))


def _edge_coordinates(nodes_xyz, edges):
    x, y, z = [], [], []
    for left, right in edges:
        for values, axis in ((x, 0), (y, 1), (z, 2)):
            values.extend((nodes_xyz[left, axis], nodes_xyz[right, axis], None))
    return x, y, z


def _add_graph(figure, nodes_xyz, edges, *, scores=None, edge_scores=None, color, name, visible):
    import plotly.graph_objects as go

    if len(edges):
        if edge_scores is None:
            edge_groups = [(np.arange(len(edges)), color, 3.0)]
        else:
            # A few confidence bins keep reports compact while making edge
            # color/width genuinely confidence-dependent.
            edge_scores = np.asarray(edge_scores, dtype=np.float32)
            bins = np.minimum((edge_scores * 5).astype(np.int64), 4)
            palette = ("#bdd7ee", "#8bbde1", "#589fd2", "#2d7fba", "#08519c")
            edge_groups = [
                (np.flatnonzero(bins == index), palette[index], 1.5 + index)
                for index in range(5)
                if np.any(bins == index)
            ]
        for group_index, (indices, edge_color, width) in enumerate(edge_groups):
            grouped_edges = np.asarray(edges)[indices]
            x, y, z = _edge_coordinates(nodes_xyz, grouped_edges)
            hover = None
            if edge_scores is not None:
                hover = []
                for edge_index in indices:
                    label = f"edge {int(edge_index)}<br>confidence={edge_scores[edge_index]:.3f}"
                    hover.extend((label, label, None))
            figure.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", text=hover,
                line={"color": edge_color, "width": width},
                name=f"{name} edges", legendgroup=name, visible=visible,
                showlegend=group_index == 0,
                hovertemplate="%{text}<extra></extra>" if hover else None,
                hoverinfo=None if hover else "skip",
            ))
    if len(nodes_xyz):
        marker = {"color": color, "size": 5 if scores is None else 4 + 5 * np.asarray(scores), "opacity": 0.95}
        hover = None
        if scores is not None:
            marker.update({"color": scores, "colorscale": "Blues", "cmin": 0, "cmax": 1, "colorbar": {"title": "Node confidence", "len": 0.45}})
            hover = [f"node {index}<br>confidence={score:.3f}" for index, score in enumerate(scores)]
        figure.add_trace(go.Scatter3d(
            x=nodes_xyz[:, 0], y=nodes_xyz[:, 1], z=nodes_xyz[:, 2], mode="markers",
            marker=marker, text=hover, hovertemplate="%{text}<extra></extra>" if hover else None,
            name=f"{name} nodes", legendgroup=name, visible=visible,
        ))


def _add_error_overlays(figure, classification, predicted_xyz, gt_xyz):
    import plotly.graph_objects as go

    def nodes(indices, coordinates, color, name, symbol):
        if not indices:
            return
        points = coordinates[np.asarray(indices, dtype=np.int64)]
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker={"size": 9, "color": color, "symbol": symbol, "line": {"color": "black", "width": 1}},
            name=name, legendgroup="likely-errors",
        ))

    def edges(edge_values, coordinates, color, name, dash="solid"):
        if not edge_values:
            return
        x, y, z = _edge_coordinates(coordinates, edge_values)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line={"color": color, "width": 7, "dash": dash},
            name=name, legendgroup="likely-errors", hoverinfo="skip",
        ))

    nodes(classification.unmatched_predicted_nodes, predicted_xyz, "#ff8c00", "Likely node FP", "x")
    nodes(classification.unmatched_gt_nodes, gt_xyz, "#ff00aa", "Likely node FN", "diamond")
    edges(classification.incident_to_unmatched_edges, predicted_xyz, "#ff8c00", "Pred edge incident to unmatched node", "dot")
    edges(classification.false_positive_edges, predicted_xyz, "#ffe600", "Likely edge FP")
    edges(classification.missing_gt_edges, gt_xyz, "#ff00aa", "Likely missing GT edge", "dash")


def _metadata_html(record, provenance, gt_nodes, gt_edges, prediction, args):
    metrics = record.get("metrics") or {}
    fields = [
        ("Source sample ID", record.get("source_sample_id")),
        ("Evaluation sample ID", record.get("sample_id")),
        ("Split", args.split),
        ("Patient ID", provenance.get("patient_id", "unavailable")),
        ("Patch start D/H/W", "/".join(provenance.get(key, "?") for key in ("start_d", "start_h", "start_w"))),
        ("GT nodes / edges", f"{len(gt_nodes)} / {len(gt_edges)}"),
        ("Predicted nodes / edges (after thresholds)", f"{len(prediction.nodes_dhw)} / {len(prediction.edges)}"),
        ("Node / edge thresholds", f"{args.node_threshold:g} / {args.edge_threshold:g}"),
    ]
    rows = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in fields)
    metric_rows = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value))}</td></tr>" for key, value in sorted(metrics.items()))
    note = ""
    if args.error_analysis:
        note = f"<p><strong>Error overlay:</strong> visualization-time one-to-one Euclidean matching in normalized D/H/W, maximum distance {args.match_distance:g}. This is not the official evaluation or training Hungarian assignment.</p>"
    return f"<h1>SyntheticMRI graph prediction</h1><table>{rows}</table>{note}<h2>Exported per-sample metrics</h2><table>{metric_rows or '<tr><td>None</td></tr>'}</table>"


def _write_html(path, figure, metadata):
    plot = figure.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>SyntheticMRI graph prediction</title><style>body{{font-family:system-ui,sans-serif;margin:1.5rem;color:#222}}table{{border-collapse:collapse;margin-bottom:1rem}}th,td{{text-align:left;border:1px solid #ccc;padding:.35rem .55rem}}th{{background:#f3f3f3}}p{{max-width:80rem}}</style></head><body>{metadata}{plot}</body></html>"""
    path.write_text(document, encoding="utf-8")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    records = load_prediction_records(args.predictions)
    if not args.sample_id and not args.evaluation_id:
        _list_records(records)
        return 0
    if args.dataset_root is None:
        args.dataset_root = _source_default()
    if args.dataset_root is None:
        parser.error("--dataset-root is required when $SCRATCH is not set")
    if not args.html and not args.png:
        parser.error("At least one of --html or --png must be enabled")

    record = resolve_prediction_record(records, sample_id=args.sample_id, evaluation_id=args.evaluation_id)
    source_id = str(record["source_sample_id"])
    paths = find_patch_triplet(args.dataset_root, args.split, source_id)
    raw = np.asarray(read_nifti(paths.image).squeeze().numpy())
    segmentation = np.asarray(read_nifti(paths.segmentation).squeeze().numpy())
    if raw.ndim != 3 or segmentation.shape != raw.shape:
        raise ValueError(f"Expected matching 3D raw/seg volumes, got {raw.shape} and {segmentation.shape}")
    gt_nodes_tensor, gt_edges_tensor = read_vtp_graph(paths.graph)
    gt_nodes = np.asarray(gt_nodes_tensor, dtype=np.float32).reshape(-1, 3)
    gt_edges = validate_graph_endpoints(gt_edges_tensor, len(gt_nodes), label="ground truth")
    prediction = filter_prediction(record, node_threshold=args.node_threshold, edge_threshold=args.edge_threshold)
    gt_xyz = normalized_dhw_to_plot_xyz(gt_nodes, raw.shape)
    predicted_xyz = normalized_dhw_to_plot_xyz(prediction.nodes_dhw, raw.shape)

    import plotly.graph_objects as go

    figure = go.Figure()
    _add_mri_slices(figure, raw, visible=args.show_mri)
    _add_segmentation(figure, segmentation, visible=args.show_segmentation)
    _add_graph(figure, gt_xyz, gt_edges, color="#d62728", name="Ground truth", visible=args.show_ground_truth)
    _add_graph(figure, predicted_xyz, prediction.edges, scores=prediction.node_scores, edge_scores=prediction.edge_scores, color="#1f77b4", name="Prediction", visible=args.show_prediction)
    if args.error_analysis:
        classification = classify_visualization_errors(prediction.nodes_dhw, prediction.edges, gt_nodes, gt_edges, max_distance=args.match_distance)
        _add_error_overlays(figure, classification, predicted_xyz, gt_xyz)
    depth, height, width = raw.shape
    figure.update_layout(
        scene={
            "xaxis": {"title": "X / W (voxels)", "range": [0, width]},
            "yaxis": {"title": "Y / H (voxels)", "range": [0, height]},
            "zaxis": {"title": "Z / D (voxels)", "range": [0, depth]},
            "aspectmode": "data",
        },
        legend={"groupclick": "togglegroup"}, margin={"l": 0, "r": 0, "b": 0, "t": 25}, height=850,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = load_patch_provenance(args.dataset_root, source_id)
    metadata = _metadata_html(record, provenance, gt_nodes, gt_edges, prediction, args)
    stem = source_id + "__" + str(record.get("sample_id", "evaluation"))
    if args.html:
        html_path = args.output_dir / f"{stem}.html"
        _write_html(html_path, figure, metadata)
        print(f"Wrote interactive HTML: {html_path}")
    if args.png:
        png_path = args.output_dir / f"{stem}.png"
        try:
            figure.write_image(png_path, width=1600, height=1000, scale=1.5)
        except Exception as error:
            raise RuntimeError("PNG export requires a working headless Plotly image backend (install a Plotly-compatible Kaleido version). The HTML output is unaffected.") from error
        print(f"Wrote static PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
