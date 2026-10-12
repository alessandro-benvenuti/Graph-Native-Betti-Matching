#!/usr/bin/env python3
"""Export validation predictions and diagnostics for corruption design.

The export deliberately retains all 120 object queries and all unordered query
pairs. Thresholded classifications are diagnostics only; raw tensor shards make
it possible to revise thresholds without another GPU inference pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

import numpy as np
import torch
import yaml

from configs import load_config
from data.loaders import build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from models.matcher import build_matcher, score_candidate_structures
from training.evaluation.evaluator import _evaluation_volumes
from training.evaluation.visualization import save_graph_comparison


def segment_distances(points, nodes, edges):
    """Return nearest GT-segment distance, segment ID, and projection fraction."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if not len(points):
        return np.empty(0), np.empty(0, dtype=int), np.empty(0)
    if not len(edges):
        return (
            np.full(len(points), np.nan),
            np.full(len(points), -1, dtype=int),
            np.full(len(points), np.nan),
        )
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    vector = b - a
    denominator = np.maximum((vector * vector).sum(-1), 1e-12)
    projection = np.clip(
        ((points[:, None] - a) * vector).sum(-1) / denominator, 0, 1
    )
    distance = np.linalg.norm(
        points[:, None] - (a + projection[..., None] * vector), axis=-1
    )
    nearest = distance.argmin(1)
    rows = np.arange(len(points))
    return distance[rows, nearest], nearest, projection[rows, nearest]


def _edge_matrix(node_count, edges):
    matrix = np.zeros((int(node_count), int(node_count)), dtype=bool)
    for left, right in np.asarray(edges, dtype=int).reshape(-1, 2):
        matrix[int(left), int(right)] = matrix[int(right), int(left)] = True
    return matrix


def topology(active, adjacency):
    """Compute elementary topology of an active undirected simple graph."""
    active = np.asarray(active, dtype=bool)
    adjacency = np.asarray(adjacency, dtype=bool)
    ids = np.flatnonzero(active)
    graph = adjacency[np.ix_(ids, ids)]
    seen: set[int] = set()
    components = 0
    for node in range(len(ids)):
        if node in seen:
            continue
        components += 1
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            for neighbour in np.flatnonzero(graph[current]):
                neighbour = int(neighbour)
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    degrees = graph.sum(1)
    edge_count = int(graph.sum() // 2)
    return {
        "nodes": len(ids),
        "edges": edge_count,
        "components": components,
        "cycle_rank": edge_count - len(ids) + components,
        "endpoints": int((degrees == 1).sum()),
        "bifurcations": int((degrees >= 3).sum()),
    }


def bridge_edges(active, adjacency):
    """Return undirected bridges as canonical ``(min, max)`` query pairs."""
    active = np.asarray(active, dtype=bool)
    adjacency = np.asarray(adjacency, dtype=bool)
    discovery = np.full(len(active), -1, dtype=int)
    low = np.full(len(active), -1, dtype=int)
    parent = np.full(len(active), -1, dtype=int)
    bridges: set[tuple[int, int]] = set()
    clock = 0

    def visit(node):
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        for raw_neighbour in np.flatnonzero(adjacency[node] & active):
            neighbour = int(raw_neighbour)
            if discovery[neighbour] < 0:
                parent[neighbour] = node
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbour))))
            elif neighbour != parent[node]:
                low[node] = min(low[node], discovery[neighbour])

    for node in np.flatnonzero(active):
        node = int(node)
        if discovery[node] < 0:
            visit(node)
    return bridges


def connected_components(node_count, edges):
    """Return the component ID of every node in an undirected graph."""
    adjacency = [[] for _ in range(int(node_count))]
    for left, right in np.asarray(edges, dtype=int).reshape(-1, 2):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    labels = np.full(int(node_count), -1, dtype=int)
    component = 0
    for start in range(int(node_count)):
        if labels[start] >= 0:
            continue
        labels[start] = component
        stack = [start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if labels[neighbour] < 0:
                    labels[neighbour] = component
                    stack.append(neighbour)
        component += 1
    return labels


def _quantiles(values, prefix):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {
            prefix + "_count": 0,
            prefix + "_mean": None,
            prefix + "_q25": None,
            prefix + "_median": None,
            prefix + "_q75": None,
            prefix + "_max": None,
        }
    return {
        prefix + "_count": int(len(values)),
        prefix + "_mean": float(values.mean()),
        prefix + "_q25": float(np.quantile(values, 0.25)),
        prefix + "_median": float(np.quantile(values, 0.50)),
        prefix + "_q75": float(np.quantile(values, 0.75)),
        prefix + "_max": float(values.max()),
    }


def diagnostics(
    x, p, pairs, q, gt, edges, source, target, thresholds, tolerance
):
    """Compute node records, soft costs, and thresholded diagnostics."""
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    p = np.asarray(p, dtype=float).reshape(-1)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    q = np.asarray(q, dtype=float).reshape(-1)
    gt = np.asarray(gt, dtype=float).reshape(-1, 3)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    source = np.asarray(source, dtype=int).reshape(-1)
    target = np.asarray(target, dtype=int).reshape(-1)
    n = len(x)

    correspondence = np.full(n, -1, dtype=int)
    correspondence[source] = target
    reverse = np.full(len(gt), -1, dtype=int)
    reverse[target] = source
    if len(gt) and np.any(reverse < 0):
        raise ValueError("Every GT node must receive one RelationFormer query")

    target_adjacency = np.zeros((n, n), dtype=float)
    for gt_left, gt_right in edges:
        left, right = reverse[int(gt_left)], reverse[int(gt_right)]
        target_adjacency[left, right] = target_adjacency[right, left] = 1.0

    adjacency = np.zeros((n, n), dtype=float)
    adjacency[pairs[:, 0], pairs[:, 1]] = q
    adjacency += adjacency.T
    effective = adjacency * p[:, None] * p[None, :]

    if len(gt):
        point_distance = np.linalg.norm(x[:, None] - gt[None], axis=-1)
        nearest = point_distance.argmin(1)
        nearest_distance = point_distance.min(1)
    else:
        nearest = np.full(n, -1, dtype=int)
        nearest_distance = np.full(n, np.nan)
    segment_distance, segment_id, segment_projection = segment_distances(x, gt, edges)
    gt_degree = np.bincount(edges.ravel(), minlength=len(gt))
    displacement = np.zeros_like(x)
    displacement[source] = x[source] - gt[target]
    matched_lookup = {int(gt_id): int(query) for query, gt_id in zip(source, target)}

    node_rows = []
    for query in range(n):
        matched_gt = int(correspondence[query])
        nearest_gt = int(nearest[query])
        nearest_matched_query = matched_lookup.get(nearest_gt, -1)
        node_rows.append(
            {
                "query": query,
                "gt_node": matched_gt,
                "matched": matched_gt >= 0,
                "confidence": float(p[query]),
                "d": float(x[query, 0]),
                "h": float(x[query, 1]),
                "w": float(x[query, 2]),
                "coordinate_error_d": float(displacement[query, 0]) if matched_gt >= 0 else None,
                "coordinate_error_h": float(displacement[query, 1]) if matched_gt >= 0 else None,
                "coordinate_error_w": float(displacement[query, 2]) if matched_gt >= 0 else None,
                "coordinate_error": float(np.linalg.norm(displacement[query])) if matched_gt >= 0 else None,
                "nearest_gt_node": nearest_gt,
                "nearest_node_distance": float(nearest_distance[query]),
                "nearest_matched_query": nearest_matched_query,
                "edge_to_nearest_matched_probability": (
                    float(adjacency[query, nearest_matched_query])
                    if nearest_matched_query >= 0 and nearest_matched_query != query
                    else None
                ),
                "nearest_segment": int(segment_id[query]),
                "segment_distance": float(segment_distance[query]),
                "segment_projection": float(segment_projection[query]),
                "boundary_distance": float(np.minimum(x[query], 1 - x[query]).min()),
                "gt_degree": int(gt_degree[matched_gt]) if matched_gt >= 0 else None,
                "raw_soft_degree": float(adjacency[query].sum()),
                "effective_soft_degree": float(effective[query].sum()),
                "duplicate_candidate": bool(matched_gt < 0 and nearest_distance[query] < tolerance),
            }
        )

    matched_mask = correspondence >= 0
    upper = np.triu_indices(n, 1)
    pair_target = target_adjacency[upper]
    pair_raw = adjacency[upper]
    pair_effective = effective[upper]
    matched_pair = matched_mask[upper[0]] & matched_mask[upper[1]]
    gt_pair = pair_target > 0
    unmatched_pair = ~matched_pair

    summary = {
        "gt_nodes": len(gt),
        "gt_edges": len(edges),
        "node_cost": float(np.abs(p - matched_mask).sum()),
        "coordinate_cost": float((p[source] * np.linalg.norm(displacement[source], axis=1)).sum()),
        "coordinate_cost_unweighted": float(np.linalg.norm(displacement[source], axis=1).sum()),
        "edge_raw_cost": float(np.abs(pair_raw - pair_target).sum()),
        "edge_effective_cost": float(np.abs(pair_effective - pair_target).sum()),
        "expected_nodes": float(p.sum()),
        "expected_edges_raw": float(pair_raw.sum()),
        "expected_edges_effective": float(pair_effective.sum()),
    }
    for key in ("node_cost", "coordinate_cost", "coordinate_cost_unweighted"):
        summary[key + "_per_gt_node"] = summary[key] / max(1, len(gt))
    for key in ("edge_raw_cost", "edge_effective_cost"):
        summary[key + "_per_gt_edge"] = summary[key] / max(1, len(edges))
        summary[key + "_per_candidate"] = summary[key] / max(1, len(pairs))
    summary.update(_quantiles(pair_raw[gt_pair], "gt_edge_probability"))
    summary.update(_quantiles(pair_raw[matched_pair & ~gt_pair], "matched_nonedge_probability"))
    summary.update(_quantiles(pair_raw[unmatched_pair], "unmatched_endpoint_edge_probability"))
    summary.update(_quantiles(p[source], "matched_node_probability"))
    summary.update(_quantiles(p[~matched_mask], "unmatched_node_probability"))
    gt_stats = topology(np.ones(len(gt), dtype=bool), _edge_matrix(len(gt), edges))
    summary.update({"gt_" + key: value for key, value in gt_stats.items()})

    nearest_pair_probability = np.zeros(n, dtype=float)
    for query in range(n):
        nearest_query = matched_lookup.get(int(nearest[query]), query)
        if nearest_query != query:
            nearest_pair_probability[query] = adjacency[query, nearest_query]

    threshold_details = {}
    for threshold in thresholds:
        active = p >= threshold
        hard = adjacency >= threshold
        live = hard & active[:, None] & active[None, :]
        stats = topology(active, hard)
        bridges = bridge_edges(active, live)
        false_live = np.argwhere(np.triu(live & (target_adjacency == 0), 1))
        false_loop_edges = sum(
            tuple(sorted((int(left), int(right)))) not in bridges
            for left, right in false_live
        )
        prefix = f"t{threshold:g}_"
        summary.update({prefix + key: value for key, value in stats.items()})
        summary[prefix + "false_edges"] = int(len(false_live))
        summary[prefix + "false_loop_edges"] = int(false_loop_edges)
        summary[prefix + "missing_edges"] = int(np.triu((target_adjacency == 1) & ~live, 1).sum())
        summary[prefix + "extra_nodes"] = int((active & ~matched_mask).sum())
        summary[prefix + "missing_nodes"] = int((~active & matched_mask).sum())
        summary[prefix + "duplicate_candidates"] = int(
            (active & ~matched_mask & (nearest_distance < tolerance)).sum()
        )
        live_degree = live.sum(1)
        summary[prefix + "subdivision_node_candidates"] = int(
            (active & ~matched_mask & (segment_distance < tolerance) & (live_degree == 2)).sum()
        )
        summary[prefix + "split_node_candidates"] = int(
            (
                active
                & ~matched_mask
                & (nearest_distance < tolerance)
                & (nearest_pair_probability < threshold)
            ).sum()
        )
        threshold_details[float(threshold)] = {
            "active": active,
            "live": live,
            "bridges": bridges,
        }
    return node_rows, summary, target_adjacency, displacement, threshold_details


def edge_event_rows(
    x, p, pairs, q, correspondence, gt_edges, target_adjacency,
    threshold_details, event_threshold
):
    """Create a compact table of GT edges and plausible predicted error edges."""
    gt_edges = np.asarray(gt_edges, dtype=int).reshape(-1, 2)
    gt_count = int(max(-1, np.asarray(correspondence).max()) + 1)
    components = connected_components(gt_count, gt_edges)
    rows = []
    for index, (left, right) in enumerate(np.asarray(pairs, dtype=int)):
        probability = float(q[index])
        is_gt = bool(target_adjacency[left, right])
        if not is_gt and probability < float(event_threshold):
            continue
        gt_left, gt_right = int(correspondence[left]), int(correspondence[right])
        both_matched = gt_left >= 0 and gt_right >= 0
        gt_path_exists = bool(
            both_matched and len(components) and components[gt_left] == components[gt_right]
        )
        row = {
            "left": int(left),
            "right": int(right),
            "left_gt_node": gt_left,
            "right_gt_node": gt_right,
            "probability": probability,
            "effective_probability": float(p[left] * p[right] * probability),
            "left_confidence": float(p[left]),
            "right_confidence": float(p[right]),
            "length": float(np.linalg.norm(x[left] - x[right])),
            "gt_edge": is_gt,
            "unmatched_endpoint": not both_matched,
            "gt_path_exists": gt_path_exists,
            "gt_loop_closing_candidate": bool(not is_gt and gt_path_exists),
        }
        for threshold, detail in threshold_details.items():
            live = bool(detail["live"][left, right])
            bridge = tuple(sorted((int(left), int(right)))) in detail["bridges"]
            row[f"t{threshold:g}_live"] = live
            row[f"t{threshold:g}_in_cycle"] = bool(live and not bridge)
        rows.append(row)
    return rows


class CsvSink:
    """Stream dictionaries to a CSV or gzip-compressed CSV."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None
        self.writer = None

    def write(self, row):
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            opener = gzip.open if self.path.suffix == ".gz" else open
            self.handle = opener(self.path, "wt", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        if self.handle is not None:
            self.handle.close()


class RawShardWriter:
    """Write bounded collections of raw tensor records using ``torch.save``."""
    def __init__(self, directory, shard_size):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.records = []
        self.shard_index = 0

    def add(self, record):
        self.records.append(record)
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.records:
            return
        torch.save(self.records, self.directory / f"shard_{self.shard_index:05d}.pt")
        self.records = []
        self.shard_index += 1

    def close(self):
        self.flush()


class ReliabilityAccumulator:
    """Accumulate confidence calibration without retaining individual pairs."""
    def __init__(self, bins=20):
        self.boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
        self.values = {}

    def add(self, name, probabilities, labels):
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float).reshape(-1)
        bucket = np.minimum(
            np.searchsorted(self.boundaries, probabilities, side="right") - 1,
            len(self.boundaries) - 2,
        )
        bucket = np.maximum(bucket, 0)
        state = self.values.setdefault(
            name,
            {
                "count": np.zeros(len(self.boundaries) - 1, dtype=np.int64),
                "probability_sum": np.zeros(len(self.boundaries) - 1),
                "label_sum": np.zeros(len(self.boundaries) - 1),
            },
        )
        np.add.at(state["count"], bucket, 1)
        np.add.at(state["probability_sum"], bucket, probabilities)
        np.add.at(state["label_sum"], bucket, labels)

    def rows(self):
        for name, state in self.values.items():
            for index, count in enumerate(state["count"]):
                yield {
                    "prediction_type": name,
                    "bin": index,
                    "lower": float(self.boundaries[index]),
                    "upper": float(self.boundaries[index + 1]),
                    "count": int(count),
                    "mean_probability": float(state["probability_sum"][index] / count) if count else None,
                    "positive_fraction": float(state["label_sum"][index] / count) if count else None,
                }


def _write_rows(path, rows: Iterable[Mapping]):
    sink = CsvSink(path)
    for row in rows:
        sink.write(row)
    sink.close()


def _load_patch_index(path, expected_ids):
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Patch index does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError(f"Missing sample_id in {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        result[sample_id] = row
    missing = sorted(set(expected_ids) - set(result))
    if missing:
        raise KeyError(
            f"{len(missing)} validation samples are absent from {path}; first: {missing[:5]}"
        )
    return result


def _checkpoint_epoch(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Checkpoint name must end in an integer epoch: {path}") from error


def _relation_logits(model, tokens, pairs, object_queries, relation_tokens, chunk_size):
    """Evaluate both edge directions for every graph in a mini-batch."""
    batch_size = tokens.shape[0]
    object_features = tokens[:, :object_queries]
    relation_features = tokens[:, object_queries : object_queries + relation_tokens]
    forward_chunks, reverse_chunks = [], []
    for chunk in pairs.split(int(chunk_size)):
        left = object_features[:, chunk[:, 0]]
        right = object_features[:, chunk[:, 1]]
        extras = []
        if relation_tokens:
            extras.append(
                relation_features.reshape(batch_size, 1, -1).expand(batch_size, len(chunk), -1)
            )
        forward_input = torch.cat((left, right, *extras), dim=-1)
        reverse_input = torch.cat((right, left, *extras), dim=-1)
        forward_chunks.append(
            model.relation_embed(forward_input.reshape(-1, forward_input.shape[-1]))
            .reshape(batch_size, len(chunk), -1).cpu()
        )
        reverse_chunks.append(
            model.relation_embed(reverse_input.reshape(-1, reverse_input.shape[-1]))
            .reshape(batch_size, len(chunk), -1).cpu()
        )
    return torch.cat(forward_chunks, 1), torch.cat(reverse_chunks, 1)


def _plot_costs(folder, patch_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key in ("node_cost", "coordinate_cost", "edge_raw_cost", "edge_effective_cost"):
        figure, axis = plt.subplots()
        axis.hist([float(row[key]) for row in patch_rows], bins=40)
        axis.set_xlabel(key)
        axis.set_ylabel("patches")
        figure.tight_layout()
        figure.savefig(folder / f"{key}.png", dpi=160)
        plt.close(figure)


def _ranked_cases(rows, count=100):
    ranked = []
    for metric in ("node_cost", "coordinate_cost", "edge_raw_cost", "edge_effective_cost"):
        ordered = sorted(rows, key=lambda item: float(item[metric]), reverse=True)[:count]
        for rank, row in enumerate(ordered, start=1):
            ranked.append({"metric": metric, "rank": rank, "sample_id": row["sample_id"], "value": row[metric]})
    return ranked


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="synthetic_mri")
    parser.add_argument("--patch-index", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-chunk-size", type=int, default=1024)
    parser.add_argument("--raw-shard-size", type=int, default=128)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--edge-event-threshold", type=float, default=0.3)
    parser.add_argument("--spatial-tolerance", type=float, default=0.03)
    parser.add_argument("--visualizations", type=int, default=20)
    parser.add_argument(
        "--replace-incomplete", action="store_true",
        help="Replace a checkpoint directory lacking COMPLETED.json",
    )
    return parser


@torch.no_grad()
def main():
    args = _parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.batch_size <= 0 or args.pair_chunk_size <= 0 or args.raw_shard_size <= 0:
        raise ValueError("batch, pair-chunk, and raw-shard sizes must be positive")
    if any(value < 0 or value > 1 for value in args.thresholds):
        raise ValueError("thresholds must lie in [0,1]")
    if not 0 <= args.edge_event_threshold <= 1:
        raise ValueError("--edge-event-threshold must lie in [0,1]")
    if args.spatial_tolerance < 0 or args.visualizations < 0:
        raise ValueError("tolerance and visualization count must be non-negative")

    config = load_config(args.config)
    config["runtime"]["workers"] = args.workers
    config["runtime"]["device"] = args.device
    config["data"]["validation_batch_size"] = args.batch_size
    loader = build_evaluation_loader(
        config, dataset_name=args.dataset, split="val", max_samples=args.max_samples
    )
    records = list(loader.dataset.records)
    sample_ids = [str(record.sample_id) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Validation sample IDs are not unique")

    dataset_root = Path(config["data"]["datasets"][args.dataset]["root"])
    patch_index_path = args.patch_index
    if patch_index_path is None and (dataset_root / "patch_index.csv").is_file():
        patch_index_path = dataset_root / "patch_index.csv"
    provenance = _load_patch_index(patch_index_path, sample_ids)
    manifest = [
        {
            "sample_id": record.sample_id,
            "graph": str(record.graph),
            "image": str(record.image),
            "segmentation": str(record.segmentation),
            "provenance": provenance.get(record.sample_id),
        }
        for record in records
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    settings = vars(args).copy()
    settings["output_dir"] = str(args.output_dir)
    settings["patch_index"] = str(patch_index_path) if patch_index_path else None
    settings["dataset_root"] = str(dataset_root)
    settings["checkpoints"] = [str(Path(item).resolve()) for item in args.checkpoint]
    settings.pop("checkpoint", None)
    (args.output_dir / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(config).to(device)
    matcher = build_matcher(config)
    decoder = config["model"]["decoder"]
    object_queries = int(decoder["object_queries"])
    relation_tokens = int(decoder["relation_tokens"])
    pairs_device = torch.combinations(torch.arange(object_queries, device=device), r=2)
    pairs = pairs_device.cpu().numpy()
    torch.save(torch.from_numpy(pairs.copy()), args.output_dir / "edge_pairs.pt")

    comparison_rows = []
    for checkpoint_value in args.checkpoint:
        checkpoint = Path(checkpoint_value)
        label = checkpoint.stem
        folder = args.output_dir / label
        completion = folder / "COMPLETED.json"
        if completion.is_file():
            with (folder / "patches.csv").open(encoding="utf-8", newline="") as handle:
                comparison_rows.extend(csv.DictReader(handle))
            print(f"{label}: already complete; skipping", flush=True)
            continue
        if folder.exists():
            if not args.replace_incomplete:
                raise FileExistsError(
                    f"Incomplete output exists at {folder}; inspect it and rerun with "
                    "--replace-incomplete to replace only this checkpoint export"
                )
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / "plots").mkdir()
        checkpoint_report = load_legacy_model_checkpoint(model, checkpoint, map_location="cpu")
        model.eval()
        epoch = _checkpoint_epoch(checkpoint)
        if hasattr(matcher, "set_training_progress"):
            matcher.set_training_progress(epoch)

        node_sink = CsvSink(folder / "nodes.csv.gz")
        edge_sink = CsvSink(folder / "edge_events.csv.gz")
        raw_writer = RawShardWriter(folder / "raw", args.raw_shard_size)
        reliability = ReliabilityAccumulator()
        patch_rows = []
        sample_index = 0

        for batch_index, batch in enumerate(loader):
            volumes = _evaluation_volumes(batch, config["training"]["input"], device)
            tokens, prediction, _ = model(volumes)
            node_logits = prediction["pred_logits"].detach().cpu()
            node_probabilities = node_logits.softmax(-1)[..., 1].numpy()
            coordinates = prediction["pred_nodes"][..., :3].detach().cpu().numpy()
            forward_logits, reverse_logits = _relation_logits(
                model, tokens, pairs_device, object_queries, relation_tokens, args.pair_chunk_size
            )
            edge_logits = (forward_logits + reverse_logits) * 0.5
            edge_probabilities = edge_logits.softmax(-1)[..., 1].numpy()

            targets = {"nodes": batch[2], "edges": batch[3]}
            if matcher.requires_structure:
                candidates = matcher.matching_candidates(prediction, targets)
                structures = score_candidate_structures(
                    tokens, model.relation_embed, candidates,
                    object_queries=object_queries,
                    relation_tokens=relation_tokens,
                    pair_chunk_size=args.pair_chunk_size,
                )
                matches = matcher(
                    prediction, targets,
                    predicted_structure=structures,
                    candidate_indices=candidates,
                )
            else:
                matches = matcher(prediction, targets)

            for local_index in range(len(batch[2])):
                record = records[sample_index]
                sample_id = str(record.sample_id)
                x = coordinates[local_index]
                p = node_probabilities[local_index]
                q = edge_probabilities[local_index]
                gt = batch[2][local_index].cpu().numpy()[:, :3]
                gt_edges = batch[3][local_index].cpu().numpy().astype(int).reshape(-1, 2)
                if len(gt) > object_queries:
                    raise ValueError(
                        f"{sample_id}: {len(gt)} GT nodes exceed {object_queries} query slots"
                    )
                source, target = [value.cpu().numpy() for value in matches[local_index]]
                node_rows, summary, target_adjacency, _, threshold_details = diagnostics(
                    x, p, pairs, q, gt, gt_edges, source, target,
                    args.thresholds, args.spatial_tolerance,
                )
                identity = {"checkpoint": label, "epoch": epoch, "sample_id": sample_id}
                provenance_row = provenance.get(sample_id, {})
                patch_row = {
                    **identity,
                    **{"provenance_" + key: value for key, value in provenance_row.items() if key != "sample_id"},
                    **summary,
                }
                patch_rows.append(patch_row)
                for row in node_rows:
                    node_sink.write({**identity, **row})

                correspondence = np.full(object_queries, -1, dtype=int)
                correspondence[source] = target
                for row in edge_event_rows(
                    x, p, pairs, q, correspondence, gt_edges, target_adjacency,
                    threshold_details, args.edge_event_threshold,
                ):
                    edge_sink.write({**identity, **row})

                upper = np.triu_indices(object_queries, 1)
                reliability.add("node", p, correspondence >= 0)
                reliability.add("edge_raw", q, target_adjacency[upper])
                reliability.add(
                    "edge_effective", p[pairs[:, 0]] * p[pairs[:, 1]] * q,
                    target_adjacency[upper],
                )
                raw_writer.add(
                    {
                        "sample_id": sample_id,
                        "checkpoint": label,
                        "epoch": epoch,
                        "nodes": torch.from_numpy(x.copy()),
                        "node_logits": node_logits[local_index].clone(),
                        "gt_nodes": torch.from_numpy(gt.copy()),
                        "gt_edges": torch.from_numpy(gt_edges.copy()),
                        "edge_logits_forward": forward_logits[local_index].clone(),
                        "edge_logits_reverse": reverse_logits[local_index].clone(),
                        "matched_queries": torch.from_numpy(source.copy()),
                        "matched_gt": torch.from_numpy(target.copy()),
                        "model_input_shape": tuple(int(value) for value in volumes.shape[-3:]),
                    }
                )

                if sample_index < args.visualizations:
                    active = np.flatnonzero(p >= 0.5)
                    remap = {int(query): index for index, query in enumerate(active)}
                    visible_edges = [
                        (remap[int(left)], remap[int(right)])
                        for (left, right), probability in zip(pairs, q)
                        if left in remap and right in remap and probability >= 0.5
                    ]
                    save_graph_comparison(
                        batch[1][local_index], batch[2][local_index], batch[3][local_index],
                        {"nodes": torch.as_tensor(x[active]), "edges": torch.as_tensor(visible_edges).reshape(-1, 2)},
                        folder / "plots" / f"{sample_index:06d}_{sample_id}.png",
                        title=f"{label}: {sample_id}",
                    )
                sample_index += 1

            print(
                f"{label}: {sample_index}/{len(records)} patches "
                f"({batch_index + 1}/{len(loader)} batches)", flush=True,
            )

        node_sink.close()
        edge_sink.close()
        raw_writer.close()
        _write_rows(folder / "patches.csv", patch_rows)
        _write_rows(folder / "calibration.csv", reliability.rows())
        _write_rows(folder / "ranked_cases.csv", _ranked_cases(patch_rows))
        _plot_costs(folder, patch_rows)
        completion.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "epoch": epoch,
                    "samples": sample_index,
                    "ignored_removed_parameters": list(checkpoint_report.ignored_removed),
                }, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        comparison_rows.extend(patch_rows)

    _write_rows(args.output_dir / "checkpoint_comparison.csv", comparison_rows)
    print(f"Analysis saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
