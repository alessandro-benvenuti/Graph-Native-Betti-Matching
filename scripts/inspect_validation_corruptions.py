#!/usr/bin/env python3
"""Inspect where validation graph errors occur using saved raw prediction shards.

This is a CPU-only post-processing pass.  It never reloads RelationFormer and
therefore does not repeat inference.  Large node/edge populations are described
with exact moments and fine fixed-bin histograms; reported quantiles for those
populations are histogram estimates.  The much smaller subdivision population
is retained exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


CORRUPTION_FIELDS = (
    "extra_nodes",
    "missing_nodes",
    "false_edges",
    "missing_edges",
    "false_loop_edges",
    "duplicate_candidates",
    "subdivision_node_candidates",
    "split_node_candidates",
)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class Distribution:
    """Streaming exact moments plus a fine, bounded histogram."""

    def __init__(self, low: float, high: float, bins: int):
        if not high > low or bins <= 0:
            raise ValueError("invalid distribution bounds")
        self.low = float(low)
        self.high = float(high)
        self.bins = int(bins)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.underflow = 0
        self.overflow = 0

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += int(len(values))
        self.total += float(values.sum())
        self.square_total += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.underflow += int((values < self.low).sum())
        self.overflow += int((values > self.high).sum())
        clipped = np.clip(values, self.low, self.high)
        indices = np.floor(
            (clipped - self.low) / (self.high - self.low) * self.bins
        ).astype(np.int64)
        indices = np.minimum(indices, self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins)

    def quantile(self, probability):
        if not self.count:
            return None
        position = float(probability) * (self.count - 1)
        lower_rank = int(math.floor(position))
        upper_rank = int(math.ceil(position))
        cumulative = np.cumsum(self.counts)
        lower_index = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
        upper_index = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
        width = (self.high - self.low) / self.bins
        lower = self.low + (lower_index + 0.5) * width
        upper = self.low + (upper_index + 0.5) * width
        return lower + (position - lower_rank) * (upper - lower)

    def summary(self):
        if not self.count:
            result = {
                "count": 0, "mean": None, "std": None, "min": None, "max": None,
                "underflow": 0, "overflow": 0,
            }
            result.update({f"q{int(q * 100):02d}": None for q in QUANTILES})
            return result
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean * mean)
        result = {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "underflow": self.underflow,
            "overflow": self.overflow,
        }
        result.update({f"q{int(q * 100):02d}": self.quantile(q) for q in QUANTILES})
        return result

    def histogram_rows(self, epoch, population, metric):
        width = (self.high - self.low) / self.bins
        for index, count in enumerate(self.counts):
            if count:
                yield {
                    "epoch": epoch,
                    "population": population,
                    "metric": metric,
                    "bin_left": self.low + index * width,
                    "bin_right": self.low + (index + 1) * width,
                    "count": int(count),
                }


def _probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values[..., 1] / values.sum(axis=-1)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _segment_distances(points, nodes, edges):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return np.full(len(points), np.nan), np.full(len(points), -1, dtype=np.int64)
    starts, ends = nodes[edges[:, 0]], nodes[edges[:, 1]]
    vectors = ends - starts
    denominator = np.maximum(np.square(vectors).sum(-1), 1e-12)
    projection = np.clip(
        ((points[:, None] - starts) * vectors).sum(-1) / denominator, 0, 1
    )
    distances = np.linalg.norm(
        points[:, None] - (starts + projection[..., None] * vectors), axis=-1
    )
    nearest = distances.argmin(1)
    return distances[np.arange(len(points)), nearest], nearest


def _point_distances(points, references):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    references = np.asarray(references, dtype=np.float64).reshape(-1, 3)
    if not len(references):
        return np.full(len(points), np.nan), np.full(len(points), -1, dtype=np.int64)
    values = np.linalg.norm(points[:, None] - references[None], axis=-1)
    nearest = values.argmin(1)
    return values[np.arange(len(points)), nearest], nearest


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _epoch(path):
    return int(path.name.rsplit("_", 1)[-1])


def _new_distributions():
    result = {}
    populations = ("unmatched_nodes", "extra_nodes", "duplicate_candidates")
    for population in populations:
        result[(population, "confidence")] = Distribution(0, 1, 10_000)
        result[(population, "nearest_gt_node_distance")] = Distribution(0, math.sqrt(3), 5_000)
        result[(population, "nearest_vessel_distance")] = Distribution(0, math.sqrt(3), 5_000)
        result[(population, "nearest_bifurcation_distance")] = Distribution(0, math.sqrt(3), 5_000)
        result[(population, "boundary_distance")] = Distribution(0, 0.5, 5_000)
    for population in ("gt_edges", "matched_nonedges", "unmatched_endpoint_edges"):
        result[(population, "raw_probability")] = Distribution(0, 1, 10_000)
        result[(population, "effective_probability")] = Distribution(0, 1, 10_000)
    return result


def _buffer_node_population(buffers, population, p, node_distance, vessel_distance,
                            bifurcation_distance, boundary_distance, mask):
    values = {
        "confidence": p,
        "nearest_gt_node_distance": node_distance,
        "nearest_vessel_distance": vessel_distance,
        "nearest_bifurcation_distance": bifurcation_distance,
        "boundary_distance": boundary_distance,
    }
    for metric, raw in values.items():
        buffers[(population, metric)].append(np.asarray(raw)[mask])


def _analyse_raw_epoch(folder, pairs, threshold, tolerance):
    epoch = _epoch(folder)
    distributions = _new_distributions()
    proximity = defaultdict(Counter)
    subdivision_rows = []
    records_seen = 0

    for shard in sorted((folder / "raw").glob("shard_*.pt")):
        buffers = defaultdict(list)
        for record in _torch_load(shard):
            records_seen += 1
            x = np.asarray(record["nodes"], dtype=np.float64)
            gt = np.asarray(record["gt_nodes"], dtype=np.float64)
            gt_edges = np.asarray(record["gt_edges"], dtype=np.int64).reshape(-1, 2)
            matched_queries = np.asarray(record["matched_queries"], dtype=np.int64)
            matched_gt = np.asarray(record["matched_gt"], dtype=np.int64)
            shape = np.asarray(record["model_input_shape"], dtype=np.float64)
            p = _probabilities(record["node_logits"])
            forward = np.asarray(record["edge_logits_forward"], dtype=np.float64)
            reverse = np.asarray(record["edge_logits_reverse"], dtype=np.float64)
            q = _probabilities((forward + reverse) * 0.5)

            correspondence = np.full(len(x), -1, dtype=np.int64)
            correspondence[matched_queries] = matched_gt
            unmatched = correspondence < 0
            active = p >= threshold
            extra = unmatched & active

            node_distance, nearest_gt = _point_distances(x, gt)
            vessel_distance, nearest_segment = _segment_distances(x, gt, gt_edges)
            degrees = np.bincount(gt_edges.ravel(), minlength=len(gt)) if len(gt) else np.empty(0)
            bifurcation_ids = np.flatnonzero(degrees >= 3)
            bifurcation_distance, _ = _point_distances(x, gt[bifurcation_ids])
            boundary_distance = np.minimum(x, 1 - x).min(axis=1)
            duplicate = extra & (node_distance < tolerance)

            for population, mask in (
                ("unmatched_nodes", unmatched),
                ("extra_nodes", extra),
                ("duplicate_candidates", duplicate),
            ):
                _buffer_node_population(
                    buffers, population, p, node_distance, vessel_distance,
                    bifurcation_distance, boundary_distance, mask,
                )
                total = int(mask.sum())
                proximity[population]["total"] += total
                proximity[population]["near_gt_node"] += int((mask & (node_distance < tolerance)).sum())
                proximity[population]["near_vessel"] += int((mask & (vessel_distance < tolerance)).sum())
                proximity[population]["near_bifurcation"] += int(
                    (mask & np.isfinite(bifurcation_distance) & (bifurcation_distance < tolerance)).sum()
                )
                proximity[population]["near_boundary"] += int((mask & (boundary_distance < tolerance)).sum())
                proximity[population]["patches_without_bifurcation"] += int(not len(bifurcation_ids))

            reverse_match = np.full(len(gt), -1, dtype=np.int64)
            reverse_match[matched_gt] = matched_queries
            target_adjacency = np.zeros((len(x), len(x)), dtype=bool)
            if len(gt_edges):
                left = reverse_match[gt_edges[:, 0]]
                right = reverse_match[gt_edges[:, 1]]
                target_adjacency[left, right] = True
                target_adjacency[right, left] = True
            target = target_adjacency[pairs[:, 0], pairs[:, 1]]
            both_matched = ~unmatched[pairs[:, 0]] & ~unmatched[pairs[:, 1]]
            endpoint_product = p[pairs[:, 0]] * p[pairs[:, 1]]
            effective = endpoint_product * q
            edge_groups = (
                ("gt_edges", target),
                ("matched_nonedges", both_matched & ~target),
                ("unmatched_endpoint_edges", ~both_matched),
            )
            for population, mask in edge_groups:
                buffers[(population, "raw_probability")].append(q[mask])
                buffers[(population, "effective_probability")].append(effective[mask])

            live_pair = active[pairs[:, 0]] & active[pairs[:, 1]] & (q >= threshold)
            live_degree = np.bincount(pairs[live_pair].ravel(), minlength=len(x))
            subdivision = extra & (vessel_distance < tolerance) & (live_degree == 2)
            for query in np.flatnonzero(subdivision):
                incident = live_pair & ((pairs[:, 0] == query) | (pairs[:, 1] == query))
                pair_ids = np.flatnonzero(incident)
                lengths_normalized = np.linalg.norm(
                    x[pairs[pair_ids, 0]] - x[pairs[pair_ids, 1]], axis=1
                )
                lengths_voxels = np.linalg.norm(
                    (x[pairs[pair_ids, 0]] - x[pairs[pair_ids, 1]]) * shape, axis=1
                )
                segment_id = int(nearest_segment[query])
                if segment_id >= 0:
                    gt_left, gt_right = gt_edges[segment_id]
                    gt_vector = gt[gt_left] - gt[gt_right]
                    gt_length_normalized = float(np.linalg.norm(gt_vector))
                    gt_length_voxels = float(np.linalg.norm(gt_vector * shape))
                else:
                    gt_length_normalized = gt_length_voxels = None
                subdivision_rows.append(
                    {
                        "epoch": epoch,
                        "sample_id": record["sample_id"],
                        "query": int(query),
                        "confidence": float(p[query]),
                        "vessel_distance": float(vessel_distance[query]),
                        "nearest_gt_segment": segment_id,
                        "incident_edge_1_length_normalized": float(lengths_normalized[0]),
                        "incident_edge_2_length_normalized": float(lengths_normalized[1]),
                        "incident_edge_sum_length_normalized": float(lengths_normalized.sum()),
                        "incident_edge_1_length_voxels": float(lengths_voxels[0]),
                        "incident_edge_2_length_voxels": float(lengths_voxels[1]),
                        "incident_edge_sum_length_voxels": float(lengths_voxels.sum()),
                        "underlying_gt_edge_length_normalized": gt_length_normalized,
                        "underlying_gt_edge_length_voxels": gt_length_voxels,
                        "path_to_gt_length_ratio": (
                            float(lengths_voxels.sum() / gt_length_voxels)
                            if gt_length_voxels and gt_length_voxels > 0 else None
                        ),
                    }
                )
        for key, chunks in buffers.items():
            if chunks:
                distributions[key].add(np.concatenate(chunks))
        print(f"epoch {epoch}: read {shard.name} ({records_seen} patches)", flush=True)
    return distributions, proximity, subdivision_rows, records_seen


def _cooccurrence(root, threshold):
    field_prefix = f"t{threshold:g}_"
    counts = defaultdict(Counter)
    pair_counts = defaultdict(Counter)
    totals = Counter()
    path = root / "checkpoint_comparison.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            present = tuple(
                name for name in CORRUPTION_FIELDS
                if float(row.get(field_prefix + name, 0) or 0) > 0
            )
            totals[epoch] += 1
            counts[epoch][present] += 1
            for left_index, left in enumerate(CORRUPTION_FIELDS):
                for right in CORRUPTION_FIELDS[left_index:]:
                    if left in present and right in present:
                        pair_counts[epoch][(left, right)] += 1
    pair_rows, pattern_rows = [], []
    for epoch in sorted(totals):
        for left_index, left in enumerate(CORRUPTION_FIELDS):
            left_total = pair_counts[epoch][(left, left)]
            for right in CORRUPTION_FIELDS[left_index:]:
                right_total = pair_counts[epoch][(right, right)]
                both = pair_counts[epoch][(left, right)]
                union = left_total + right_total - both
                pair_rows.append(
                    {
                        "epoch": epoch,
                        "left": left,
                        "right": right,
                        "patches_with_both": both,
                        "fraction_of_all_patches": both / totals[epoch],
                        "p_right_given_left": both / left_total if left_total else None,
                        "p_left_given_right": both / right_total if right_total else None,
                        "jaccard": both / union if union else None,
                    }
                )
        for pattern, count in counts[epoch].most_common():
            pattern_rows.append(
                {
                    "epoch": epoch,
                    "corruption_types": ";".join(pattern) if pattern else "none",
                    "number_of_types": len(pattern),
                    "patches": count,
                    "fraction": count / totals[epoch],
                }
            )
    return pair_rows, pattern_rows


def _exact_describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    result = {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    result.update({f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES})
    return result


def _format(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown(summary, primary_epoch):
    lines = [
        "# Detailed validation corruption analysis",
        "",
        f"Definitions use node/edge threshold **{summary['settings']['threshold']:.2f}** and "
        f"spatial tolerance **{summary['settings']['spatial_tolerance']:.3f}** in normalized coordinates.",
        "Distances for large populations use exact means and fine-histogram estimated quantiles.",
        "",
    ]
    for epoch in summary["epochs"]:
        values = summary["epochs"][str(epoch)]
        lines.extend([f"## Epoch {epoch}", "", f"Patches read: **{values['patches']}**", ""])
        no_bifurcation = values["proximity"]["unmatched_nodes"]["patches_without_bifurcation"]
        lines.extend([
            f"Patches without any GT bifurcation: **{no_bifurcation:,}** "
            f"({100 * no_bifurcation / values['patches']:.1f}%).",
            "",
        ])
        lines.extend([
            "### Unmatched-node location and confidence",
            "",
            "| Population | Count | Mean confidence | Median confidence | Median vessel distance | Median bifurcation distance | Median boundary distance | Near vessel | Near bifurcation | Near boundary |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for population in ("unmatched_nodes", "extra_nodes", "duplicate_candidates"):
            dist = values["distributions"]
            prox = values["proximity"][population]
            total = prox["total"]
            pct = lambda key: 100 * prox[key] / total if total else None
            lines.append(
                f"| {population.replace('_', ' ')} | {total:,} | "
                f"{_format(dist[population + '/confidence']['mean'])} | "
                f"{_format(dist[population + '/confidence']['q50'])} | "
                f"{_format(dist[population + '/nearest_vessel_distance']['q50'])} | "
                f"{_format(dist[population + '/nearest_bifurcation_distance']['q50'])} | "
                f"{_format(dist[population + '/boundary_distance']['q50'])} | "
                f"{_format(pct('near_vessel'), 1)}% | {_format(pct('near_bifurcation'), 1)}% | "
                f"{_format(pct('near_boundary'), 1)}% |"
            )
        lines.extend([
            "",
            "### Edge-score populations",
            "",
            "| Pair population | Count | Raw mean | Raw median | Raw q95 | Effective mean | Effective median | Effective q95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        dist = values["distributions"]
        for population in ("gt_edges", "matched_nonedges", "unmatched_endpoint_edges"):
            raw = dist[population + "/raw_probability"]
            effective = dist[population + "/effective_probability"]
            lines.append(
                f"| {population.replace('_', ' ')} | {raw['count']:,} | {_format(raw['mean'], 6)} | "
                f"{_format(raw['q50'], 6)} | {_format(raw['q95'], 6)} | "
                f"{_format(effective['mean'], 6)} | {_format(effective['q50'], 6)} | "
                f"{_format(effective['q95'], 6)} |"
            )
        subdivision = values["subdivision"]
        lines.extend([
            "",
            "### Subdivision candidates",
            "",
            f"Candidates: **{subdivision['candidates']}**",
            "",
            "| Measurement | Mean | Median | q10 | q90 |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric, item in subdivision["metrics"].items():
            lines.append(
                f"| {metric.replace('_', ' ')} | {_format(item['mean'])} | "
                f"{_format(item.get('q50'))} | {_format(item.get('q10'))} | {_format(item.get('q90'))} |"
            )
        lines.append("")

    patterns = summary["cooccurrence_patterns"]
    primary_patterns = [row for row in patterns if row["epoch"] == primary_epoch][:15]
    lines.extend([
        f"## Most frequent exact corruption combinations at epoch {primary_epoch}", "",
        "| Types present in patch | Patches | Fraction |", "|---|---:|---:|",
    ])
    for row in primary_patterns:
        lines.append(
            f"| {row['corruption_types'].replace('_', ' ')} | {row['patches']:,} | "
            f"{100 * row['fraction']:.2f}% |"
        )
    lines.extend([
        "",
        "The full pairwise conditional/Jaccard table and every exact combination are saved as CSV files.",
        "`Unmatched nodes` includes every unused query slot, including inactive background queries. "
        "`Extra nodes` restricts this group to confidence >= the analysis threshold and is therefore "
        "the relevant population for designing spatial corruptions. ",
        "`GT edges` are matched query pairs corresponding to a GT edge; `matched nonedges` are two matched "
        "queries without a GT edge; `unmatched endpoint edges` have at least one unmatched query. Edge "
        "confidence distributions include all 7,140 candidate pairs per patch, not only exported edge events.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, nargs="*")
    parser.add_argument("--primary-epoch", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spatial-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    root = args.analysis_root
    output = args.output_dir or root / "detailed_summary"
    output.mkdir(parents=True, exist_ok=True)
    pairs = np.asarray(_torch_load(root / "edge_pairs.pt"), dtype=np.int64)
    folders = sorted(path for path in root.glob("epoch_*" ) if (path / "COMPLETED.json").is_file())
    if args.epochs:
        requested = set(args.epochs)
        folders = [folder for folder in folders if _epoch(folder) in requested]
    if not folders:
        raise FileNotFoundError(f"No completed epoch folders found in {root}")

    summary = {
        "settings": {
            "threshold": args.threshold,
            "spatial_tolerance": args.spatial_tolerance,
            "quantiles": "estimated from fixed-bin histograms for large populations; exact for subdivisions",
        },
        "epochs": {},
    }
    histogram_rows = []
    all_subdivision_rows = []
    for folder in folders:
        epoch = _epoch(folder)
        distributions, proximity, subdivision_rows, patches = _analyse_raw_epoch(
            folder, pairs, args.threshold, args.spatial_tolerance
        )
        all_subdivision_rows.extend(subdivision_rows)
        distribution_summary = {
            population + "/" + metric: distribution.summary()
            for (population, metric), distribution in distributions.items()
        }
        for (population, metric), distribution in distributions.items():
            histogram_rows.extend(distribution.histogram_rows(epoch, population, metric))
        subdivision_metrics = {}
        if subdivision_rows:
            for metric in (
                "incident_edge_1_length_voxels",
                "incident_edge_2_length_voxels",
                "incident_edge_sum_length_voxels",
                "underlying_gt_edge_length_voxels",
                "path_to_gt_length_ratio",
            ):
                subdivision_metrics[metric] = _exact_describe(
                    [row[metric] for row in subdivision_rows if row[metric] is not None]
                )
        summary["epochs"][str(epoch)] = {
            "patches": patches,
            "distributions": distribution_summary,
            "proximity": {
                population: {
                    key: int(proximity[population][key])
                    for key in (
                        "total", "near_gt_node", "near_vessel", "near_bifurcation",
                        "near_boundary", "patches_without_bifurcation",
                    )
                }
                for population in ("unmatched_nodes", "extra_nodes", "duplicate_candidates")
            },
            "subdivision": {"candidates": len(subdivision_rows), "metrics": subdivision_metrics},
        }

    pair_rows, pattern_rows = _cooccurrence(root, args.threshold)
    summary["cooccurrence_patterns"] = pattern_rows
    summary["cooccurrence_pairwise"] = pair_rows
    _write_csv(output / "distribution_histograms.csv", histogram_rows)
    _write_csv(output / "subdivision_candidates.csv", all_subdivision_rows)
    _write_csv(output / "cooccurrence_pairwise.csv", pair_rows)
    _write_csv(output / "cooccurrence_patterns.csv", pattern_rows)
    (output / "detailed_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "detailed_summary.md").write_text(
        _markdown(summary, args.primary_epoch), encoding="utf-8"
    )
    (output / "COMPLETED.json").write_text(
        json.dumps({"epochs": [_epoch(folder) for folder in folders]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Detailed analysis saved to {output}", flush=True)


if __name__ == "__main__":
    main()
