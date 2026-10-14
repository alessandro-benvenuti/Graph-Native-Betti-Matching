#!/usr/bin/env python3
"""Build an empirical, hierarchical RelationFormer corruption model.

The input is the lossless validation export produced by
``analyze_validation_predictions.py``.  This CPU-only pass creates patch-level
joint error profiles, mutually exclusive corruption events, soft-confidence
baselines for retained structure, empirical samples, parametric candidate fits,
and a human-readable report.  It never reloads RelationFormer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Iterable
import warnings

import numpy as np
import torch
from scipy import stats

try:
    from scripts.inspect_validation_corruptions import (
        _point_distances, _probabilities, _segment_distances, _torch_load,
    )
except ModuleNotFoundError:  # Direct execution as ``python scripts/...``.
    from inspect_validation_corruptions import (  # type: ignore
        _point_distances, _probabilities, _segment_distances, _torch_load,
    )


EVENT_TYPES = (
    "missing_node",
    "missing_edge",
    "generic_extra_node",
    "duplicate_node",
    "split_node",
    "subdivision_node",
    "generic_false_edge",
    "loop_closing_false_edge",
)
COUNT_FIELDS = tuple("count_" + name for name in EVENT_TYPES)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

EVENT_FIELDS = (
    "sample_id", "patient_id", "study_split", "epoch", "event_type", "event_index",
    "query", "left_query", "right_query", "matched_gt_node", "nearest_gt_node",
    "nearest_gt_segment", "confidence", "severity", "raw_edge_probability",
    "effective_edge_probability", "left_node_confidence", "right_node_confidence",
    "coordinate_error_normalized", "coordinate_error_voxels",
    "nearest_node_distance_normalized", "nearest_node_distance_voxels",
    "nearest_vessel_distance_normalized", "nearest_vessel_distance_voxels",
    "boundary_distance_normalized", "boundary_distance_voxels",
    "edge_length_normalized", "edge_length_voxels", "gt_edge_length_normalized",
    "gt_edge_length_voxels", "subdivision_fraction", "transverse_offset_normalized",
    "transverse_offset_voxels", "incident_probability_1", "incident_probability_2",
    "incident_edge_count", "incident_probability_mean",
    "incident_effective_probability_mean", "incident_cycle_edge_count",
)

BASELINE_FIELDS = (
    "sample_id", "patient_id", "study_split", "epoch", "category", "query",
    "left_query", "right_query", "gt_node", "confidence", "raw_edge_probability",
    "effective_edge_probability", "left_node_confidence", "right_node_confidence",
    "coordinate_error_normalized", "coordinate_error_voxels", "gt_degree",
    "edge_length_normalized", "edge_length_voxels",
)


class CsvGzipSink:
    def __init__(self, path: Path, fields: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "wt", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.handle.close()


class Reservoir:
    """Deterministic uniform reservoir sampler."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.random = random.Random(int(seed))
        self.values = []
        self.seen = 0

    def add(self, value):
        if value is None or not math.isfinite(float(value)):
            return
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(float(value))
            return
        index = self.random.randrange(self.seen)
        if index < self.capacity:
            self.values[index] = float(value)


class RowReservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.random = random.Random(int(seed))
        self.rows = []
        self.seen = 0

    def add(self, row):
        self.seen += 1
        if len(self.rows) < self.capacity:
            self.rows.append(dict(row))
            return
        index = self.random.randrange(self.seen)
        if index < self.capacity:
            self.rows[index] = dict(row)


def _stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _study_split(patient_id, holdout_fraction):
    value = _stable_seed("patient", patient_id) / float(2**64)
    return "holdout" if value < holdout_fraction else "fit"


def _epoch(path):
    return int(path.name.rsplit("_", 1)[-1])


def _bridges(active, adjacency):
    active = np.asarray(active, dtype=bool)
    adjacency = np.asarray(adjacency, dtype=bool)
    discovery = np.full(len(active), -1, dtype=np.int64)
    low = np.full(len(active), -1, dtype=np.int64)
    parent = np.full(len(active), -1, dtype=np.int64)
    result = set()
    clock = 0

    def visit(node):
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        for raw in np.flatnonzero(adjacency[node] & active):
            neighbour = int(raw)
            if discovery[neighbour] < 0:
                parent[neighbour] = node
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    result.add(tuple(sorted((node, neighbour))))
            elif neighbour != parent[node]:
                low[node] = min(low[node], discovery[neighbour])

    for raw in np.flatnonzero(active):
        node = int(raw)
        if discovery[node] < 0:
            visit(node)
    return result


def _distance(a, b, scale=None):
    vector = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if scale is not None:
        vector = vector * np.asarray(scale, dtype=float)
    return float(np.linalg.norm(vector))


def _empty_event(identity, event_type, event_index):
    return {field: "" for field in EVENT_FIELDS} | identity | {
        "event_type": event_type,
        "event_index": event_index,
    }


def _empty_baseline(identity, category):
    return {field: "" for field in BASELINE_FIELDS} | identity | {"category": category}


def _projection(point, start, end):
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(point - start, vector) / denominator, 0, 1))


def classify_record(record, pairs, provenance, threshold=0.5, tolerance=0.03,
                    holdout_fraction=0.2, background_nodes_per_patch=16,
                    background_edges_per_category_patch=16):
    """Return one patch profile plus event and retained-structure rows."""
    x = np.asarray(record["nodes"], dtype=np.float64)
    gt = np.asarray(record["gt_nodes"], dtype=np.float64)
    gt_edges = np.asarray(record["gt_edges"], dtype=np.int64).reshape(-1, 2)
    shape = np.asarray(record["model_input_shape"], dtype=np.float64)
    matched_queries = np.asarray(record["matched_queries"], dtype=np.int64)
    matched_gt = np.asarray(record["matched_gt"], dtype=np.int64)
    p = _probabilities(record["node_logits"])
    forward = np.asarray(record["edge_logits_forward"], dtype=np.float64)
    reverse = np.asarray(record["edge_logits_reverse"], dtype=np.float64)
    q = _probabilities((forward + reverse) * 0.5)
    n = len(x)

    sample_id = str(record["sample_id"])
    patient_id = str(provenance.get("patient_id", "unknown"))
    epoch = int(record["epoch"])
    split = _study_split(patient_id, holdout_fraction)
    identity = {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "study_split": split,
        "epoch": epoch,
    }

    correspondence = np.full(n, -1, dtype=np.int64)
    correspondence[matched_queries] = matched_gt
    reverse_match = np.full(len(gt), -1, dtype=np.int64)
    reverse_match[matched_gt] = matched_queries
    matched = correspondence >= 0
    active = p >= threshold
    extra = active & ~matched

    adjacency = np.zeros((n, n), dtype=np.float64)
    adjacency[pairs[:, 0], pairs[:, 1]] = q
    adjacency += adjacency.T
    live = (adjacency >= threshold) & active[:, None] & active[None, :]
    np.fill_diagonal(live, False)
    bridges = _bridges(active, live)

    target = np.zeros((n, n), dtype=bool)
    if len(gt_edges):
        left = reverse_match[gt_edges[:, 0]]
        right = reverse_match[gt_edges[:, 1]]
        target[left, right] = target[right, left] = True

    nearest_node_distance, nearest_gt = _point_distances(x, gt)
    vessel_distance, nearest_segment = _segment_distances(x, gt, gt_edges)
    boundary_distance = np.minimum(x, 1 - x).min(axis=1)
    gt_degree = np.bincount(gt_edges.ravel(), minlength=len(gt)) if len(gt) else np.empty(0)
    live_degree = live.sum(1)
    events = []
    baselines = []
    counts = Counter({name: 0 for name in EVENT_TYPES})

    def add_event(event_type, values):
        index = counts[event_type]
        counts[event_type] += 1
        events.append(_empty_event(identity, event_type, index) | values)

    # Matched nodes are either retained soft structure or missing-node events.
    for query, gt_id in zip(matched_queries, matched_gt):
        query, gt_id = int(query), int(gt_id)
        error_norm = _distance(x[query], gt[gt_id])
        error_vox = _distance(x[query], gt[gt_id], shape)
        if active[query]:
            baselines.append(
                _empty_baseline(identity, "correct_node") | {
                    "query": query,
                    "gt_node": gt_id,
                    "confidence": float(p[query]),
                    "coordinate_error_normalized": error_norm,
                    "coordinate_error_voxels": error_vox,
                    "gt_degree": int(gt_degree[gt_id]),
                }
            )
        else:
            add_event("missing_node", {
                "query": query,
                "matched_gt_node": gt_id,
                "confidence": float(p[query]),
                "severity": float(1 - p[query]),
                "coordinate_error_normalized": error_norm,
                "coordinate_error_voxels": error_vox,
            })

    # Mutually exclusive active-extra-node subtypes.
    for query in np.flatnonzero(extra):
        query = int(query)
        segment_id = int(nearest_segment[query])
        neighbours = np.flatnonzero(live[query])
        strict_subdivision = False
        fraction = transverse_norm = transverse_vox = None
        incident_probabilities = []
        gt_edge_norm = gt_edge_vox = None
        if segment_id >= 0 and vessel_distance[query] < tolerance and len(neighbours) == 2:
            gt_left, gt_right = map(int, gt_edges[segment_id])
            expected = {int(reverse_match[gt_left]), int(reverse_match[gt_right])}
            strict_subdivision = set(map(int, neighbours)) == expected
            if strict_subdivision:
                direct_live = bool(live[int(reverse_match[gt_left]), int(reverse_match[gt_right])])
                strict_subdivision = not direct_live
            if strict_subdivision:
                start, end = gt[gt_left], gt[gt_right]
                fraction = _projection(x[query], start, end)
                projection = start + fraction * (end - start)
                transverse_norm = _distance(x[query], projection)
                transverse_vox = _distance(x[query], projection, shape)
                gt_edge_norm = _distance(start, end)
                gt_edge_vox = _distance(start, end, shape)
                incident_probabilities = [float(adjacency[query, node]) for node in neighbours]

        nearest_id = int(nearest_gt[query])
        nearest_query = int(reverse_match[nearest_id]) if nearest_id >= 0 else -1
        near_node = nearest_id >= 0 and nearest_node_distance[query] < tolerance
        disconnected_from_source = near_node and not bool(live[query, nearest_query])
        if strict_subdivision:
            event_type = "subdivision_node"
        elif disconnected_from_source:
            event_type = "split_node"
        elif near_node:
            event_type = "duplicate_node"
        else:
            event_type = "generic_extra_node"
        values = {
            "query": query,
            "nearest_gt_node": nearest_id,
            "nearest_gt_segment": segment_id,
            "confidence": float(p[query]),
            "severity": float(p[query]),
            "nearest_node_distance_normalized": float(nearest_node_distance[query]),
            "nearest_node_distance_voxels": (
                _distance(x[query], gt[nearest_id], shape) if nearest_id >= 0 else ""
            ),
            "nearest_vessel_distance_normalized": float(vessel_distance[query]),
            "boundary_distance_normalized": float(boundary_distance[query]),
            "boundary_distance_voxels": float(
                np.minimum(x[query] * shape, (1 - x[query]) * shape).min()
            ),
            "incident_edge_count": int(len(neighbours)),
            "incident_probability_mean": (
                float(np.mean(adjacency[query, neighbours])) if len(neighbours) else 0.0
            ),
            "incident_effective_probability_mean": (
                float(np.mean(p[query] * p[neighbours] * adjacency[query, neighbours]))
                if len(neighbours) else 0.0
            ),
            "incident_cycle_edge_count": int(sum(
                tuple(sorted((query, int(neighbour)))) not in bridges
                for neighbour in neighbours
            )),
        }
        if segment_id >= 0:
            gt_left, gt_right = map(int, gt_edges[segment_id])
            fraction_for_distance = _projection(x[query], gt[gt_left], gt[gt_right])
            projection = gt[gt_left] + fraction_for_distance * (gt[gt_right] - gt[gt_left])
            values["nearest_vessel_distance_voxels"] = _distance(x[query], projection, shape)
        if strict_subdivision:
            values.update({
                "gt_edge_length_normalized": gt_edge_norm,
                "gt_edge_length_voxels": gt_edge_vox,
                "subdivision_fraction": fraction,
                "transverse_offset_normalized": transverse_norm,
                "transverse_offset_voxels": transverse_vox,
                "incident_probability_1": incident_probabilities[0],
                "incident_probability_2": incident_probabilities[1],
            })
        add_event(event_type, values)

    # GT edges are retained, missing, or suppressed by a missing endpoint.
    for gt_left, gt_right in gt_edges:
        left, right = int(reverse_match[int(gt_left)]), int(reverse_match[int(gt_right)])
        raw = float(adjacency[left, right])
        effective = float(p[left] * p[right] * raw)
        gt_length_norm = _distance(gt[int(gt_left)], gt[int(gt_right)])
        gt_length_vox = _distance(gt[int(gt_left)], gt[int(gt_right)], shape)
        if active[left] and active[right] and raw >= threshold:
            baselines.append(
                _empty_baseline(identity, "correct_edge") | {
                    "left_query": left,
                    "right_query": right,
                    "raw_edge_probability": raw,
                    "effective_edge_probability": effective,
                    "left_node_confidence": float(p[left]),
                    "right_node_confidence": float(p[right]),
                    "edge_length_normalized": gt_length_norm,
                    "edge_length_voxels": gt_length_vox,
                }
            )
        elif active[left] and active[right]:
            add_event("missing_edge", {
                "left_query": left,
                "right_query": right,
                "raw_edge_probability": raw,
                "effective_edge_probability": effective,
                "left_node_confidence": float(p[left]),
                "right_node_confidence": float(p[right]),
                "severity": float(1 - raw),
                "gt_edge_length_normalized": gt_length_norm,
                "gt_edge_length_voxels": gt_length_vox,
            })

    # False-edge events concern retained matched structure. Edges incident to an
    # inserted node are attributes of that node-insertion event, preventing a
    # subdivision or duplicate from being sampled a second time as false edges.
    retained_pair = matched[:, None] & matched[None, :]
    for left, right in np.argwhere(np.triu(live & ~target & retained_pair, 1)):
        left, right = int(left), int(right)
        raw = float(adjacency[left, right])
        effective = float(p[left] * p[right] * raw)
        in_cycle = tuple(sorted((left, right))) not in bridges
        event_type = "loop_closing_false_edge" if in_cycle else "generic_false_edge"
        add_event(event_type, {
            "left_query": left,
            "right_query": right,
            "raw_edge_probability": raw,
            "effective_edge_probability": effective,
            "left_node_confidence": float(p[left]),
            "right_node_confidence": float(p[right]),
            "severity": effective,
            "edge_length_normalized": _distance(x[left], x[right]),
            "edge_length_voxels": _distance(x[left], x[right], shape),
        })

    # Background query rows are sampled later; return them separately.
    background = []

    def sampled_indices(mask, capacity, label):
        indices = np.flatnonzero(mask)
        if len(indices) <= capacity:
            return indices
        generator = np.random.default_rng(_stable_seed(sample_id, epoch, label))
        return np.sort(generator.choice(indices, size=capacity, replace=False))

    inactive_queries = sampled_indices(
        ~matched & ~active, int(background_nodes_per_patch), "inactive_query"
    )
    for query in inactive_queries:
        background.append(
            _empty_baseline(identity, "inactive_query") | {
                "query": int(query),
                "confidence": float(p[query]),
            }
        )

    matched_pair = matched[pairs[:, 0]] & matched[pairs[:, 1]]
    target_pair = target[pairs[:, 0], pairs[:, 1]]
    for category, mask in (
        ("background_matched_nonedge", matched_pair & ~target_pair & ~live[pairs[:, 0], pairs[:, 1]]),
        ("background_unmatched_endpoint", ~matched_pair & ~live[pairs[:, 0], pairs[:, 1]]),
    ):
        for pair_index in sampled_indices(
            mask, int(background_edges_per_category_patch), category
        ):
            left, right = map(int, pairs[pair_index])
            raw = float(q[pair_index])
            background.append(
                _empty_baseline(identity, category) | {
                    "left_query": left,
                    "right_query": right,
                    "raw_edge_probability": raw,
                    "effective_edge_probability": float(p[left] * p[right] * raw),
                    "left_node_confidence": float(p[left]),
                    "right_node_confidence": float(p[right]),
                }
            )

    node_target = matched.astype(np.float64)
    pair_target = target[pairs[:, 0], pairs[:, 1]].astype(np.float64)
    effective_pairs = p[pairs[:, 0]] * p[pairs[:, 1]] * q
    displacement = x[matched_queries] - gt[matched_gt]
    node_cost = float(np.abs(p - node_target).sum())
    coordinate_cost = float((p[matched_queries] * np.linalg.norm(displacement, axis=1)).sum())
    edge_cost = float(np.abs(effective_pairs - pair_target).sum())
    graph_node_bin = "0-9" if len(gt) < 10 else "10-19" if len(gt) < 20 else "20-29" if len(gt) < 30 else "30+"
    graph_edge_bin = "0-9" if len(gt_edges) < 10 else "10-19" if len(gt_edges) < 20 else "20-29" if len(gt_edges) < 30 else "30+"
    positive = tuple(name for name in EVENT_TYPES if counts[name] > 0)
    profile = identity | {
        "gt_nodes": len(gt),
        "gt_edges": len(gt_edges),
        "gt_node_bin": graph_node_bin,
        "gt_edge_bin": graph_edge_bin,
        "number_corruption_families": len(positive),
        "corruption_pattern": ";".join(positive) if positive else "none",
        **{"count_" + name: int(counts[name]) for name in EVENT_TYPES},
        "node_cost_per_gt_node": node_cost / max(1, len(gt)),
        "coordinate_cost_per_gt_node": coordinate_cost / max(1, len(gt)),
        "edge_effective_cost_per_gt_edge": edge_cost / max(1, len(gt_edges)),
        "difficulty_score": (
            node_cost / max(1, len(gt))
            + coordinate_cost / max(1, len(gt))
            + edge_cost / max(1, len(gt_edges))
        ),
    }
    return profile, events, baselines, background


def _describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, **{f"q{int(q*100):02d}": None for q in QUANTILES}}
    result = {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std())}
    result.update({f"q{int(q*100):02d}": float(np.quantile(values, q)) for q in QUANTILES})
    return result


def _fit_continuous(train, test, support=None):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 50 or len(test) < 20:
        return []
    if float(np.ptp(train)) < 1e-12:
        return []
    epsilon = 1e-6
    candidates = []

    def record(name, parameters, train_logpdf, test_logpdf, cdf, evaluation_values=None):
        train_ll = float(np.sum(train_logpdf))
        test_ll = float(np.mean(test_logpdf))
        parameter_count = sum(key not in {"lower", "upper"} for key in parameters)
        evaluated = test if evaluation_values is None else evaluation_values
        ks = float(stats.kstest(evaluated, cdf).statistic)
        candidates.append({
            "family": name,
            "parameters": {key: float(value) for key, value in parameters.items()},
            "train_log_likelihood": train_ll,
            "heldout_mean_log_likelihood": test_ll,
            "heldout_ks": ks,
            "aic": float(2 * parameter_count - 2 * train_ll),
            "bic": float(parameter_count * math.log(len(train)) - 2 * train_ll),
        })

    warning_context = warnings.catch_warnings()
    try:
        warning_context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        if support is not None:
            lower, upper = map(float, support)
            width = upper - lower
            scaled_train = np.clip((train - lower) / width, epsilon, 1 - epsilon)
            scaled_test = np.clip((test - lower) / width, epsilon, 1 - epsilon)
            mean = float(scaled_train.mean())
            variance = float(scaled_train.var())
            concentration = mean * (1 - mean) / max(variance, epsilon) - 1
            if concentration <= 0:
                raise ValueError("scaled Beta method-of-moments fit is undefined")
            alpha, beta = mean * concentration, (1 - mean) * concentration
            record(
                "scaled_beta", {"alpha": alpha, "beta": beta, "lower": lower, "upper": upper},
                stats.beta.logpdf(scaled_train, alpha, beta) - math.log(width),
                stats.beta.logpdf(scaled_test, alpha, beta) - math.log(width),
                lambda value: stats.beta.cdf(
                    np.clip((np.asarray(value) - lower) / width, epsilon, 1 - epsilon), alpha, beta
                ),
            )
            logits_train = np.log(scaled_train / (1 - scaled_train))
            mu, sigma = float(logits_train.mean()), float(max(logits_train.std(), epsilon))
            def logit_logpdf(values):
                values = np.clip((np.asarray(values) - lower) / width, epsilon, 1 - epsilon)
                logits = np.log(values / (1 - values))
                return (
                    stats.norm.logpdf(logits, mu, sigma)
                    - np.log(values) - np.log1p(-values) - math.log(width)
                )
            record(
                "scaled_logit_normal", {"mu": mu, "sigma": sigma, "lower": lower, "upper": upper},
                logit_logpdf(train), logit_logpdf(test),
                lambda value: stats.norm.cdf(
                    np.log(
                        np.clip((np.asarray(value) - lower) / width, epsilon, 1 - epsilon)
                        / (1 - np.clip((np.asarray(value) - lower) / width, epsilon, 1 - epsilon))
                    ),
                    mu, sigma,
                ),
            )
        else:
            positive_train = train[train > 0]
            positive_test = test[test > 0]
            if len(positive_train) < 50 or len(positive_test) < 20:
                return []
            shape, _, scale = stats.gamma.fit(positive_train, floc=0)
            record(
                "gamma", {"shape": shape, "scale": scale},
                stats.gamma.logpdf(positive_train, shape, scale=scale),
                stats.gamma.logpdf(positive_test, shape, scale=scale),
                lambda value: stats.gamma.cdf(value, shape, scale=scale),
                positive_test,
            )
            sigma, _, scale = stats.lognorm.fit(positive_train, floc=0)
            record(
                "lognormal", {"sigma": sigma, "scale": scale},
                stats.lognorm.logpdf(positive_train, sigma, scale=scale),
                stats.lognorm.logpdf(positive_test, sigma, scale=scale),
                lambda value: stats.lognorm.cdf(value, sigma, scale=scale),
                positive_test,
            )
    except Exception:
        return []
    finally:
        warning_context.__exit__(None, None, None)
    return sorted(candidates, key=lambda item: (-item["heldout_mean_log_likelihood"], item["heldout_ks"]))


def _fit_counts(train, test):
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if len(train) < 50 or len(test) < 20:
        return []
    presence = float((train > 0).mean())
    positives = train[train > 0]
    if not len(positives):
        return []
    epsilon = 1e-12
    candidates = []
    for family in ("hurdle_poisson", "hurdle_negative_binomial"):
        if family == "hurdle_poisson":
            mean = float(max(positives.mean(), epsilon))
            parameters = {"presence_probability": presence, "positive_mean": mean}
            base_pmf = lambda value: stats.poisson.pmf(value, mean)
            normalizer = 1 - stats.poisson.pmf(0, mean)
        else:
            mean = float(positives.mean())
            variance = float(positives.var())
            if variance <= mean + 1e-8:
                continue
            n = mean * mean / (variance - mean)
            probability = n / (n + mean)
            parameters = {"presence_probability": presence, "n": n, "probability": probability}
            base_pmf = lambda value: stats.nbinom.pmf(value, n, probability)
            normalizer = 1 - stats.nbinom.pmf(0, n, probability)
        probability_test = np.where(
            test == 0,
            1 - presence,
            presence * base_pmf(test) / max(normalizer, epsilon),
        )
        mean_ll = float(np.log(np.maximum(probability_test, epsilon)).mean())
        candidates.append({
            "family": family,
            "parameters": parameters,
            "heldout_mean_log_likelihood": mean_ll,
        })
    return sorted(candidates, key=lambda item: -item["heldout_mean_log_likelihood"])


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _register(bank, epoch, population, metric, split, value, capacity):
    if value == "" or value is None:
        return
    for target_split in (str(split), "all"):
        key = (int(epoch), str(population), str(metric), target_split)
        if key not in bank:
            bank[key] = Reservoir(capacity, _stable_seed(*key))
        bank[key].add(value)


def _profile_statistics(profiles):
    occurrence_rows, pattern_rows, correlation_rows = [], [], []
    cooccurrence_rows, conditional_rows, family_rows, profile_correlation_rows = [], [], [], []
    for epoch in sorted({int(row["epoch"]) for row in profiles}):
        rows = [row for row in profiles if int(row["epoch"]) == epoch]
        patterns = Counter(row["corruption_pattern"] for row in rows)
        for pattern, count in patterns.most_common():
            pattern_rows.append({
                "epoch": epoch, "pattern": pattern,
                "number_corruption_families": 0 if pattern == "none" else len(pattern.split(";")),
                "patches": count, "fraction": count / len(rows),
            })
        family_counts = Counter(int(row["number_corruption_families"]) for row in rows)
        for count, patches in sorted(family_counts.items()):
            family_rows.append({
                "epoch": epoch, "number_corruption_families": count,
                "patches": patches, "fraction": patches / len(rows),
            })
        matrix = np.asarray([[float(row[field]) for field in COUNT_FIELDS] for row in rows])
        for index, event in enumerate(EVENT_TYPES):
            values = matrix[:, index]
            description = _describe(values)
            occurrence_rows.append({
                "epoch": epoch,
                "event_type": event,
                "patch_fraction": float((values > 0).mean()),
                **description,
            })
        for left_index, left in enumerate(EVENT_TYPES):
            for right_index, right in enumerate(EVENT_TYPES):
                if left_index == right_index:
                    value = 1.0 if np.unique(matrix[:, left_index]).size > 1 else math.nan
                elif np.unique(matrix[:, left_index]).size < 2 or np.unique(matrix[:, right_index]).size < 2:
                    value = math.nan
                else:
                    value = float(stats.spearmanr(matrix[:, left_index], matrix[:, right_index]).statistic)
                correlation_rows.append({
                    "epoch": epoch, "left": left, "right": right,
                    "spearman": value if math.isfinite(value) else None,
                })
        present = matrix > 0
        for left_index, left in enumerate(EVENT_TYPES):
            left_total = int(present[:, left_index].sum())
            for right_index in range(left_index, len(EVENT_TYPES)):
                right = EVENT_TYPES[right_index]
                right_total = int(present[:, right_index].sum())
                both = int((present[:, left_index] & present[:, right_index]).sum())
                union = left_total + right_total - both
                cooccurrence_rows.append({
                    "epoch": epoch, "left": left, "right": right,
                    "patches_with_both": both, "fraction_of_patches": both / len(rows),
                    "p_right_given_left": both / left_total if left_total else None,
                    "p_left_given_right": both / right_total if right_total else None,
                    "jaccard": both / union if union else None,
                })
        for node_bin in ("0-9", "10-19", "20-29", "30+"):
            selected = [row for row in rows if row["gt_node_bin"] == node_bin]
            if not selected:
                continue
            for event, field in zip(EVENT_TYPES, COUNT_FIELDS):
                values = np.asarray([float(row[field]) for row in selected])
                conditional_rows.append({
                    "epoch": epoch, "gt_node_bin": node_bin, "patches": len(selected),
                    "event_type": event, "patch_fraction": float((values > 0).mean()),
                    **_describe(values),
                })
        profile_fields = (
            "gt_nodes", "gt_edges", "number_corruption_families", *COUNT_FIELDS,
            "node_cost_per_gt_node", "coordinate_cost_per_gt_node",
            "edge_effective_cost_per_gt_edge", "difficulty_score",
        )
        for left_index, left in enumerate(profile_fields):
            left_values = np.asarray([float(row[left]) for row in rows])
            for right in profile_fields[left_index:]:
                right_values = np.asarray([float(row[right]) for row in rows])
                if np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
                    value = math.nan
                else:
                    value = float(stats.spearmanr(left_values, right_values).statistic)
                profile_correlation_rows.append({
                    "epoch": epoch, "left": left, "right": right,
                    "spearman": value if math.isfinite(value) else None,
                })
    return (
        occurrence_rows, pattern_rows, correlation_rows, cooccurrence_rows,
        conditional_rows, family_rows, profile_correlation_rows,
    )


def _fit_support(population, metric, threshold):
    if metric in {"effective_edge_probability", "subdivision_fraction"}:
        return (0.0, 1.0)
    if metric == "confidence":
        if population in {"inactive_query", "missing_node"}:
            return (0.0, threshold)
        if population in {"correct_node", "generic_extra_node", "duplicate_node", "split_node", "subdivision_node"}:
            return (threshold, 1.0)
    if metric == "raw_edge_probability":
        if population == "missing_edge":
            return (0.0, threshold)
        if population in {"correct_edge", "generic_false_edge", "loop_closing_false_edge"}:
            return (threshold, 1.0)
        return (0.0, 1.0)
    if metric == "severity":
        if population in {"missing_node", "missing_edge", "generic_extra_node", "duplicate_node", "split_node", "subdivision_node"}:
            return (threshold, 1.0)
        if population in {"generic_false_edge", "loop_closing_false_edge"}:
            return (0.0, 1.0)
    if metric in {"incident_probability_mean", "incident_effective_probability_mean"}:
        return (0.0, 1.0)
    return None


def _markdown(profiles, occurrence, patterns, empirical, fits, primary_epoch):
    lines = [
        "# Empirical corruption meta-distribution", "",
        "The same validation patches are followed across checkpoints; checkpoint rows are not treated as independent patients.",
        "Missing edges are counted only when both endpoint nodes are retained. Node insertions and false-edge insertions use mutually exclusive subtypes.",
        "",
    ]
    epochs = sorted({int(row["epoch"]) for row in profiles})
    lines.extend([
        "## Patch-level corruption counts", "",
        "| Epoch | Event | P(any) | Mean | Median | q90 |", "|---:|---|---:|---:|---:|---:|",
    ])
    for row in occurrence:
        lines.append(
            f"| {row['epoch']} | {row['event_type'].replace('_', ' ')} | {100*row['patch_fraction']:.1f}% | "
            f"{row['mean']:.3f} | {row['q50']:.3f} | {row['q90']:.3f} |"
        )
    lines.extend(["", f"## Most frequent joint profiles at epoch {primary_epoch}", "",
                  "| Pattern | Patches | Fraction |", "|---|---:|---:|"])
    for row in [item for item in patterns if item["epoch"] == primary_epoch][:15]:
        lines.append(f"| {row['pattern'].replace('_', ' ')} | {row['patches']:,} | {100*row['fraction']:.2f}% |")
    lines.extend(["", "## Soft retained-structure and event severities", "",
                  "| Epoch | Population | Metric | Count | Mean | Median | q10 | q90 |",
                  "|---:|---|---|---:|---:|---:|---:|---:|"])
    selected = {
        ("correct_node", "confidence"), ("correct_node", "coordinate_error_voxels"),
        ("correct_edge", "raw_edge_probability"), ("correct_edge", "effective_edge_probability"),
        ("inactive_query", "confidence"),
    }
    selected.update((event, "severity") for event in EVENT_TYPES)
    for item in empirical:
        if item["study_split"] != "all" or (item["population"], item["metric"]) not in selected:
            continue
        lines.append(
            f"| {item['epoch']} | {item['population'].replace('_', ' ')} | {item['metric'].replace('_', ' ')} | "
            f"{item['count']:,} | {item['mean']:.4f} | {item['q50']:.4f} | {item['q10']:.4f} | {item['q90']:.4f} |"
        )
    lines.extend(["", "## Parametric-fit diagnostics", "",
                  "The winning family below is selected by held-out mean log likelihood. A poor KS statistic supports retaining empirical sampling.", "",
                  "| Epoch | Population | Metric | Winner | Held-out mean log likelihood | KS |",
                  "|---:|---|---|---|---:|---:|"])
    for item in fits:
        if not item["candidates"]:
            continue
        winner = item["candidates"][0]
        lines.append(
            f"| {item['epoch']} | {item['population'].replace('_', ' ')} | {item['metric'].replace('_', ' ')} | "
            f"{winner['family']} | {winner['heldout_mean_log_likelihood']:.4f} | "
            f"{winner.get('heldout_ks', float('nan')):.4f} |"
        )
    lines.extend(["", "Machine-readable empirical samples, full fit parameters, count correlations and event tables accompany this report.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, nargs="*")
    parser.add_argument("--primary-epoch", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--spatial-tolerance", type=float, default=0.03)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--sample-capacity", type=int, default=200_000)
    parser.add_argument("--background-rows-per-category-epoch", type=int, default=100_000)
    parser.add_argument("--background-nodes-per-patch", type=int, default=16)
    parser.add_argument("--background-edges-per-category-patch", type=int, default=16)
    parser.add_argument("--max-records-per-epoch", type=int)
    parser.add_argument("--replace-output", action="store_true")
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise ValueError("--holdout-fraction must lie in (0,1)")
    root = args.analysis_root
    output = args.output_dir or root / "empirical_corruption_model"
    if output.exists() and any(output.iterdir()):
        if not args.replace_output:
            raise FileExistsError(
                f"Output is not empty: {output}; use a new directory or --replace-output"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs = np.asarray(_torch_load(root / "edge_pairs.pt"), dtype=np.int64)
    manifest = json.loads((root / "validation_manifest.json").read_text(encoding="utf-8"))
    provenance = {str(row["sample_id"]): (row.get("provenance") or {}) for row in manifest}
    folders = sorted(path for path in root.glob("epoch_*") if (path / "COMPLETED.json").is_file())
    if args.epochs:
        requested = set(args.epochs)
        folders = [folder for folder in folders if _epoch(folder) in requested]
    if not folders:
        raise FileNotFoundError(f"No completed epochs in {root}")

    event_sink = CsvGzipSink(output / "corruption_events.csv.gz", EVENT_FIELDS)
    baseline_sink = CsvGzipSink(output / "correct_elements.csv.gz", BASELINE_FIELDS)
    profile_fields = (
        "sample_id", "patient_id", "study_split", "epoch", "gt_nodes", "gt_edges",
        "gt_node_bin", "gt_edge_bin", "number_corruption_families", "corruption_pattern",
        *COUNT_FIELDS, "node_cost_per_gt_node", "coordinate_cost_per_gt_node",
        "edge_effective_cost_per_gt_edge", "difficulty_score",
    )
    profiles = []
    sample_bank = {}
    background_rows = {}

    for folder in folders:
        epoch = _epoch(folder)
        epoch_records = 0
        for shard in sorted((folder / "raw").glob("shard_*.pt")):
            for record in _torch_load(shard):
                if args.max_records_per_epoch is not None and epoch_records >= args.max_records_per_epoch:
                    break
                sample_id = str(record["sample_id"])
                profile, events, baselines, background = classify_record(
                    record, pairs, provenance.get(sample_id, {}), args.threshold,
                    args.spatial_tolerance, args.holdout_fraction,
                    args.background_nodes_per_patch,
                    args.background_edges_per_category_patch,
                )
                profiles.append(profile)
                epoch_records += 1
                for event in events:
                    event_sink.write(event)
                    for metric in EVENT_FIELDS:
                        if metric in {"confidence", "severity", "raw_edge_probability", "effective_edge_probability",
                                      "coordinate_error_voxels", "nearest_node_distance_voxels",
                                      "nearest_vessel_distance_voxels", "edge_length_voxels",
                                      "gt_edge_length_voxels", "subdivision_fraction", "transverse_offset_voxels",
                                      "incident_edge_count", "incident_probability_mean",
                                      "incident_effective_probability_mean", "incident_cycle_edge_count"}:
                            _register(sample_bank, epoch, event["event_type"], metric, event["study_split"], event.get(metric), args.sample_capacity)
                for row in baselines:
                    baseline_sink.write(row)
                    for metric in ("confidence", "raw_edge_probability", "effective_edge_probability",
                                   "coordinate_error_voxels", "edge_length_voxels"):
                        _register(sample_bank, epoch, row["category"], metric, row["study_split"], row.get(metric), args.sample_capacity)
                for row in background:
                    key = (epoch, row["category"])
                    if key not in background_rows:
                        background_rows[key] = RowReservoir(
                            args.background_rows_per_category_epoch, _stable_seed(*key)
                        )
                    background_rows[key].add(row)
                    for metric in ("confidence", "raw_edge_probability", "effective_edge_probability"):
                        _register(sample_bank, epoch, row["category"], metric, row["study_split"], row.get(metric), args.sample_capacity)
            print(f"epoch {epoch}: processed {shard.name} ({len(profiles)} total profiles)", flush=True)
            if args.max_records_per_epoch is not None and epoch_records >= args.max_records_per_epoch:
                break

    for reservoir in background_rows.values():
        for row in reservoir.rows:
            baseline_sink.write(row)
    event_sink.close()
    baseline_sink.close()
    with gzip.open(output / "patch_profiles.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=profile_fields)
        writer.writeheader()
        writer.writerows(profiles)

    (
        occurrence, patterns, correlations, cooccurrence, conditional_occurrence,
        family_distribution, profile_correlations,
    ) = _profile_statistics(profiles)
    _write_csv(output / "count_occurrence.csv", occurrence)
    _write_csv(output / "joint_patterns.csv", patterns)
    _write_csv(output / "count_correlations.csv", correlations)
    _write_csv(output / "cooccurrence.csv", cooccurrence)
    _write_csv(output / "conditional_count_occurrence.csv", conditional_occurrence)
    _write_csv(output / "family_count_distribution.csv", family_distribution)
    _write_csv(output / "profile_metric_correlations.csv", profile_correlations)

    empirical = []
    arrays = {}
    grouped = defaultdict(dict)
    for (epoch, population, metric, split), reservoir in sample_bank.items():
        grouped[(epoch, population, metric)][split] = reservoir
        key = f"e{epoch}__{population}__{metric}__{split}"
        arrays[key] = np.asarray(reservoir.values, dtype=np.float32)
        empirical.append({
            "epoch": epoch, "population": population, "metric": metric,
            "study_split": split, "seen": reservoir.seen, **_describe(reservoir.values),
        })
    np.savez_compressed(output / "empirical_samples.npz", **arrays)
    (output / "empirical_distributions.json").write_text(
        json.dumps(empirical, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    fits = []
    for (epoch, population, metric), splits in sorted(grouped.items()):
        if "fit" not in splits or "holdout" not in splits:
            continue
        candidates = _fit_continuous(
            splits["fit"].values, splits["holdout"].values,
            _fit_support(population, metric, args.threshold),
        )
        fits.append({"epoch": epoch, "population": population, "metric": metric, "candidates": candidates})
    for epoch in sorted({int(row["epoch"]) for row in profiles}):
        epoch_rows = [row for row in profiles if int(row["epoch"]) == epoch]
        for event, field in zip(EVENT_TYPES, COUNT_FIELDS):
            train = [row[field] for row in epoch_rows if row["study_split"] == "fit"]
            test = [row[field] for row in epoch_rows if row["study_split"] == "holdout"]
            fits.append({
                "epoch": epoch, "population": event, "metric": "count",
                "candidates": _fit_counts(train, test),
            })
    (output / "parametric_fits.json").write_text(
        json.dumps(fits, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = _markdown(profiles, occurrence, patterns, empirical, fits, args.primary_epoch)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "model_settings.json").write_text(json.dumps({
        "threshold": args.threshold,
        "spatial_tolerance": args.spatial_tolerance,
        "holdout_fraction": args.holdout_fraction,
        "epochs": [_epoch(folder) for folder in folders],
        "event_types": EVENT_TYPES,
        "subdivision_definition": "degree-two extra near GT segment; neighbours are matched GT-edge endpoints; direct edge absent",
        "missing_edge_definition": "GT edge below threshold with both endpoint nodes retained",
        "node_subtype_priority": ["subdivision", "split", "duplicate", "generic_extra"],
        "false_edge_definition": "live non-GT edge between retained matched nodes; loop-closing and generic are exclusive",
        "node_insertion_edges": "live edges incident to inserted nodes are stored as node-event severity attributes, not separate false-edge events",
        "background_sampling": {
            "nodes_per_patch": args.background_nodes_per_patch,
            "edges_per_category_patch": args.background_edges_per_category_patch,
            "rows_per_category_epoch": args.background_rows_per_category_epoch,
        },
    }, indent=2) + "\n", encoding="utf-8")
    (output / "COMPLETED.json").write_text(json.dumps({
        "profiles": len(profiles), "events": sum(sum(row[field] for field in COUNT_FIELDS) for row in profiles),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Empirical corruption model saved to {output}", flush=True)


if __name__ == "__main__":
    main()
