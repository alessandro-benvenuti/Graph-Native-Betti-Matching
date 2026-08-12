"""Dependency-light graph metrics for RelationFormer evaluation.

The AP/AR protocol retains the useful decisions from the trusted legacy
evaluator while correcting its double conversion of node boxes. Coordinates
are normalized and ordered as D/H/W throughout this module.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10, endpoint=True)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101, endpoint=True)


def _array(values, *, columns=None, dtype=np.float32):
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        shape = (0, columns) if columns is not None else (0,)
        return np.empty(shape, dtype=dtype)
    if columns is not None:
        return array.reshape(-1, columns)
    return array.reshape(-1)


def center_size_to_corners(boxes, *, clip=True):
    """Convert ``[D,H,W,size_D,size_H,size_W]`` to legacy 3D IoU order."""

    boxes = _array(boxes, columns=6)
    center = boxes[:, :3]
    radius = 0.5 * boxes[:, 3:]
    low = center - radius
    high = center + radius
    corners = np.stack(
        (low[:, 0], low[:, 1], high[:, 0], high[:, 1], low[:, 2], high[:, 2]),
        axis=1,
    )
    return np.clip(corners, 0.0, 1.0) if clip else corners


def edge_boxes(nodes, edges, *, half_width=0.1, clip=True):
    """Represent graph edges by the fixed-width boxes used by the baseline."""

    nodes = _array(nodes, columns=3)
    edges = canonical_edges(edges, len(nodes))
    if not edges:
        return np.empty((0, 6), dtype=np.float32)
    indices = np.asarray(edges, dtype=np.int64)
    endpoints = nodes[indices]
    low = endpoints.min(axis=1) - float(half_width)
    high = endpoints.max(axis=1) + float(half_width)
    corners = np.stack(
        (low[:, 0], low[:, 1], high[:, 0], high[:, 1], low[:, 2], high[:, 2]),
        axis=1,
    )
    return np.clip(corners, 0.0, 1.0) if clip else corners


def box_iou_3d(left, right):
    left = _array(left, columns=6)
    right = _array(right, columns=6)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    left_volume = (
        np.clip(left[:, 2] - left[:, 0], a_min=0, a_max=None)
        * np.clip(left[:, 3] - left[:, 1], a_min=0, a_max=None)
        * np.clip(left[:, 5] - left[:, 4], a_min=0, a_max=None)
    )
    right_volume = (
        np.clip(right[:, 2] - right[:, 0], a_min=0, a_max=None)
        * np.clip(right[:, 3] - right[:, 1], a_min=0, a_max=None)
        * np.clip(right[:, 5] - right[:, 4], a_min=0, a_max=None)
    )
    low_d = np.maximum(left[:, None, 0], right[None, :, 0])
    low_h = np.maximum(left[:, None, 1], right[None, :, 1])
    high_d = np.minimum(left[:, None, 2], right[None, :, 2])
    high_h = np.minimum(left[:, None, 3], right[None, :, 3])
    low_w = np.maximum(left[:, None, 4], right[None, :, 4])
    high_w = np.minimum(left[:, None, 5], right[None, :, 5])
    intersection = (
        np.clip(high_d - low_d, a_min=0, a_max=None)
        * np.clip(high_h - low_h, a_min=0, a_max=None)
        * np.clip(high_w - low_w, a_min=0, a_max=None)
    )
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def detection_ap_ar(
    predicted_boxes,
    predicted_scores,
    target_boxes,
    *,
    iou_thresholds=IOU_THRESHOLDS,
    max_detections=40,
):
    """Return per-image COCO-style mAP/mAR for one foreground class."""

    predicted_boxes = _array(predicted_boxes, columns=6)
    predicted_scores = _array(predicted_scores)
    target_boxes = _array(target_boxes, columns=6)
    if len(predicted_boxes) != len(predicted_scores):
        raise ValueError("Every predicted box must have exactly one score")
    if not len(target_boxes):
        return float("nan"), float("nan")

    order = np.argsort(-predicted_scores, kind="mergesort")[: int(max_detections)]
    predicted_boxes = predicted_boxes[order]
    predicted_scores = predicted_scores[order]
    overlaps = box_iou_3d(predicted_boxes, target_boxes)
    average_precisions = []
    recalls = []
    for threshold in np.asarray(iou_thresholds, dtype=np.float64):
        matched = np.zeros(len(target_boxes), dtype=bool)
        true_positive = np.zeros(len(predicted_boxes), dtype=np.float32)
        for prediction_index in range(len(predicted_boxes)):
            candidates = np.where(
                np.logical_and(~matched, overlaps[prediction_index] >= threshold)
            )[0]
            if candidates.size:
                best = candidates[np.argmax(overlaps[prediction_index, candidates])]
                matched[best] = True
                true_positive[prediction_index] = 1.0
        false_positive = 1.0 - true_positive
        tp = np.cumsum(true_positive)
        fp = np.cumsum(false_positive)
        recall = tp / float(len(target_boxes))
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        for index in range(len(precision) - 1, 0, -1):
            precision[index - 1] = max(precision[index - 1], precision[index])
        sampled_precision = np.zeros(len(RECALL_THRESHOLDS), dtype=np.float32)
        indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        valid = indices < len(precision)
        sampled_precision[valid] = precision[indices[valid]]
        average_precisions.append(float(sampled_precision.mean()))
        recalls.append(float(recall[-1]) if len(recall) else 0.0)
    return float(np.mean(average_precisions)), float(np.mean(recalls))


def detection_match_state(
    predicted_boxes,
    predicted_scores,
    target_boxes,
    *,
    iou_thresholds=IOU_THRESHOLDS,
    max_detections=40,
):
    """Store the per-image matches needed for dataset-level COCO AP/AR."""

    predicted_boxes = _array(predicted_boxes, columns=6)
    predicted_scores = _array(predicted_scores)
    target_boxes = _array(target_boxes, columns=6)
    if len(predicted_boxes) != len(predicted_scores):
        raise ValueError("Every predicted box must have exactly one score")
    order = np.argsort(-predicted_scores, kind="mergesort")[: int(max_detections)]
    predicted_boxes = predicted_boxes[order]
    predicted_scores = predicted_scores[order]
    overlaps = box_iou_3d(predicted_boxes, target_boxes)
    thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    matches = np.zeros((len(thresholds), len(predicted_boxes)), dtype=bool)
    for threshold_index, threshold in enumerate(thresholds):
        matched_targets = np.zeros(len(target_boxes), dtype=bool)
        for prediction_index in range(len(predicted_boxes)):
            candidates = np.where(
                np.logical_and(
                    ~matched_targets,
                    overlaps[prediction_index] >= threshold,
                )
            )[0]
            if candidates.size:
                best = candidates[np.argmax(overlaps[prediction_index, candidates])]
                matched_targets[best] = True
                matches[threshold_index, prediction_index] = True
    return {
        "scores": predicted_scores.astype(np.float32, copy=False),
        "matches": matches,
        "target_count": int(len(target_boxes)),
    }


def aggregate_detection_ap_ar(states, *, iou_thresholds=IOU_THRESHOLDS):
    """Aggregate per-image states as the baseline COCO metric does."""

    states = list(states)
    target_count = sum(int(state["target_count"]) for state in states)
    if target_count == 0:
        return float("nan"), float("nan")
    thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    score_parts = [np.asarray(state["scores"]).reshape(-1) for state in states]
    match_parts = [
        np.asarray(state["matches"], dtype=bool).reshape(len(thresholds), -1)
        for state in states
    ]
    scores = (
        np.concatenate(score_parts)
        if score_parts
        else np.empty((0,), dtype=np.float32)
    )
    matches = (
        np.concatenate(match_parts, axis=1)
        if match_parts
        else np.empty((len(thresholds), 0), dtype=bool)
    )
    order = np.argsort(-scores, kind="mergesort")
    matches = matches[:, order]
    average_precisions = []
    recalls = []
    for threshold_matches in matches:
        true_positive = np.cumsum(threshold_matches, dtype=np.float64)
        false_positive = np.cumsum(~threshold_matches, dtype=np.float64)
        recall = true_positive / float(target_count)
        precision = true_positive / np.maximum(
            true_positive + false_positive, np.finfo(np.float64).eps
        )
        for index in range(len(precision) - 1, 0, -1):
            precision[index - 1] = max(precision[index - 1], precision[index])
        sampled_precision = np.zeros(len(RECALL_THRESHOLDS), dtype=np.float64)
        indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        valid = indices < len(precision)
        sampled_precision[valid] = precision[indices[valid]]
        average_precisions.append(float(sampled_precision.mean()))
        recalls.append(float(recall[-1]) if len(recall) else 0.0)
    return float(np.mean(average_precisions)), float(np.mean(recalls))


def canonical_edges(edges, num_nodes):
    array = _array(edges, columns=2, dtype=np.int64)
    clean = set()
    for raw_left, raw_right in array:
        left, right = int(raw_left), int(raw_right)
        if left == right or min(left, right) < 0 or max(left, right) >= int(num_nodes):
            continue
        clean.add((left, right) if left < right else (right, left))
    return sorted(clean)


def graph_betti_numbers(num_nodes, edges):
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    parent = list(range(num_nodes))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in canonical_edges(edges, num_nodes):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    beta0 = len({find(node) for node in range(num_nodes)}) if num_nodes else 0
    edge_count = len(canonical_edges(edges, num_nodes))
    return beta0, edge_count - num_nodes + beta0


def graph_point_cloud(nodes, edges, *, num_points=100):
    """Sample a deterministic point cloud by arc length over actual edges."""

    nodes = torch.as_tensor(nodes, dtype=torch.float32).reshape(-1, 3).cpu()
    clean_edges = canonical_edges(edges, len(nodes))
    if not clean_edges:
        return None
    indices = torch.as_tensor(clean_edges, dtype=torch.long)
    starts = nodes[indices[:, 0]]
    ends = nodes[indices[:, 1]]
    lengths = torch.sqrt(torch.sum((ends - starts) ** 2, dim=1))
    positive = lengths > 0
    starts, ends, lengths = starts[positive], ends[positive], lengths[positive]
    if not len(lengths):
        return None
    cumulative = torch.cumsum(lengths, dim=0)
    positions = torch.linspace(0.0, float(cumulative[-1]), int(num_points))
    edge_ids = []
    for position in positions:
        candidates = torch.nonzero(cumulative >= position, as_tuple=False)
        edge_ids.append(
            candidates[0, 0]
            if candidates.numel()
            else cumulative.new_tensor(len(cumulative) - 1, dtype=torch.long)
        )
    edge_ids = torch.stack(edge_ids).long()
    previous = torch.cat((cumulative.new_zeros(1), cumulative[:-1]))
    fractions = (positions - previous[edge_ids]) / lengths[edge_ids]
    return starts[edge_ids] + fractions.unsqueeze(1) * (
        ends[edge_ids] - starts[edge_ids]
    )


def sinkhorn_distance(left, right, *, epsilon=1.0e-7, max_iterations=100):
    """Baseline-compatible entropic transport cost for two point clouds."""

    left = torch.as_tensor(left, dtype=torch.float32).cpu()
    right = torch.as_tensor(right, dtype=torch.float32).cpu()
    cost = torch.sum(torch.abs(left[:, None] - right[None, :]) ** 2, dim=-1)
    mu = torch.full((len(left),), 1.0 / len(left))
    nu = torch.full((len(right),), 1.0 / len(right))
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    for _ in range(int(max_iterations)):
        previous = u
        modified = (-cost + u[:, None] + v[None, :]) / float(epsilon)
        u = float(epsilon) * (
            torch.log(mu + 1.0e-8) - torch.logsumexp(modified, dim=-1)
        ) + u
        modified = (-cost + u[:, None] + v[None, :]) / float(epsilon)
        v = float(epsilon) * (
            torch.log(nu + 1.0e-8) - torch.logsumexp(modified.T, dim=-1)
        ) + v
        if float((u - previous).abs().sum()) < 1.0e-1:
            break
    plan = torch.exp((-cost + u[:, None] + v[None, :]) / float(epsilon))
    return float(torch.sum(plan * cost))


def street_mover_distance(
    target_nodes,
    target_edges,
    predicted_nodes,
    predicted_edges,
    *,
    num_points=100,
    epsilon=1.0e-7,
    max_iterations=100,
):
    target = graph_point_cloud(target_nodes, target_edges, num_points=num_points)
    predicted = graph_point_cloud(
        predicted_nodes, predicted_edges, num_points=num_points
    )
    if target is None or predicted is None:
        return float("nan")
    return sinkhorn_distance(
        target,
        predicted,
        epsilon=epsilon,
        max_iterations=max_iterations,
    )


def evaluate_graph(
    prediction: Mapping,
    target_nodes,
    target_edges,
    *,
    protocol=None,
    return_detection_states=False,
):
    protocol = dict(protocol or {})
    target_nodes = torch.as_tensor(target_nodes).detach().cpu().float().reshape(-1, 3)
    target_edges = torch.as_tensor(target_edges).detach().cpu().long().reshape(-1, 2)
    predicted_nodes = torch.as_tensor(prediction["nodes"]).detach().cpu().float()
    predicted_edges = torch.as_tensor(prediction["edges"]).detach().cpu().long()
    node_state, edge_state = graph_detection_states(
        prediction, target_nodes, target_edges, protocol=protocol
    )
    node_map, node_mar = aggregate_detection_ap_ar(
        [node_state], iou_thresholds=protocol.get("iou_thresholds", IOU_THRESHOLDS)
    )
    edge_map, edge_mar = aggregate_detection_ap_ar(
        [edge_state], iou_thresholds=protocol.get("iou_thresholds", IOU_THRESHOLDS)
    )
    target_beta0, target_beta1 = graph_betti_numbers(len(target_nodes), target_edges)
    predicted_beta0, predicted_beta1 = graph_betti_numbers(
        len(predicted_nodes), predicted_edges
    )
    metrics = {
        "smd": street_mover_distance(
            target_nodes,
            target_edges,
            predicted_nodes,
            predicted_edges,
            num_points=int(protocol.get("smd_points", 100)),
            epsilon=float(protocol.get("smd_epsilon", 1.0e-7)),
            max_iterations=int(protocol.get("smd_iterations", 100)),
        ),
        "node_mAP": node_map,
        "node_mAR": node_mar,
        "edge_mAP": edge_map,
        "edge_mAR": edge_mar,
        "beta0_absolute_error": float(abs(predicted_beta0 - target_beta0)),
        "beta1_absolute_error": float(abs(predicted_beta1 - target_beta1)),
        "target_beta0": float(target_beta0),
        "predicted_beta0": float(predicted_beta0),
        "target_beta1": float(target_beta1),
        "predicted_beta1": float(predicted_beta1),
        "target_nodes": float(len(target_nodes)),
        "predicted_nodes": float(len(predicted_nodes)),
        "node_count_absolute_error": float(
            abs(len(predicted_nodes) - len(target_nodes))
        ),
        "target_edges": float(len(canonical_edges(target_edges, len(target_nodes)))),
        "predicted_edges": float(
            len(canonical_edges(predicted_edges, len(predicted_nodes)))
        ),
    }
    metrics["edge_count_absolute_error"] = abs(
        metrics["predicted_edges"] - metrics["target_edges"]
    )
    if return_detection_states:
        return metrics, node_state, edge_state
    return metrics


def graph_detection_states(prediction: Mapping, target_nodes, target_edges, *, protocol=None):
    """Return node/edge matching states for fold-level AP/AR aggregation."""

    protocol = dict(protocol or {})
    thresholds = protocol.get("iou_thresholds", IOU_THRESHOLDS)
    max_detections = int(protocol.get("max_detections", 40))
    target_node_size = float(protocol.get("target_node_size", 0.2))
    edge_half_width = float(protocol.get("edge_half_width", 0.1))
    target_nodes = torch.as_tensor(target_nodes).detach().cpu().float().reshape(-1, 3)
    target_edges = torch.as_tensor(target_edges).detach().cpu().long().reshape(-1, 2)
    predicted_nodes = torch.as_tensor(prediction["nodes"]).detach().cpu().float()
    predicted_edges = torch.as_tensor(prediction["edges"]).detach().cpu().long()
    predicted_boxes = torch.as_tensor(prediction["boxes"]).detach().cpu().float()
    target_boxes = torch.cat(
        (target_nodes, torch.full_like(target_nodes, target_node_size)), dim=1
    )
    node_state = detection_match_state(
        center_size_to_corners(predicted_boxes.numpy()),
        torch.as_tensor(prediction["node_scores"]).detach().cpu().numpy(),
        center_size_to_corners(target_boxes.numpy()),
        iou_thresholds=thresholds,
        max_detections=max_detections,
    )
    edge_state = detection_match_state(
        edge_boxes(predicted_nodes.numpy(), predicted_edges.numpy(), half_width=edge_half_width),
        torch.as_tensor(prediction["edge_scores"]).detach().cpu().numpy(),
        edge_boxes(target_nodes.numpy(), target_edges.numpy(), half_width=edge_half_width),
        iou_thresholds=thresholds,
        max_detections=max_detections,
    )
    return node_state, edge_state


def summarize_metrics(
    rows: Sequence[Mapping[str, float]],
    *,
    folds=5,
    node_detection_states=None,
    edge_detection_states=None,
    iou_thresholds=IOU_THRESHOLDS,
):
    """Return global means and contiguous-fold standard deviations."""

    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    if iou_thresholds is None:
        iou_thresholds = IOU_THRESHOLDS
    names = sorted(set.intersection(*(set(row) for row in rows)))
    summary = {"samples": int(len(rows))}
    split_indices = np.array_split(np.arange(len(rows)), min(int(folds), len(rows)))
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = float(finite.mean()) if finite.size else float("nan")
        fold_means = []
        for indices in split_indices:
            fold = values[indices]
            fold = fold[np.isfinite(fold)]
            if fold.size:
                fold_means.append(float(fold.mean()))
        summary[name + "_std"] = (
            float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0
        )
    detection_groups = (
        ("node", node_detection_states),
        ("edge", edge_detection_states),
    )
    for prefix, states in detection_groups:
        if states is None:
            continue
        states = list(states)
        if len(states) != len(rows):
            raise ValueError("Detection states must align one-to-one with metric rows")
        fold_ap = []
        fold_ar = []
        for indices in split_indices:
            ap, ar = aggregate_detection_ap_ar(
                (states[int(index)] for index in indices),
                iou_thresholds=iou_thresholds,
            )
            if np.isfinite(ap):
                fold_ap.append(ap)
            if np.isfinite(ar):
                fold_ar.append(ar)
        summary[prefix + "_mAP"] = (
            float(np.mean(fold_ap)) if fold_ap else float("nan")
        )
        summary[prefix + "_mAP_std"] = (
            float(np.std(fold_ap, ddof=1)) if len(fold_ap) > 1 else 0.0
        )
        summary[prefix + "_mAR"] = (
            float(np.mean(fold_ar)) if fold_ar else float("nan")
        )
        summary[prefix + "_mAR_std"] = (
            float(np.std(fold_ar, ddof=1)) if len(fold_ar) > 1 else 0.0
        )
    return summary


__all__ = [
    "IOU_THRESHOLDS",
    "aggregate_detection_ap_ar",
    "box_iou_3d",
    "canonical_edges",
    "center_size_to_corners",
    "detection_ap_ar",
    "detection_match_state",
    "edge_boxes",
    "evaluate_graph",
    "graph_betti_numbers",
    "graph_detection_states",
    "graph_point_cloud",
    "sinkhorn_distance",
    "street_mover_distance",
    "summarize_metrics",
]
