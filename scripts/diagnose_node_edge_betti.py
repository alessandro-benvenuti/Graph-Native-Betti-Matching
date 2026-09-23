#!/usr/bin/env python3
"""Audit matched-only and node-aware Betti losses on fixed graph patches."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configs import load_config, validate_config
from data.loaders import build_evaluation_loader
from models import build_model
from models.checkpoint import load_legacy_model_checkpoint
from models.matcher import build_matcher
from training.losses.betti_filtration import node_edge_confidences
from training.losses.betti_h0 import h0_betti_matching_loss
from training.losses.betti_h1 import (
    compute_cycle_space_matching,
    cycle_space_matching_loss,
)
from training.losses.criterion import GraphCriterion


DEFAULT_BOUNDARY_AUDIT_SAMPLES = (
    "sample_000038_0518",
    "sample_000056_0493",
    "sample_000060_0216",
    "sample_000131_0216",
    "sample_000036_0419",
    "sample_000125_0880",
    "sample_000025_0167",
    "sample_000125_0657",
    "sample_000121_0167",
    "sample_000051_0877",
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id", action="append", dest="sample_ids")
    selection.add_argument("--sample-list", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--aggregation", choices=("min", "product", "hybrid"), default="hybrid"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--unmatched-object-threshold", type=float, default=0.25)
    parser.add_argument("--max-active-unmatched", type=int, default=8)
    parser.add_argument(
        "--normalization",
        choices=("feature_count", "matched_mean"),
        default="matched_mean",
    )
    parser.add_argument(
        "--detach-unmatched-edge-probabilities",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("matched_only", "node_aware"),
        dest="modes",
        help="Topology complex to diagnose; repeat for both (default: both).",
    )
    return parser


def _selected_samples(args):
    if args.sample_ids is not None:
        result = args.sample_ids
    elif args.sample_list is not None:
        if not args.sample_list.is_file():
            raise FileNotFoundError(args.sample_list)
        result = [
            line.strip()
            for line in args.sample_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        result = list(DEFAULT_BOUNDARY_AUDIT_SAMPLES)
    if not result or len(result) != len(set(result)):
        raise ValueError("selected sample IDs must be non-empty and unique")
    return result


def _dataset_name(config, requested):
    if requested is not None:
        return requested
    targets = [
        name
        for name, settings in config["data"]["datasets"].items()
        if settings["role"] == "target"
    ]
    if len(targets) != 1:
        raise ValueError("--dataset is required unless exactly one target is configured")
    return targets[0]


def _direction(value, tolerance=1.0e-12):
    if value > tolerance:
        return "decrease"
    if value < -tolerance:
        return "increase"
    return "none"


def _gradients(loss, node_probabilities, edge_probabilities, *, retain_graph):
    inputs = [edge_probabilities]
    has_node_gradient = bool(node_probabilities.requires_grad)
    if has_node_gradient:
        inputs.append(node_probabilities)
    gradients = torch.autograd.grad(
        loss,
        inputs,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    edge_gradient = gradients[0]
    if edge_gradient is None:
        edge_gradient = torch.zeros_like(edge_probabilities)
    if has_node_gradient:
        node_gradient = gradients[1]
        if node_gradient is None:
            node_gradient = torch.zeros_like(node_probabilities)
    else:
        node_gradient = torch.zeros_like(node_probabilities)
    return node_gradient.detach(), edge_gradient.detach()


def _h0_pair(pair, selected_queries):
    edge = tuple(int(value) for value in pair.death_edge)
    return {
        "birth_local_node": int(pair.birth_vertex),
        "birth_query_id": int(selected_queries[pair.birth_vertex]),
        "birth": float(pair.birth),
        "death_local_edge": list(edge),
        "death_query_edge": [int(selected_queries[index]) for index in edge],
        "death": float(pair.death),
        "persistence": float(pair.persistence),
    }


def _h1_class(item, selected_queries):
    birth_edge = tuple(int(value) for value in item.birth_edge)
    cycle = [tuple(int(value) for value in edge) for edge in item.cycle_edges]
    return {
        "birth_local_edge": list(birth_edge),
        "birth_query_edge": [int(selected_queries[index]) for index in birth_edge],
        "birth": float(item.birth),
        "persistence_to_cap": float(1.0 - item.birth),
        "cycle_local_edges": [list(edge) for edge in cycle],
        "cycle_query_edges": [
            [int(selected_queries[index]) for index in edge] for edge in cycle
        ],
    }


def _matching_summary(h0_matching, h1_matching, selected_queries):
    h0_false = set(h0_matching.unmatched_prediction_indices)
    h1_false = set(h1_matching.unmatched_prediction_indices)
    return {
        "h0": {
            "matched_rank": h0_matching.matched_rank,
            "false_prediction_rank": h0_matching.false_prediction_rank,
            "missed_target_rank": h0_matching.missed_target_rank,
            "prediction_pairs": [
                {**_h0_pair(pair, selected_queries), "false_prediction": index in h0_false}
                for index, pair in enumerate(h0_matching.prediction_pairs)
            ],
        },
        "h1": {
            "prediction_rank": len(h1_matching.prediction_classes),
            "target_rank": len(h1_matching.target_classes),
            "matched_rank": h1_matching.shared_rank,
            "false_prediction_rank": h1_matching.false_prediction_rank,
            "missed_target_rank": h1_matching.missed_target_rank,
            "union_only_rank": h1_matching.union_only_rank,
            "prediction_classes": [
                {**_h1_class(item, selected_queries), "false_prediction": index in h1_false}
                for index, item in enumerate(h1_matching.prediction_classes)
            ],
        },
    }


def _original_target_cycle_diagnostics(
    target_edges,
    *,
    target_node_count,
    matched_target_nodes,
):
    """Describe which deterministic GT basis cycles lose a matched vertex."""

    edge_tuples = tuple(
        tuple(sorted((int(left), int(right))))
        for left, right in torch.as_tensor(target_edges).reshape(-1, 2).tolist()
    )
    matching = compute_cycle_space_matching(
        [1.0] * len(edge_tuples),
        edge_tuples,
        edge_tuples,
        num_vertices=int(target_node_count),
    )
    matched = {int(index) for index in matched_target_nodes}
    cycles = []
    for index, item in enumerate(matching.target_classes):
        vertices = sorted({vertex for edge in item.cycle_edges for vertex in edge})
        missing = [vertex for vertex in vertices if vertex not in matched]
        cycles.append(
            {
                "basis_index": index,
                "target_edges": [list(edge) for edge in item.cycle_edges],
                "target_vertices": vertices,
                "missing_matched_target_nodes": missing,
                "represented_in_local_target": not missing,
            }
        )
    return cycles


def _h1_match_details(
    matching,
    *,
    selected_queries,
    matched_targets,
    edge_records,
    true_edges,
):
    """Attach the selected differentiable birth edge to every H1 match."""

    records_by_edge = {
        tuple(record["local_edge"]): record for record in edge_records
    }
    truth = {tuple(sorted(map(int, edge))) for edge in true_edges}
    details = []
    for index, match in enumerate(matching.matches):
        prediction = matching.prediction_classes[match.prediction_index]
        target = matching.target_classes[match.target_index]
        birth_edge = tuple(prediction.birth_edge)
        birth_record = records_by_edge[birth_edge]
        shared_edges = tuple(tuple(edge) for edge in match.shared_cycle_edges)
        shared_records = [records_by_edge[edge] for edge in shared_edges]
        bottleneck = max(
            shared_records,
            key=lambda record: (
                record["filtration"],
                tuple(record["local_edge"]),
            ),
        )
        target_cycle_gt_edges = []
        for left, right in target.cycle_edges:
            if left >= len(matched_targets) or right >= len(matched_targets):
                raise RuntimeError("A target H1 class used an unmatched local vertex")
            target_cycle_gt_edges.append(
                sorted((int(matched_targets[left]), int(matched_targets[right])))
            )
        details.append(
            {
                "match_index": index,
                "prediction_class_index": int(match.prediction_index),
                "target_class_index": int(match.target_index),
                "selected_birth_local_edge": list(birth_edge),
                "selected_birth_query_edge": [
                    int(selected_queries[vertex]) for vertex in birth_edge
                ],
                "selected_birth_raw_relation_probability": birth_record[
                    "raw_relation_probability"
                ],
                "selected_birth_effective_confidence": birth_record[
                    "effective_confidence"
                ],
                "selected_birth_filtration": birth_record["filtration"],
                "selected_birth_dloss_dp": birth_record["h1_dloss_dp"],
                "selected_birth_gradient_descent": birth_record[
                    "h1_gradient_descent"
                ],
                "selected_birth_is_local_true_edge": birth_edge in truth,
                "selected_birth_is_in_shared_generator": birth_edge in shared_edges,
                "shared_generator_local_edges": [list(edge) for edge in shared_edges],
                "shared_generator_bottleneck_local_edge": bottleneck["local_edge"],
                "shared_generator_bottleneck_effective_confidence": bottleneck[
                    "effective_confidence"
                ],
                "selected_birth_is_shared_generator_bottleneck": (
                    birth_edge == tuple(bottleneck["local_edge"])
                ),
                "target_cycle_gt_edges": target_cycle_gt_edges,
            }
        )
    return details


def _evaluate_mode(
    criterion,
    tokens,
    node_logits,
    predicted_nodes,
    target_edges,
    source,
    target,
    *,
    mode,
    aggregation,
    alpha,
    target_node_count=None,
):
    started = time.perf_counter()
    criterion.topology["complex"]["mode"] = mode
    local_logits = node_logits.detach().clone().requires_grad_(True)
    selected, node_probabilities, target_presence = criterion._topology_queries(
        local_logits, source
    )
    count = int(selected.numel())
    if count < 2:
        return {
            "mode": mode,
            "selected_query_ids": selected.detach().cpu().tolist(),
            "skipped": "fewer than two selected vertices",
            "elapsed_seconds": time.perf_counter() - started,
        }

    device = tokens.device
    pairs = torch.combinations(torch.arange(count, device=device), r=2)
    object_tokens = tokens[: criterion.object_queries]
    relation_tokens = tokens[
        criterion.object_queries : criterion.object_queries + criterion.relation_tokens
    ]
    raw = criterion._symmetric_edge_probabilities(
        object_tokens[selected], relation_tokens, pairs
    ).detach().requires_grad_(True)
    effective = raw
    if mode == "node_aware":
        filtration_raw = raw
        if criterion.topology["complex"][
            "detach_unmatched_edge_probabilities"
        ]:
            incident_to_absent = (
                target_presence[pairs[:, 0]] < 0.5
            ) | (target_presence[pairs[:, 1]] < 0.5)
            filtration_raw = torch.where(
                incident_to_absent, raw.detach(), raw
            )
        effective = node_edge_confidences(
            node_probabilities,
            filtration_raw,
            pairs,
            aggregation=aggregation,
            alpha=alpha,
        )
    truth = criterion._local_true_edges(target_edges, target, device)

    h0_config = criterion.topology["betti_h0"]
    normalization = (
        str(h0_config["normalization"])
        if mode == "node_aware"
        else "feature_count"
    )
    h0_keywords = dict(
        num_vertices=count,
        unmatched_weight=float(h0_config["unmatched_weight"]),
        diagonal_factor=float(h0_config["diagonal_factor"]),
        normalize=bool(h0_config["normalize"]),
        normalization=normalization,
    )
    if mode == "node_aware":
        h0_keywords.update(
            node_probabilities=node_probabilities,
            target_node_presence=target_presence,
        )
    h0_loss, h0_matching = h0_betti_matching_loss(
        effective, pairs, truth, **h0_keywords
    )

    h1_config = criterion.topology["betti_h1"]
    h1_loss, h1_matching = cycle_space_matching_loss(
        effective,
        pairs,
        truth,
        num_vertices=count,
        false_positive_weight=float(h1_config["false_positive_weight"]),
        false_negative_weight=float(h1_config["false_negative_weight"]),
        diagonal_factor=float(h1_config["diagonal_factor"]),
        normalize=bool(h1_config["normalize"]),
        normalization=(
            str(h1_config["normalization"])
            if mode == "node_aware"
            else "feature_count"
        ),
    )
    h0_node_gradient, h0_edge_gradient = _gradients(
        h0_loss, node_probabilities, raw, retain_graph=True
    )
    h1_node_gradient, h1_edge_gradient = _gradients(
        h1_loss, node_probabilities, raw, retain_graph=False
    )

    source_cpu = source.detach().cpu().tolist()
    target_cpu = target.detach().cpu().tolist()
    target_by_query = dict(zip(source_cpu, target_cpu))
    selected_cpu = selected.detach().cpu().tolist()
    node_probability_cpu = node_probabilities.detach().cpu().tolist()
    coordinates = predicted_nodes[selected].detach().cpu().tolist()
    nodes = []
    for local, query in enumerate(selected_cpu):
        matched = query in target_by_query
        nodes.append(
            {
                "local_node": local,
                "query_id": query,
                "matched": matched,
                "target_node": target_by_query.get(query),
                "probability": float(node_probability_cpu[local]),
                "coordinates_dhw": coordinates[local][:3],
                "topology_can_update_node": bool(mode == "node_aware" and not matched),
                "h0_dloss_dq": float(h0_node_gradient[local]),
                "h0_gradient_descent": _direction(float(h0_node_gradient[local])),
                "h1_dloss_dq": float(h1_node_gradient[local]),
                "h1_gradient_descent": _direction(float(h1_node_gradient[local])),
            }
        )

    pair_cpu = pairs.detach().cpu().tolist()
    raw_cpu = raw.detach().cpu().tolist()
    effective_cpu = effective.detach().cpu().tolist()
    edges = []
    for index, ((left, right), probability, score) in enumerate(
        zip(pair_cpu, raw_cpu, effective_cpu)
    ):
        edges.append(
            {
                "local_edge": [left, right],
                "query_edge": [selected_cpu[left], selected_cpu[right]],
                "raw_relation_probability": float(probability),
                "effective_confidence": float(score),
                "filtration": float(1.0 - score),
                "h0_dloss_dp": float(h0_edge_gradient[index]),
                "h0_gradient_descent": _direction(float(h0_edge_gradient[index])),
                "h1_dloss_dp": float(h1_edge_gradient[index]),
                "h1_gradient_descent": _direction(float(h1_edge_gradient[index])),
            }
        )

    if target_node_count is None:
        target_node_count = max(target_cpu, default=-1) + 1
    target_cycle_diagnostics = _original_target_cycle_diagnostics(
        target_edges,
        target_node_count=target_node_count,
        matched_target_nodes=target_cpu,
    )
    h1_matches = _h1_match_details(
        h1_matching,
        selected_queries=selected_cpu,
        matched_targets=target_cpu,
        edge_records=edges,
        true_edges=truth.detach().cpu().tolist(),
    )

    return {
        "mode": mode,
        "aggregation": aggregation if mode == "node_aware" else None,
        "alpha": alpha if mode == "node_aware" else None,
        "normalization": normalization,
        "selected_query_ids": selected_cpu,
        "matched_vertex_count": len(source_cpu),
        "target_vertex_count": int(target_node_count),
        "missing_target_node_count": int(target_node_count) - len(target_cpu),
        "missing_target_node_indices": sorted(
            set(range(int(target_node_count))) - set(target_cpu)
        ),
        "active_unmatched_count": len(selected_cpu) - len(source_cpu),
        "elapsed_seconds": time.perf_counter() - started,
        "h0_loss": float(h0_loss.detach()),
        "h1_loss": float(h1_loss.detach()),
        "nodes": nodes,
        "edges": edges,
        **_matching_summary(h0_matching, h1_matching, selected_cpu),
        "original_target_cycle_rank": len(target_cycle_diagnostics),
        "original_target_cycles_blocked_by_missing_node": sum(
            not item["represented_in_local_target"]
            for item in target_cycle_diagnostics
        ),
        "original_target_cycle_diagnostics": target_cycle_diagnostics,
        "h1_matches": h1_matches,
    }


def _summary(records, modes):
    result = {"samples": len(records), "modes": {}}
    for mode in modes:
        entries = [record["modes"][mode] for record in records]
        valid = [entry for entry in entries if "skipped" not in entry]
        divisor = max(1, len(valid))
        result["modes"][mode] = {
            "evaluated_samples": len(valid),
            "mean_h0_loss": sum(entry["h0_loss"] for entry in valid) / divisor,
            "mean_h1_loss": sum(entry["h1_loss"] for entry in valid) / divisor,
            "total_false_h0_rank": sum(
                entry["h0"]["false_prediction_rank"] for entry in valid
            ),
            "total_false_h1_rank": sum(
                entry["h1"]["false_prediction_rank"] for entry in valid
            ),
            "total_active_unmatched": sum(
                entry["active_unmatched_count"] for entry in valid
            ),
            "total_original_target_cycle_rank": sum(
                entry["original_target_cycle_rank"] for entry in valid
            ),
            "total_original_cycles_blocked_by_missing_node": sum(
                entry["original_target_cycles_blocked_by_missing_node"]
                for entry in valid
            ),
            "total_local_target_cycle_rank": sum(
                entry["h1"]["target_rank"]
                for entry in valid
            ),
            "total_matched_h1_rank": sum(
                entry["h1"]["matched_rank"] for entry in valid
            ),
            "matched_birth_gradient_increase": sum(
                match["selected_birth_gradient_descent"] == "increase"
                for entry in valid
                for match in entry["h1_matches"]
            ),
            "matched_birth_gradient_none": sum(
                match["selected_birth_gradient_descent"] == "none"
                for entry in valid
                for match in entry["h1_matches"]
            ),
            "matched_birth_gradient_decrease": sum(
                match["selected_birth_gradient_descent"] == "decrease"
                for entry in valid
                for match in entry["h1_matches"]
            ),
            "selected_birth_is_shared_bottleneck": sum(
                match["selected_birth_is_shared_generator_bottleneck"]
                for entry in valid
                for match in entry["h1_matches"]
            ),
        }
    return result


def _write_csv(path, records):
    fields = (
        "source_sample_id",
        "mode",
        "h0_loss",
        "h1_loss",
        "active_unmatched_count",
        "false_h0_rank",
        "false_h1_rank",
        "missed_h0_rank",
        "missed_h1_rank",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for mode, entry in record["modes"].items():
                if "skipped" in entry:
                    continue
                writer.writerow(
                    {
                        "source_sample_id": record["source_sample_id"],
                        "mode": mode,
                        "h0_loss": entry["h0_loss"],
                        "h1_loss": entry["h1_loss"],
                        "active_unmatched_count": entry["active_unmatched_count"],
                        "false_h0_rank": entry["h0"]["false_prediction_rank"],
                        "false_h1_rank": entry["h1"]["false_prediction_rank"],
                        "missed_h0_rank": entry["h0"]["missed_target_rank"],
                        "missed_h1_rank": entry["h1"]["missed_target_rank"],
                    }
                )


def main():
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.max_active_unmatched < 0:
        raise ValueError("batch size must be positive; workers/cap must be non-negative")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must lie in [0,1]")
    if not 0.0 <= args.unmatched_object_threshold <= 1.0:
        raise ValueError("--unmatched-object-threshold must lie in [0,1]")
    requested_modes = tuple(
        dict.fromkeys(args.modes or ("matched_only", "node_aware"))
    )
    output = Path(args.output_dir)
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise FileExistsError(f"output directory is not empty: {output}")

    sample_ids = _selected_samples(args)
    config = copy.deepcopy(load_config(args.config))
    config["runtime"]["device"] = args.device
    config["runtime"]["workers"] = args.workers
    config["data"]["batch_size"] = args.batch_size
    config["data"]["validation_batch_size"] = args.batch_size
    config["topology"]["complex"].update(
        mode="node_aware",
        aggregation=args.aggregation,
        alpha=args.alpha,
        unmatched_object_threshold=args.unmatched_object_threshold,
        max_active_unmatched=args.max_active_unmatched,
        detach_unmatched_edge_probabilities=(
            args.detach_unmatched_edge_probabilities
        ),
    )
    for name in ("betti_h0", "betti_h1"):
        config["topology"][name]["normalization"] = args.normalization
    validate_config(config)
    dataset_name = _dataset_name(config, args.dataset)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    loader = build_evaluation_loader(
        config,
        dataset_name=dataset_name,
        split=args.split,
        max_samples=None,
        sample_ids=sample_ids,
    )
    model = build_model(config).to(device).eval()
    checkpoint_report = load_legacy_model_checkpoint(
        model, args.checkpoint, map_location="cpu"
    )
    matcher = build_matcher(config)
    criterion = GraphCriterion(config, matcher, model.relation_embed)
    records = []
    dataset_records = getattr(loader.dataset, "records", ())
    sample_index = 0

    for batch in loader:
        volumes = batch[0] if config["training"]["input"] == "image" else batch[1]
        volumes = volumes.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = {
            "nodes": [nodes.to(device) for nodes in batch[2]],
            "edges": [edges.to(device) for edges in batch[3]],
        }
        with torch.no_grad():
            tokens, predictions, _ = model(volumes)
            assignments = matcher(predictions, targets)

        for local_index, (source, target) in enumerate(assignments):
            source_sample_id = str(dataset_records[sample_index].sample_id)
            mode_results = {}
            for mode in requested_modes:
                mode_results[mode] = _evaluate_mode(
                    criterion,
                    tokens[local_index],
                    predictions["pred_logits"][local_index],
                    predictions["pred_nodes"][local_index],
                    targets["edges"][local_index],
                    source.to(device),
                    target.to(device),
                    mode=mode,
                    aggregation=args.aggregation,
                    alpha=args.alpha,
                    target_node_count=int(targets["nodes"][local_index].shape[0]),
                )
            records.append(
                {
                    "source_sample_id": source_sample_id,
                    "target_node_count": int(targets["nodes"][local_index].shape[0]),
                    "target_edge_count": int(targets["edges"][local_index].shape[0]),
                    "modes": mode_results,
                }
            )
            sample_index += 1

    output.mkdir(parents=True, exist_ok=True)
    summary = _summary(records, requested_modes)
    (output / "per-sample.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "per-sample.csv", records)
    with (output / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ignored_removed_parameters": list(checkpoint_report.ignored_removed),
        "dataset": dataset_name,
        "split": args.split,
        "requested_sample_ids": sample_ids,
        "aggregation": args.aggregation,
        "alpha": args.alpha,
        "unmatched_object_threshold": args.unmatched_object_threshold,
        "max_active_unmatched": args.max_active_unmatched,
        "normalization": args.normalization,
        "detach_unmatched_edge_probabilities": (
            args.detach_unmatched_edge_probabilities
        ),
        "modes": list(requested_modes),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Detailed output: {output}")


if __name__ == "__main__":
    main()
