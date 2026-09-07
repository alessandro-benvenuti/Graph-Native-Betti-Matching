#!/usr/bin/env python3
"""Compare Hungarian and hardened-FGW assignments on a fixed checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
from models.matcher import (
    FusedGromovWassersteinMatcher,
    HungarianMatcher,
    score_candidate_structures,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.1, 0.2, 0.4))
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--pair-chunk-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


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


def _validate_args(args):
    for name in (
        "max_samples",
        "batch_size",
        "candidate_count",
        "pair_chunk_size",
        "max_iter",
        "progress_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if float(args.tolerance) <= 0:
        raise ValueError("--tolerance must be positive")
    if not args.alphas or len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas must contain distinct values")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("every --alphas value must lie in [0,1]")


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _assignment_vector(assignment, target_count):
    source, target = assignment
    if len(source) != target_count or len(target) != target_count:
        raise RuntimeError("hard assignment does not cover every target")
    if source.unique().numel() != target_count or target.unique().numel() != target_count:
        raise RuntimeError("hard assignment is not one-to-one")
    result = torch.full((target_count,), -1, dtype=torch.long)
    result[target.long().cpu()] = source.long().cpu()
    if bool((result < 0).any()):
        raise RuntimeError("hard assignment omitted a target")
    return result


def _safe_mean(values):
    return float(values.mean().detach().cpu()) if values.numel() else float("nan")


def assignment_metrics(
    predicted_nodes,
    predicted_logits,
    target_nodes,
    target_edges,
    candidates,
    candidate_structure,
    assignment,
    *,
    dimensions,
    reference_assignment=None,
):
    """Measure geometry and induced adjacency for one hard assignment."""

    target_count = int(target_nodes.shape[0])
    assignment_vector = _assignment_vector(assignment, target_count)
    device = predicted_nodes.device
    source = assignment_vector.to(device)
    truth = target_nodes.to(device)
    coordinate_l1 = torch.abs(
        predicted_nodes[source, :dimensions] - truth[:, :dimensions]
    ).sum(dim=1)
    object_probability = predicted_logits.softmax(-1)[:, 1]

    candidate_list = candidates.long().cpu().tolist()
    local_by_global = {global_index: local for local, global_index in enumerate(candidate_list)}
    try:
        local_source = torch.as_tensor(
            [local_by_global[index] for index in assignment_vector.tolist()],
            dtype=torch.long,
            device=candidate_structure.device,
        )
    except KeyError as error:
        raise RuntimeError("hard assignment contains a query outside the candidate pool") from error
    aligned_structure = candidate_structure[local_source][:, local_source]
    truth_structure = FusedGromovWassersteinMatcher.target_structure(
        target_count,
        target_edges,
        dtype=aligned_structure.dtype,
        device=aligned_structure.device,
    )
    upper = torch.triu(
        torch.ones_like(truth_structure, dtype=torch.bool), diagonal=1
    )
    edge_mask = upper & (truth_structure > 0.5)
    nonedge_mask = upper & ~edge_mask
    edge_mean = _safe_mean(aligned_structure[edge_mask])
    nonedge_mean = _safe_mean(aligned_structure[nonedge_mask])
    structural_mse = _safe_mean(
        (aligned_structure[upper] - truth_structure[upper]).square()
    )
    result = {
        "matched_query_ids": assignment_vector.tolist(),
        "target_count": target_count,
        "candidate_count": int(candidates.numel()),
        "coordinate_l1_mean": _safe_mean(coordinate_l1),
        "matched_object_probability_mean": _safe_mean(object_probability[source]),
        "structural_mse": structural_mse,
        "gt_edge_probability_mean": edge_mean,
        "gt_nonedge_probability_mean": nonedge_mean,
        "edge_nonedge_separation": edge_mean - nonedge_mean,
        "hard_unique": True,
    }
    if reference_assignment is None:
        result["changed_target_fraction"] = 0.0
        result["changed_any"] = 0.0
    else:
        reference = _assignment_vector(reference_assignment, target_count)
        changed = assignment_vector != reference
        result["changed_target_fraction"] = float(changed.float().mean())
        result["changed_any"] = float(bool(changed.any()))
    return result


def transport_metrics(transport):
    """Return concentration diagnostics for a semi-relaxed plan."""

    plan = np.asarray(transport, dtype=np.float64)
    if plan.size == 0:
        return {"soft_argmax_collisions": 0, "normalized_row_entropy": 0.0}
    conditional = np.clip(plan, 0.0, None)
    conditional = conditional / np.maximum(
        conditional.sum(axis=1, keepdims=True), 1e-15
    )
    positive = np.clip(conditional, 1e-15, None)
    entropy = -np.sum(conditional * np.log(positive), axis=1)
    if plan.shape[1] > 1:
        entropy = entropy / math.log(plan.shape[1])
    else:
        entropy = np.zeros_like(entropy)
    collisions = plan.shape[0] - len(np.unique(np.argmax(plan, axis=1)))
    return {
        "soft_argmax_collisions": int(collisions),
        "normalized_row_entropy": float(np.mean(entropy)),
    }


def summarize_rows(rows):
    """Aggregate finite numeric diagnostics independently for every method."""

    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        metric_names = sorted(set().union(*(row.keys() for row in selected)))
        metrics = {}
        for name in metric_names:
            if name in {"method", "sample_id", "source_sample_id"}:
                continue
            values = [
                float(row[name])
                for row in selected
                if isinstance(row.get(name), (int, float))
                and math.isfinite(float(row[name]))
            ]
            if values:
                metrics[name] = float(np.mean(values))
        summaries[method] = {"samples": len(selected), **metrics}
    return summaries


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    fields = ["sample_id", "source_sample_id", "method"] + sorted(
        set().union(*(row.keys() for row in rows))
        - {"sample_id", "source_sample_id", "method"}
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )


@torch.no_grad()
def run_diagnostic(model, loader, config, device, args):
    model.eval()
    decoder = config["model"]["decoder"]
    matcher_config = config["model"]["matcher"]
    dimensions = int(config["data"]["spatial_dims"])
    common = dict(
        class_cost=float(matcher_config["class_cost"]),
        node_cost=float(matcher_config["node_cost"]),
        dimensions=dimensions,
    )
    hungarian = HungarianMatcher(**common)
    fgw_matchers = {
        alpha: FusedGromovWassersteinMatcher(
            **common,
            structure_weight=alpha,
            candidate_count=args.candidate_count,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            random_state=int(config["experiment"]["seed"]),
        )
        for alpha in args.alphas
    }
    candidate_matcher = next(iter(fgw_matchers.values()))
    relation_embed = model.relation_embed
    dataset_records = getattr(loader.dataset, "records", ())
    rows = []
    sample_index = 0
    total_samples = len(loader.dataset)

    for batch in loader:
        volumes = batch[0] if config["training"]["input"] == "image" else batch[1]
        volumes = volumes.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = {
            "nodes": [nodes.to(device) for nodes in batch[2]],
            "edges": [edges.to(device) for edges in batch[3]],
        }

        _synchronize(device)
        started = time.perf_counter()
        tokens, predictions, _ = model(volumes)
        _synchronize(device)
        forward_seconds = time.perf_counter() - started

        started = time.perf_counter()
        candidates = candidate_matcher.matching_candidates(predictions, targets)
        structures = score_candidate_structures(
            tokens,
            relation_embed,
            candidates,
            object_queries=int(decoder["object_queries"]),
            relation_tokens=int(decoder["relation_tokens"]),
            pair_chunk_size=args.pair_chunk_size,
        )
        _synchronize(device)
        structure_seconds = time.perf_counter() - started

        started = time.perf_counter()
        reference_assignments = hungarian(predictions, targets)
        hungarian_seconds = time.perf_counter() - started
        batch_count = len(reference_assignments)

        method_results = {}
        for alpha, matcher in fgw_matchers.items():
            started = time.perf_counter()
            assignments, transports = matcher(
                predictions,
                targets,
                predicted_structure=structures,
                candidate_indices=candidates,
                return_transport=True,
            )
            method_results[alpha] = (
                assignments,
                transports,
                time.perf_counter() - started,
            )

        for local_index in range(batch_count):
            sample_id = f"sample_{sample_index:06d}"
            source_sample_id = (
                str(dataset_records[sample_index].sample_id)
                if sample_index < len(dataset_records)
                else None
            )
            common_metrics = dict(
                predicted_nodes=predictions["pred_nodes"][local_index],
                predicted_logits=predictions["pred_logits"][local_index],
                target_nodes=targets["nodes"][local_index],
                target_edges=targets["edges"][local_index],
                candidates=candidates[local_index],
                candidate_structure=structures[local_index],
                dimensions=dimensions,
            )
            reference = reference_assignments[local_index]
            baseline = assignment_metrics(
                **common_metrics,
                assignment=reference,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": source_sample_id,
                    "method": "hungarian",
                    **baseline,
                    "model_forward_seconds": forward_seconds / batch_count,
                    "structure_scoring_seconds": 0.0,
                    "matcher_seconds": hungarian_seconds / batch_count,
                }
            )
            for alpha, (assignments, transports, elapsed) in method_results.items():
                metrics = assignment_metrics(
                    **common_metrics,
                    assignment=assignments[local_index],
                    reference_assignment=reference,
                )
                for name in (
                    "coordinate_l1_mean",
                    "structural_mse",
                    "edge_nonedge_separation",
                ):
                    metrics[name + "_delta_vs_hungarian"] = (
                        metrics[name] - baseline[name]
                    )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_sample_id": source_sample_id,
                        "method": f"fgw_alpha_{alpha:g}",
                        **metrics,
                        **transport_metrics(transports[local_index]),
                        "model_forward_seconds": forward_seconds / batch_count,
                        "structure_scoring_seconds": structure_seconds / batch_count,
                        "matcher_seconds": elapsed / batch_count,
                    }
                )
            sample_index += 1
        if sample_index % args.progress_every == 0 or sample_index == total_samples:
            print(
                f"diagnostic progress: {sample_index}/{total_samples} samples",
                flush=True,
            )
    return rows


def main():
    args = _parser().parse_args()
    _validate_args(args)
    config = copy.deepcopy(load_config(args.config))
    config["runtime"]["device"] = args.device
    config["runtime"]["workers"] = args.workers
    config["data"]["batch_size"] = args.batch_size
    config["data"]["validation_batch_size"] = args.batch_size
    validate_config(config)

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("diagnostic requests CUDA but CUDA is unavailable")

    dataset_name = _dataset_name(config, args.dataset)
    loader = build_evaluation_loader(
        config,
        dataset_name=dataset_name,
        split=args.split,
        max_samples=args.max_samples,
    )
    model = build_model(config).to(device)
    report = load_legacy_model_checkpoint(model, args.checkpoint, map_location="cpu")
    rows = run_diagnostic(model, loader, config, device, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per-sample.csv", rows)
    (output_dir / "per-sample.json").write_text(
        json.dumps(_json_safe(rows), indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset": dataset_name,
        "split": args.split,
        "methods": summarize_rows(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "samples": len(rows) // (1 + len(args.alphas)),
        "alphas": list(args.alphas),
        "candidate_count": args.candidate_count,
        "pair_chunk_size": args.pair_chunk_size,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "progress_every": args.progress_every,
        "ignored_removed_parameters": list(report.ignored_removed),
        "augmentation": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
