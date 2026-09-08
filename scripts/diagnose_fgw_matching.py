#!/usr/bin/env python3
"""Compare Hungarian and hardened-FGW assignments on a fixed checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
from pathlib import Path
import random
import subprocess
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
    fgw_objective_terms,
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
    parser.add_argument("--max-iter", type=int, default=10_000)
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
    edge_squared_error = _safe_mean(
        (aligned_structure[edge_mask] - truth_structure[edge_mask]).square()
    )
    nonedge_squared_error = _safe_mean(
        (aligned_structure[nonedge_mask] - truth_structure[nonedge_mask]).square()
    )
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
        "gt_edge_squared_error": edge_squared_error,
        "gt_nonedge_squared_error": nonedge_squared_error,
        "gt_edge_probability_mean": edge_mean,
        "gt_nonedge_probability_mean": nonedge_mean,
        "edge_nonedge_separation": edge_mean - nonedge_mean,
        "gt_edge_pair_count": int(edge_mask.sum()),
        "gt_nonedge_pair_count": int(nonedge_mask.sum()),
        "graph_pair_count": int(upper.sum()),
        "graph_too_small_for_pair_metrics": int(target_count < 2),
        "edge_metrics_undefined": int(not bool(edge_mask.any())),
        "nonedge_metrics_undefined": int(not bool(nonedge_mask.any())),
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
    """Return concentration diagnostics for a partial transport plan."""

    plan = np.asarray(transport, dtype=np.float64)
    if plan.size == 0:
        return {
            "soft_argmax_collisions": 0,
            "normalized_row_entropy": 0.0,
            "active_prediction_columns": 0,
            "max_prediction_capacity_ratio": 0.0,
            "transport_total_mass": 0.0,
        }
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
    column_mass = plan.sum(axis=0)
    capacity = 1.0 / plan.shape[0]
    return {
        "soft_argmax_collisions": int(collisions),
        "normalized_row_entropy": float(np.mean(entropy)),
        "active_prediction_columns": int(np.count_nonzero(column_mass > 1e-12)),
        "max_prediction_capacity_ratio": float(column_mass.max() / capacity),
        "transport_total_mass": float(plan.sum()),
    }


def objective_diagnostics(
    feature_cost,
    target_structure,
    predicted_structure,
    soft_transport,
    hard_assignment,
    candidates,
    alpha,
):
    """Compare initial, final soft, and globally hardened FGW objectives."""

    feature_cost = np.asarray(feature_cost, dtype=np.float64)
    initial = FusedGromovWassersteinMatcher.initial_transport(feature_cost)
    target_count = feature_cost.shape[0]
    hard = np.zeros_like(feature_cost)
    assignment = _assignment_vector(hard_assignment, target_count)
    local_by_global = {
        int(query): local for local, query in enumerate(candidates.long().cpu().tolist())
    }
    for target, query in enumerate(assignment.tolist()):
        hard[target, local_by_global[query]] = 1.0 / target_count
    stages = {
        "initial": fgw_objective_terms(
            feature_cost, target_structure, predicted_structure, initial, alpha
        ),
        "soft": fgw_objective_terms(
            feature_cost, target_structure, predicted_structure, soft_transport, alpha
        ),
        "hard": fgw_objective_terms(
            feature_cost, target_structure, predicted_structure, hard, alpha
        ),
    }
    names = {
        "feature": "feature_term",
        "structural": "structural_term",
        "weighted_total": "objective",
    }
    result = {
        f"{stage}_{names[name]}": value
        for stage, terms in stages.items()
        for name, value in terms.items()
    }
    initial_total = stages["initial"]["weighted_total"]
    result.update(
        soft_objective_change_vs_hungarian=(
            stages["soft"]["weighted_total"] - initial_total
        ),
        hard_objective_change_vs_hungarian=(
            stages["hard"]["weighted_total"] - initial_total
        ),
        hardening_objective_gap=(
            stages["hard"]["weighted_total"] - stages["soft"]["weighted_total"]
        ),
    )
    return result


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
        undefined_counts = {
            name: sum(int(bool(row.get(name))) for row in selected)
            for name in (
                "graph_too_small_for_pair_metrics",
                "edge_metrics_undefined",
                "nonedge_metrics_undefined",
                "invariant_failure",
            )
        }
        summaries[method] = {
            "samples": len(selected),
            **metrics,
            "undefined_or_failure_counts": undefined_counts,
        }
    return summaries


def summarize_by_target_count(rows):
    """Group method summaries by ground-truth graph size."""

    counts = sorted(
        {int(row["target_count"]) for row in rows if "target_count" in row}
    )
    return {
        str(count): summarize_rows(
            [row for row in rows if row.get("target_count") == count]
        )
        for count in counts
    }


def _run_metadata():
    def git(*arguments):
        try:
            return subprocess.run(
                ("git", *arguments),
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        import ot
        pot_version = ot.__version__
    except ImportError:
        pot_version = None
    status = git("status", "--short")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pot": pot_version,
        "git_revision": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_has_local_changes": None if status is None else bool(status),
        "git_status_short": None if status is None else status.splitlines(),
    }


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


def format_console_summary(summary):
    """Return a compact terminal report while detailed artifacts stay on disk."""

    methods = summary["methods"]
    baseline = methods.get("hungarian", {})
    lines = [
        (
            f"FGW diagnostic: dataset={summary['dataset']} split={summary['split']} "
            f"samples={int(baseline.get('samples', 0))}"
        ),
        (
            "method          graphs changed  targets changed  coordinate Δ  "
            "structure Δ  edge-sep Δ   soft obj Δ   hard obj Δ   "
            "projection gap  iterations  solver ms  failures"
        ),
    ]
    for method, values in methods.items():
        if method == "hungarian":
            continue
        samples = int(values.get("samples", 0))
        changed_graphs = int(round(float(values.get("changed_any", 0.0)) * samples))
        failures = int(
            values.get("undefined_or_failure_counts", {}).get(
                "invariant_failure", 0
            )
        )
        lines.append(
            f"{method:<15} "
            f"{changed_graphs:>3}/{samples:<3} "
            f"{float(values.get('changed_target_fraction', float('nan'))):>14.4%} "
            f"{float(values.get('coordinate_l1_mean_delta_vs_hungarian', float('nan'))):>13.6g} "
            f"{float(values.get('structural_mse_delta_vs_hungarian', float('nan'))):>12.6g} "
            f"{float(values.get('edge_nonedge_separation_delta_vs_hungarian', float('nan'))):>11.6g} "
            f"{float(values.get('soft_objective_change_vs_hungarian', float('nan'))):>12.6g} "
            f"{float(values.get('hard_objective_change_vs_hungarian', float('nan'))):>12.6g} "
            f"{float(values.get('hardening_objective_gap', float('nan'))):>15.6g} "
            f"{float(values.get('solver_iterations', float('nan'))):>10.2f} "
            f"{1000.0 * float(values.get('solver_seconds', float('nan'))):>9.3f} "
            f"{failures:>8}"
        )
    alpha_zero = methods.get("fgw_alpha_0")
    if alpha_zero is not None:
        lines.append(
            "alpha-zero check: "
            f"unary cost Δ={float(alpha_zero.get('alpha_zero_unary_cost_delta', float('nan'))):.6g}, "
            f"changed graphs={float(alpha_zero.get('changed_any', float('nan'))):.4%}, "
            f"capacity ratio={float(alpha_zero.get('max_prediction_capacity_ratio', float('nan'))):.6g}"
        )
    lines.append(
        "Detailed results: summary.json, per-sample.csv, per-sample.json, "
        "metadata.json, resolved-config.yaml"
    )
    return "\n".join(lines)


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
            try:
                assignments, transports, diagnostics = matcher(
                    predictions,
                    targets,
                    predicted_structure=structures,
                    candidate_indices=candidates,
                    return_transport=True,
                    return_diagnostics=True,
                )
                method_results[alpha] = {
                    "assignments": assignments,
                    "transports": transports,
                    "diagnostics": diagnostics,
                    "elapsed": time.perf_counter() - started,
                    "error": None,
                }
            except Exception as error:
                method_results[alpha] = {
                    "elapsed": time.perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                }

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
                    "relation_scoring_seconds": 0.0,
                    "matcher_seconds": hungarian_seconds / batch_count,
                    "solver_seconds": 0.0,
                    "invariant_failure": 0,
                }
            )
            for alpha, result in method_results.items():
                if result["error"] is not None:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source_sample_id": source_sample_id,
                            "method": f"fgw_alpha_{alpha:g}",
                            "target_count": int(targets["nodes"][local_index].shape[0]),
                            "candidate_count": int(candidates[local_index].numel()),
                            "invariant_failure": 1,
                            "failure": result["error"],
                            "model_forward_seconds": forward_seconds / batch_count,
                            "structure_scoring_seconds": structure_seconds / batch_count,
                            "relation_scoring_seconds": structure_seconds / batch_count,
                            "matcher_seconds": result["elapsed"] / batch_count,
                            "solver_seconds": float("nan"),
                        }
                    )
                    continue
                assignments = result["assignments"]
                transports = result["transports"]
                diagnostics = result["diagnostics"][local_index]
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
                target_count = int(targets["nodes"][local_index].shape[0])
                feature_cost = matcher._feature_cost(
                    predictions, targets["nodes"][local_index], local_index
                )[:, candidates[local_index]].detach().double().cpu().numpy()
                truth_structure = matcher.target_structure(
                    target_count,
                    targets["edges"][local_index],
                    dtype=structures[local_index].dtype,
                    device=structures[local_index].device,
                ).detach().double().cpu().numpy()
                prediction_structure = (
                    structures[local_index].detach().double().cpu().numpy().copy()
                )
                np.fill_diagonal(prediction_structure, 0.0)
                objectives = objective_diagnostics(
                    feature_cost,
                    truth_structure,
                    prediction_structure,
                    transports[local_index],
                    assignments[local_index],
                    candidates[local_index],
                    alpha,
                )
                reference_queries = set(
                    _assignment_vector(reference, target_count).tolist()
                )
                matched_queries = set(metrics["matched_query_ids"])
                retained = len(reference_queries & matched_queries)
                mapping_changed = bool(metrics["changed_any"])
                if alpha == 0.0:
                    unary_delta = objectives["hard_feature_term"] - objectives[
                        "initial_feature_term"
                    ]
                    alpha_zero_status = (
                        "same_mapping"
                        if not mapping_changed
                        else "tied_optimum"
                        if abs(unary_delta) <= args.tolerance
                        else "unary_cost_regression"
                    )
                else:
                    unary_delta = float("nan")
                    alpha_zero_status = "not_applicable"
                flat_solver = {
                    f"solver_{name}": value
                    for name, value in diagnostics.items()
                    if name != "solver_seconds"
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_sample_id": source_sample_id,
                        "method": f"fgw_alpha_{alpha:g}",
                        **metrics,
                        **transport_metrics(transports[local_index]),
                        **objectives,
                        **flat_solver,
                        "alpha_zero_status": alpha_zero_status,
                        "alpha_zero_unary_cost_delta": unary_delta,
                        "candidate_extra_query_count": int(candidates[local_index].numel()) - target_count,
                        "candidate_retains_all_hungarian_queries": int(
                            reference_queries.issubset(
                                set(candidates[local_index].long().cpu().tolist())
                            )
                        ),
                        "query_subset_changed_vs_hungarian": int(
                            matched_queries != reference_queries
                        ),
                        "query_subset_replaced_fraction": 1.0 - retained / target_count,
                        "invariant_failure": 0,
                        "model_forward_seconds": forward_seconds / batch_count,
                        "structure_scoring_seconds": structure_seconds / batch_count,
                        "relation_scoring_seconds": structure_seconds / batch_count,
                        "matcher_seconds": result["elapsed"] / batch_count,
                        "solver_seconds": diagnostics.get("solver_seconds", float("nan")),
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
        "by_target_count": summarize_by_target_count(rows),
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
        "seed": seed,
        "device": str(device),
        "versions_and_revision": _run_metadata(),
        "solver": {
            "name": "POT partial_fused_gromov_wasserstein",
            "algorithm": "conditional_gradient_with_emd",
            "loss_fun": "square_loss",
            "symmetric": True,
            "max_iter": args.max_iter,
            "tolerance": args.tolerance,
            "initialization": "unary_hungarian",
            "target_mass": "uniform_1_over_target_count",
            "candidate_capacity": "uniform_1_over_target_count",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "resolved-config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(format_console_summary(summary))


if __name__ == "__main__":
    main()
