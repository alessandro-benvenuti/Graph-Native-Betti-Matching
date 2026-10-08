#!/usr/bin/env python3
"""Profile full and graph-subsampled Betti training on fixed train streams."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from configs import load_config, validate_config
from data.loaders import build_data_loaders
from models import build_model
from training.checkpoint import load_runtime_state
from training.engine import _move_graph_batch
from training.losses import build_criterion


MODES = (
    "baseline",
    "h0_full",
    "h1_full",
    "both_full",
    "both_sample_2",
    "both_sample_4",
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--gradient-repeats", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=364505)
    return parser


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mode_config(base, mode):
    config = copy.deepcopy(base)
    config["runtime"].update(device="cuda", distributed=False)
    config["tracking"]["enabled"] = False
    config["evaluation"]["training_metrics"]["enabled"] = False
    for name in ("betti_h0", "betti_h1"):
        config["topology"][name].update(
            enabled=False,
            log_only=False,
            weight=0.0,
            warmup_epochs=0,
            ramp_epochs=0,
            activation_epoch=None,
        )
    config["topology"]["sampling"].update(
        enabled=False,
        max_graphs_per_rank=None,
        betti_h0_max_graphs_per_rank=None,
        betti_h1_max_graphs_per_rank=None,
    )
    if mode != "baseline":
        if mode in {"h0_full", "both_full", "both_sample_2", "both_sample_4"}:
            config["topology"]["betti_h0"].update(enabled=True, weight=0.003)
        if mode in {"h1_full", "both_full", "both_sample_2", "both_sample_4"}:
            config["topology"]["betti_h1"].update(
                enabled=True, weight=0.003, false_positive_weight=0.1
            )
    if mode.startswith("both_sample_"):
        limit = int(mode.rsplit("_", 1)[1])
        config["topology"]["sampling"].update(
            enabled=True, max_graphs_per_rank=limit
        )
    validate_config(config)
    return config


def _load_model(config, checkpoint, device):
    model = build_model(config).to(device)
    payload = load_runtime_state(checkpoint)
    if "net" not in payload:
        raise ValueError("checkpoint does not contain model state under 'net'")
    model.load_state_dict(payload["net"], strict=True)
    model.eval()
    return model, int(payload.get("epoch", -1))


def _loader(config, seed):
    _seed_everything(seed)
    train, _ = build_data_loaders(config, rank=0, world_size=1)
    return train


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _profile_mode(base, model, mode, args, device):
    config = _mode_config(base, mode)
    config["data"]["batch_size"] = args.batch_size
    config["runtime"]["workers"] = args.workers
    loader = _loader(config, args.seed)
    criterion = build_criterion(config, model).to(device)
    criterion.set_training_progress(301, 100.0)
    times = []
    coverage = {
        "topology_betti_h0_eligible_graphs": [],
        "topology_betti_h0_selected_graphs": [],
        "topology_betti_h1_eligible_graphs": [],
        "topology_betti_h1_selected_graphs": [],
        "topology_scored_graphs": [],
    }
    needed = args.warmup_steps + args.measured_steps
    processed = 0
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        prepared = _move_graph_batch(
            batch,
            device,
            config["training"]["input"],
            bool(config["loss"]["supervise_target_graphs"]),
        )
        if prepared is None:
            continue
        volumes, targets = prepared
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        tokens, predictions, _ = model(volumes)
        losses = criterion(tokens, predictions, targets)
        losses["total"].backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if processed >= args.warmup_steps:
            times.append(elapsed)
            for name in coverage:
                coverage[name].append(float(losses[name].detach().cpu()))
        processed += 1
        if processed >= needed:
            break
    if len(times) != args.measured_steps:
        raise RuntimeError(
            f"{mode} produced {len(times)} measured steps, expected {args.measured_steps}"
        )
    return {
        "mode": mode,
        "steps": len(times),
        "mean_step_seconds": statistics.fmean(times),
        "median_step_seconds": statistics.median(times),
        "p90_step_seconds": _percentile(times, 0.9),
        "steps_per_second": 1.0 / statistics.fmean(times),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        **{
            name + "_mean": statistics.fmean(values)
            for name, values in coverage.items()
        },
    }


def _derived_components(rows):
    timing = {row["mode"]: row["mean_step_seconds"] for row in rows}
    base = timing["baseline"]
    h0 = timing["h0_full"]
    h1 = timing["h1_full"]
    both = timing["both_full"]
    return {
        "baseline_step_seconds": base,
        "shared_pair_scoring_seconds_estimate": h0 + h1 - both - base,
        "h0_specific_seconds_estimate": both - h1,
        "h1_specific_seconds_estimate": both - h0,
        "full_topology_overhead_seconds": both - base,
        "sample_2_speedup_vs_full_betti": both / timing["both_sample_2"],
        "sample_4_speedup_vs_full_betti": both / timing["both_sample_4"],
        "sample_2_slowdown_vs_baseline": timing["both_sample_2"] / base,
        "sample_4_slowdown_vs_baseline": timing["both_sample_4"] / base,
    }


def _flatten_gradients(gradients, references):
    chunks = []
    for gradient, reference in zip(gradients, references):
        chunks.append(
            torch.zeros_like(reference).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    return torch.cat(chunks)


def _cosine(left, right):
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def _norm_ratio(value, reference):
    denominator = float(reference.norm())
    if denominator == 0.0:
        return float("nan")
    return float(value.norm()) / denominator


def _topology_gradient(criterion, tokens, predictions, targets, assignments, parameters):
    losses = criterion.loss_topology(
        tokens, predictions["pred_logits"], targets["edges"], assignments
    )
    loss = 0.003 * losses["betti_h0"] + 0.003 * losses["betti_h1"]
    references = [*parameters, predictions["pred_logits"]]
    gradients = torch.autograd.grad(
        loss,
        references,
        retain_graph=True,
        allow_unused=True,
    )
    relation = _flatten_gradients(gradients[:-1], parameters)
    nodes = _flatten_gradients(gradients[-1:], references[-1:])
    return float(loss.detach()), relation, nodes


def _gradient_agreement(base, model, args, device):
    config = _mode_config(base, "both_full")
    config["data"]["batch_size"] = args.batch_size
    config["runtime"]["workers"] = args.workers
    batch = next(iter(_loader(config, args.seed + 17)))
    prepared = _move_graph_batch(
        batch,
        device,
        config["training"]["input"],
        bool(config["loss"]["supervise_target_graphs"]),
    )
    if prepared is None:
        raise RuntimeError("gradient diagnostic received an empty supervised batch")
    volumes, targets = prepared
    model.zero_grad(set_to_none=True)
    tokens, predictions, _ = model(volumes)
    full = build_criterion(config, model).to(device)
    assignments = full.matcher(predictions, targets)
    parameters = tuple(model.relation_embed.parameters())
    full_loss, full_relation, full_nodes = _topology_gradient(
        full, tokens, predictions, targets, assignments, parameters
    )
    result = {"full_loss": full_loss, "budgets": {}}
    for limit in (2, 4):
        sampled_config = _mode_config(base, f"both_sample_{limit}")
        sampled = build_criterion(sampled_config, model).to(device)
        losses = []
        relation_cosines = []
        node_cosines = []
        relation_norm_ratios = []
        node_norm_ratios = []
        relation_gradients = []
        node_gradients = []
        for _ in range(args.gradient_repeats):
            loss, relation, nodes = _topology_gradient(
                sampled, tokens, predictions, targets, assignments, parameters
            )
            losses.append(loss)
            relation_gradients.append(relation)
            node_gradients.append(nodes)
            relation_cosines.append(_cosine(relation, full_relation))
            node_cosines.append(_cosine(nodes, full_nodes))
            relation_norm_ratios.append(_norm_ratio(relation, full_relation))
            node_norm_ratios.append(_norm_ratio(nodes, full_nodes))
        mean_relation = torch.stack(relation_gradients).mean(0)
        mean_nodes = torch.stack(node_gradients).mean(0)
        result["budgets"][str(limit)] = {
            "repeats": args.gradient_repeats,
            "loss_mean": statistics.fmean(losses),
            "loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "loss_bias_vs_full": statistics.fmean(losses) - full_loss,
            "relation_gradient_cosine_mean": statistics.fmean(relation_cosines),
            "relation_mean_gradient_cosine": _cosine(mean_relation, full_relation),
            "relation_gradient_norm_ratio_mean": statistics.fmean(relation_norm_ratios),
            "node_gradient_cosine_mean": statistics.fmean(node_cosines),
            "node_mean_gradient_cosine": _cosine(mean_nodes, full_nodes),
            "node_gradient_norm_ratio_mean": statistics.fmean(node_norm_ratios),
        }
    return result


def _write_outputs(output, payload):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload["timings"]
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parser().parse_args()
    if args.warmup_steps < 1 or args.measured_steps < 1 or args.gradient_repeats < 2:
        raise ValueError("warmup/measured steps must be positive and repeats >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires CUDA")
    device = torch.device("cuda:0")
    base = load_config(args.config)
    base["data"]["batch_size"] = args.batch_size
    base["runtime"]["workers"] = args.workers
    model, checkpoint_epoch = _load_model(base, args.checkpoint, device)
    timings = []
    for mode in MODES:
        print(f"profiling mode={mode}", flush=True)
        row = _profile_mode(base, model, mode, args, device)
        timings.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print("profiling gradient agreement", flush=True)
    gradient_agreement = _gradient_agreement(base, model, args, device)
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "seed": args.seed,
        "batch_size_per_gpu": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "timings": timings,
        "derived_components": _derived_components(timings),
        "gradient_agreement": gradient_agreement,
    }
    _write_outputs(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("profile complete:", args.output, flush=True)


if __name__ == "__main__":
    main()
