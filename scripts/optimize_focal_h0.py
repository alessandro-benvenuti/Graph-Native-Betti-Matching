#!/usr/bin/env python3
"""H0-focused TPE study with a frozen existing focal control and finite promotions.

Only epoch-525 objectives are submitted to Optuna. Longer budgets are independent
reports, selected manually from their Pareto trade-offs. No test data is read.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config, validate_config
from scripts.optimize_node_edge_betti import (
    AllocationLock, CampaignError, OBJECTIVES, _file_sha256, _read_records,
    aggregate_tail, create_storage, run_command, utc_now,
)

H1_ABLATION_WEIGHTS = (0.001, 0.003, 0.01, 0.03)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise CampaignError(f"Refusing to change frozen inputs: {path}")
    atomic_json(path, value)


def checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def add_signed_errors(metrics):
    for suffix in ("beta0", "beta1", "nodes", "edges"):
        predicted, target = f"predicted_{suffix}", f"target_{suffix}"
        if predicted in metrics and target in metrics:
            metrics[f"signed_{suffix}_error"] = metrics[predicted] - metrics[target]


def scientific_reference(config, settings):
    """Reject a plausible-looking but scientifically different control."""
    node, edge = config["loss"]["node"]["classification"], config["loss"]["edge"]
    checks = {
        "node focal gamma=2": node["name"] == "focal" and node["focal_gamma"] == 2,
        "unweighted immediate node focal": node["class_weights"] == [1., 1.] and not node["curriculum"]["enabled"],
        "edge focal gamma=0.5": edge["classification"]["name"] == "focal" and edge["classification"]["focal_gamma"] == .5,
        "immediate edge focal": not edge["classification"]["curriculum"]["enabled"],
        "matched-only edges": not edge["candidates"]["include_unmatched"],
        "no ratio upsampling": edge["balancing"]["mode"] == "none",
        "Hungarian matcher": config["model"]["matcher"]["type"] == "hungarian",
        "continuation horizon": config["training"]["epochs"] == settings["continuation_epochs"],
        "tail schedule": config["training"]["scheduler"]["duration_epochs"] == settings["schedule_duration_epochs"],
        "learning rate": config["training"]["optimizer"]["base_lr"] == settings["base_lr"],
        "no LR warmup": config["training"]["warmup_epochs"] == 0,
        "validation frequency": config["evaluation"]["interval_epochs"] == settings["validation_interval"],
        "full training data": config["data"]["datasets"]["synthetic_mri"].get("train_samples") is None,
        "full validation data": config["data"]["datasets"]["synthetic_mri"].get("validation_samples") is None,
        "full metric evaluation": config["evaluation"]["training_metrics"]["max_samples"] is None,
        "edge mAP checkpoint selection": config["evaluation"]["training_metrics"]["selection_metric"] == "edge_mAP",
        "50-epoch patience": config["training"]["early_stopping"]["enabled"] and config["training"]["early_stopping"]["patience_epochs"] == 50,
        "edge-mAP stopping": config["training"]["early_stopping"]["monitor"] == "edge_mAP" and config["training"]["early_stopping"]["mode"] == "max" and config["training"]["early_stopping"]["min_delta"] == 0,
    }
    for name in ("betti_h0", "betti_h1"):
        item = config["topology"][name]
        checks[f"control {name} inactive"] = not item["enabled"] or item["log_only"] or float(item["weight"]) == 0
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise CampaignError("Incompatible focal reference: " + ", ".join(failed))


def load_settings(path):
    settings = yaml.safe_load(path.read_text())
    branch, endpoints = settings["branch_epoch"], settings["endpoints"]
    if len(endpoints) != 3 or not branch < endpoints[0] < endpoints[1] < endpoints[2] <= settings["continuation_epochs"]:
        raise CampaignError("Need three increasing continuation endpoints")
    if settings["gpus"] < 1 or settings["global_batch_size"] % settings["gpus"]:
        raise CampaignError("Global batch must divide evenly across GPUs")
    low, high = settings["h0_range"]
    if not 0 < low < high or not all(low <= w <= high for w in settings["h0_anchors"]):
        raise CampaignError("Invalid logarithmic H0 search range or anchors")
    if len(settings["h0_anchors"]) > settings["trials"]:
        raise CampaignError("More anchors than trials")
    if not settings["trials"] >= settings["promotion_counts"][0] >= settings["promotion_counts"][1] >= 1:
        raise CampaignError("Invalid promotion counts")
    if settings["tail_observations"] * settings["validation_interval"] > endpoints[0] - branch:
        raise CampaignError("Screening is too short for its validation tail")
    return settings


def prepare(args):
    settings = load_settings(args.settings)
    source, reference = args.source_run.resolve(), args.reference_run.resolve()
    source_checkpoint = source / "models/latest_checkpoint.pt"
    # Hold the small preparation lock only; it never locks existing training.
    with AllocationLock(args.output / ".locks/prepare.lock"):
        if (args.output / "campaign.json").is_file():
            saved = state(args.output)
            if saved["settings"] != settings or saved["source_run"] != str(source) or saved["reference_run"] != str(reference):
                raise CampaignError("Campaign already exists with different inputs")
            print("Prepared campaign already exists; reusing frozen inputs")
            return
        base = load_config(reference / "resolved-config.yaml")
        scientific_reference(base, settings)
        # This is historical provenance, not a configuration we train with.
        # Older resolved files legitimately lack later topology/sampling keys.
        # Validate the current reference above, then compare the original
        # scientific sections directly without requiring today's full schema.
        original = yaml.safe_load((source / "resolved-config.yaml").read_text())
        if not isinstance(original, dict):
            raise CampaignError("Historical source configuration must be a mapping")
        for key in ("loss", "model", "data", "augmentation"):
            if not isinstance(original.get(key), dict):
                raise CampaignError(f"Historical source configuration lacks {key}")
            # Batch sizes can legitimately change from original 2-GPU training.
            left, right = copy.deepcopy(original[key]), copy.deepcopy(base[key])
            if key == "data":
                for field in ("batch_size", "validation_batch_size"):
                    left.pop(field, None)
                    right.pop(field, None)
            if left != right:
                raise CampaignError(f"Source and reference differ in {key}")
        payload = checkpoint(source_checkpoint)
        if int(payload["epoch"]) != settings["branch_epoch"]:
            raise CampaignError("Source checkpoint is not at the branch epoch")
        if not all(key in payload for key in ("optimizer", "scheduler", "net")):
            raise CampaignError("Source must contain full model/optimizer/scheduler state")
        original_world = int(payload.get("trainer_state", {}).get("world_size", 1))
        del payload
        # The running reference may advance: inspect launch provenance, not its
        # live latest checkpoint, to verify the original restart protocol.
        provenance = json.loads((reference / "extension-source.json").read_text())
        required = {
            "source_epoch": settings["branch_epoch"], "target_epoch": settings["continuation_epochs"],
            "scheduler_restart_base_lr": settings["base_lr"],
            "scheduler_duration_epochs": settings["schedule_duration_epochs"],
            "early_stopping_reset": True, "source_world_size": original_world,
            "continuation_world_size": settings["gpus"], "global_batch_size": settings["global_batch_size"],
        }
        if any(provenance.get(key) != value for key, value in required.items()):
            raise CampaignError("Existing control extension-source.json differs from the study protocol")
        recorded_source = provenance.get("resume_checkpoint")
        if not recorded_source or _file_sha256(Path(recorded_source)) != _file_sha256(source_checkpoint):
            raise CampaignError("Existing control did not restart from the selected epoch-500 checkpoint")
        manifest = json.loads((reference / "dataset-manifest.json").read_text())
        if manifest != json.loads((source / "dataset-manifest.json").read_text()):
            raise CampaignError("Source/reference dataset manifests differ")
        initial = args.output / "initialization/epoch_0500.pt"
        initial.parent.mkdir(parents=True, exist_ok=True)
        digest = _file_sha256(source_checkpoint)
        if initial.exists() and _file_sha256(initial) != digest:
            raise CampaignError("Initialization copy differs")
        if not initial.exists():
            temporary = initial.with_suffix(".pt.tmp")
            shutil.copy2(source_checkpoint, temporary)
            temporary.replace(initial)
        if _file_sha256(initial) != digest:
            raise CampaignError("Initialization copy checksum failed")
        bootstrap = args.output / "initialization/metric-history.jsonl"
        records = [r for r in _read_records(source / "validation-metrics.jsonl") if int(r["epoch"]) <= settings["branch_epoch"]]
        if not records or int(records[-1]["epoch"]) != settings["branch_epoch"]:
            raise CampaignError("Source metric history does not reach branch epoch")
        bootstrap.write_text("".join(json.dumps(r) + "\n" for r in records))
        # Clone the actual control, rather than reconstructing it from defaults.
        base["experiment"]["output_dir"] = str(args.output)
        base["runtime"].update(distributed=settings["gpus"] > 1)
        base["data"]["batch_size"] = settings["global_batch_size"] // settings["gpus"]
        base["tracking"].update(project="gnbm", group="edge-focal-g05-h0-study", bootstrap_metrics_path=str(bootstrap))
        base["tracking"]["tags"] = ["gnbm", "edge-focal-g05", "h0-study", "validation-only"]
        base["tracking"].pop("bootstrap_history_path", None)
        base["training"]["checkpoint"].update(latest_interval_epochs=1, milestone_epochs=settings["endpoints"])
        validate_config(base)
        freeze(args.output / "base-config.json", base)
        freeze(args.output / "dataset-manifest.json", manifest)
        freeze(args.output / "campaign.json", {
            "settings": settings, "source_run": str(source), "reference_run": str(reference),
            "checkpoint": str(initial), "checkpoint_sha256": digest,
            "original_world_size": original_world, "test_split_used": False,
            "reference_config_sha256": _file_sha256(reference / "resolved-config.yaml"),
        })
    print(f"Prepared H0 study: {args.output}; no control training submitted")


def state(output):
    return json.loads((output / "campaign.json").read_text())


def stage_reference(output, endpoint):
    saved = state(output)
    settings = saved["settings"]
    if endpoint not in settings["endpoints"]:
        raise CampaignError("Endpoint is not a configured fidelity")
    path = output / "references" / f"epoch_{endpoint:04d}.json"
    if path.exists():
        return json.loads(path.read_text())
    interval, count = settings["validation_interval"], settings["tail_observations"]
    epochs = list(range(endpoint - interval * (count - 1), endpoint + 1, interval))
    from scripts.optimize_node_edge_betti import JsonlMonitor
    records = {int(r["epoch"]): r for r in JsonlMonitor(Path(saved["reference_run"]) / "validation-metrics.jsonl").read_new()}
    if not all(epoch in records for epoch in epochs):
        raise CampaignError(f"Existing control lacks required epochs {epochs}")
    aggregation = aggregate_tail([records[e] for e in epochs], count)
    add_signed_errors(aggregation["metrics"])
    with AllocationLock(output / ".locks/reference.lock"):
        freeze(path, aggregation)
    return aggregation


def study_for(output, family, worker=0):
    import optuna
    saved = state(output)
    settings = saved["settings"]
    storage_dir = output / "studies" / family
    storage_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"focal-g05-{family}-e{settings['branch_epoch']}",
        storage=create_storage(optuna, storage_dir, "journal"),
        directions=[direction for _, direction in OBJECTIVES],
        sampler=optuna.samplers.TPESampler(seed=settings["seed"] + worker, n_startup_trials=settings["tpe_startup_trials"]),
        load_if_exists=True,
    )
    return study


def initialize_study(output):
    with AllocationLock(output / ".locks/ask.lock"):
        study = study_for(output, "h0")
        if not study.trials:
            for weight in state(output)["settings"]["h0_anchors"]:
                study.enqueue_trial({"h0_weight": weight})


def validate_h1_weights(values):
    try:
        weights = [float(value) for value in values]
    except (ValueError, TypeError) as error:
        raise CampaignError("H1 weights must be finite nonnegative numbers") from error
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise CampaignError("H1 weights must be finite nonnegative numbers")
    if len(set(weights)) != len(weights):
        raise CampaignError("Duplicate H1 ablation weights")
    return weights


def ablation_reference(output, endpoint):
    """Return the H0-only comparator only at the requested, identical fidelity."""
    plan = json.loads((output / "h1-plan.json").read_text())
    if endpoint == plan["endpoint"] and "h0_reference" in plan:
        row = plan["h0_reference"]
    else:
        path = result_path(output, "h0", endpoint, plan["parent_trial"])
        if not path.exists():
            return None
        row = json.loads(path.read_text())
    reference = stage_reference(output, endpoint)
    if (row["endpoint"] != endpoint
            or row["aggregation"]["contributing_epochs"] != reference["contributing_epochs"]
            or row["control"] != reference["metrics"]):
        raise CampaignError("H0 ablation reference does not match the control fidelity")
    return row


def ablation_prepare(output, parent, h1_weights=None):
    saved = state(output)
    initial_endpoint = saved["settings"]["endpoints"][0]
    path = result_path(output, "h0", initial_endpoint, parent)
    if not path.is_file():
        raise CampaignError(f"Select an H0 parent that completed epoch {initial_endpoint}")
    row = json.loads(path.read_text())
    reference = stage_reference(output, initial_endpoint)
    if (row["family"] != "h0" or row["trial"] != parent
            or row["endpoint"] != initial_endpoint
            or row["aggregation"]["contributing_epochs"] != reference["contributing_epochs"]
            or row["control"] != reference["metrics"]
            or float(row["parameters"].get("h1_weight", 0)) != 0):
        raise CampaignError("Parent must be an epoch-matched, completed H0-only result")
    weight = float(row["parameters"]["h0_weight"])
    if not math.isfinite(weight) or weight <= 0:
        raise CampaignError("Parent must have a finite positive H0 weight")
    with AllocationLock(output / ".locks/ask.lock"):
        plan_path = output / "h1-plan.json"
        existing = json.loads(plan_path.read_text()) if plan_path.exists() else None
        # The ablation plan, not campaign.json, owns this new finite grid. Never
        # rewrite frozen H0 settings; already prepared legacy grids remain valid.
        defaults = existing["h1_weights"] if existing else H1_ABLATION_WEIGHTS
        weights = validate_h1_weights(defaults if h1_weights is None else h1_weights)
        identity = {"parent_trial": parent, "h0_weight": weight,
                    "h1_weights": weights, "endpoint": initial_endpoint}
        if existing:
            if any(existing.get(key) != value for key, value in identity.items()):
                raise CampaignError("Refusing to change frozen H1 ablation plan")
            if existing.get("parent_result_sha256", _file_sha256(path)) != _file_sha256(path):
                raise CampaignError("Frozen H0 parent result changed")
            plan = existing
        else:
            plan = {**identity, "schema_version": 2,
                    "parent_result_sha256": _file_sha256(path),
                    "h0_reference": row,
                    "initial_checkpoint": saved["checkpoint"],
                    "initial_checkpoint_sha256": saved["checkpoint_sha256"],
                    "h1_false_positive_weight": saved["settings"]["h1_false_positive_weight"],
                    "h1_false_negative_weight": saved["settings"]["h1_false_negative_weight"],
                    "h1_graphs_per_rank": 1, "test_split_used": False}
        freeze(plan_path, plan)
        study = study_for(output, "h1")
        enqueued = set()
        for trial in study.trials:
            parameters = trial.params or trial.system_attrs.get("fixed_params", {})
            if (parameters.get("h0_weight") != weight
                    or parameters.get("h1_weight") not in weights
                    or parameters["h1_weight"] in enqueued):
                raise CampaignError("H1 journal differs from the frozen finite ablation grid")
            enqueued.add(parameters["h1_weight"])
        # Also recover an interruption partway through enqueueing the grid.
        for h1 in weights:
            if h1 not in enqueued:
                study.enqueue_trial({"h0_weight": weight, "h1_weight": h1})
    print(f"Prepared H1 ablation: H0={weight:g}; H1={weights}; "
          f"H0-only reference=trial {parent} at epoch {initial_endpoint}; "
          f"all new runs restart from epoch {saved['settings']['branch_epoch']}")


def concrete_config(output, family, number, parameters, endpoint):
    saved = state(output)
    settings = saved["settings"]
    config = json.loads((output / "base-config.json").read_text())
    config["experiment"]["name"] = f"runs/{family}/trial_{number:04d}"
    config["tracking"]["run_name"] = f"g05-{family}-{number:03d}"
    config["training"]["stop_after_epoch"] = endpoint
    config["topology"]["complex"].update(mode="node_aware", aggregation="hybrid", alpha=.5, detach_unmatched_edge_probabilities=True)
    config["topology"]["sampling"].update(
        enabled=parameters.get("h1_weight", 0.) > 0, max_graphs_per_rank=None,
        betti_h0_max_graphs_per_rank=settings["global_batch_size"] // settings["gpus"],
        betti_h1_max_graphs_per_rank=1,
    )
    for name, weight in (("betti_h0", parameters["h0_weight"]), ("betti_h1", parameters.get("h1_weight", 0.))):
        config["topology"][name].update(
            enabled=weight > 0, log_only=False, weight=float(weight),
            normalization="matched_mean", warmup_epochs=0,
            activation_epoch=settings["branch_epoch"], ramp_epochs=settings["ramp_epochs"],
        )
    config["topology"]["betti_h1"].update(false_positive_weight=settings["h1_false_positive_weight"], false_negative_weight=settings["h1_false_negative_weight"])
    config["staged_metadata"] = {"h0_focal_study": True, "family": family, "trial": number, "branch_epoch": settings["branch_epoch"], "endpoint": endpoint}
    validate_config(config)
    return config


def single_node_launcher(gpus):
    """Use the agent-owned static store on the cluster's PyTorch 2.3 runtime.

    Unlike standalone/c10d in 2.3, static rendezvous tells workers to reuse the
    launcher's store instead of having rank zero bind a second TCPStore. Select
    an available loopback port per invocation, not a shared fixed/default port.
    Port selection is advisory until torchrun binds it (as with any free-port
    probe); a unique rendezvous ID also isolates independent training launches.
    """
    command = [sys.executable, "-u"]
    if gpus > 1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command += [
            "-m", "torch.distributed.run", "--rdzv-backend=static",
            f"--rdzv-id={uuid.uuid4().hex}", "--master-addr=127.0.0.1",
            f"--master-port={port}", "--nnodes=1", "--node-rank=0",
            f"--nproc_per_node={gpus}",
        ]
    return command


def training_command(output, config_path, run_name, resume, first, train_command=None):
    saved = state(output)
    if train_command:
        command = [part.format(config=config_path, output=output, run_name=run_name) for part in train_command]
    else:
        gpus = saved["settings"]["gpus"]
        command = single_node_launcher(gpus)
        command += [str(ROOT / "train.py"), "--config", str(config_path), "--output-dir", str(output), "--run-name", run_name]
        if gpus > 1:
            command.append("--distributed")
    command += ["--resume", str(resume)]
    if first:
        command += ["--reset-scheduler-on-resume", "--reset-early-stopping-on-resume"]
        if saved["original_world_size"] != saved["settings"]["gpus"]:
            command.append("--allow-world-size-change-on-resume")
    return command


def result_path(output, family, endpoint, number):
    return output / "results" / family / f"epoch_{endpoint:04d}" / f"trial_{number:04d}.json"


def evaluate_trial(args, family, trial, endpoint):
    output, settings = args.output, state(args.output)["settings"]
    reference = stage_reference(output, endpoint)
    target = result_path(output, family, endpoint, trial.number)
    if target.exists():
        return json.loads(target.read_text())
    config = concrete_config(output, family, trial.number, trial.params, endpoint)
    run_name = config["experiment"]["name"]
    run_dir = output / run_name
    config_path = output / "configs" / family / f"trial_{trial.number:04d}-e{endpoint:04d}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False)
    if config_path.exists() and config_path.read_text() != text:
        raise CampaignError("Saved trial configuration changed")
    config_path.write_text(text)
    own = run_dir / "models/latest_checkpoint.pt"
    first = not own.exists()
    resume = Path(state(output)["checkpoint"]) if first else own
    payload = checkpoint(resume)
    start = int(payload["epoch"])
    if not settings["branch_epoch"] <= start <= endpoint:
        raise CampaignError("Checkpoint outside trial phase")
    if not first and int(payload.get("trainer_state", {}).get("world_size", -1)) != settings["gpus"]:
        raise CampaignError("Strict trial resume requires the configured world size")
    stopped = payload.get("trainer_state", {}).get("early_stopping", {}).get("stopped", False)
    if stopped and not first and start < endpoint:
        atomic_json(target.with_suffix(".incomplete.json"), {
            "trial": trial.number, "endpoint": endpoint, "state": "EARLY_STOPPED",
            "actual_epoch": start, "parameters": dict(trial.params),
        })
        raise CampaignError("Trial already early-stopped; refusing to restart patience")
    del payload
    if start < endpoint:
        code = run_command(training_command(output, config_path, run_name, resume, first, args.train_command), run_dir, poll=args.poll_interval)
        if code:
            raise CampaignError(f"Training exited with status {code}; inspect {run_dir}")
    final = checkpoint(own)
    if int(final["epoch"]) != endpoint:
        status_path = run_dir / "training-status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("reason") == "early_stopping":
            atomic_json(target.with_suffix(".incomplete.json"), {
                "trial": trial.number, "endpoint": endpoint, "state": "EARLY_STOPPED",
                "actual_epoch": int(final["epoch"]), "parameters": dict(trial.params),
            })
            raise CampaignError("Trial early-stopped before the required endpoint; no endpoint objective assigned")
        raise CampaignError("Training did not produce the endpoint checkpoint")
    del final
    manifest = json.loads((run_dir / "dataset-manifest.json").read_text())
    if manifest != json.loads((output / "dataset-manifest.json").read_text()):
        raise CampaignError("Trial dataset manifest differs from existing focal control")
    records = {int(r["epoch"]): r for r in _read_records(run_dir / "validation-metrics.jsonl")}
    epochs = reference["contributing_epochs"]
    if not all(epoch in records for epoch in epochs):
        raise CampaignError("Trial lacks exact endpoint tail epochs")
    aggregation = aggregate_tail([records[e] for e in epochs], settings["tail_observations"])
    metrics = aggregation["metrics"]
    add_signed_errors(metrics)
    deltas = {name: metrics[name] - value for name, value in reference["metrics"].items() if name in metrics}
    result = {
        "family": family, "trial": trial.number, "endpoint": endpoint,
        "parameters": dict(trial.params), "aggregation": aggregation,
        "control": reference["metrics"], "metrics": metrics, "deltas": deltas,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "completed_at": utc_now(),
    }
    performance_path = run_dir / "performance.jsonl"
    if performance_path.exists():
        performance = [r for r in _read_records(performance_path) if settings["branch_epoch"] < int(r["epoch"]) <= endpoint]
        seconds = [float(r["train_seconds"]) for r in performance if "train_seconds" in r]
        if seconds:
            result["mean_train_seconds_per_epoch"] = sum(seconds) / len(seconds)
    atomic_json(target, result)
    return result


@contextmanager
def trial_lock(output, family, number):
    path = output / ".locks" / family / f"trial_{number:04d}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def manual_stop_path(output, family, number):
    return output / f"runs/{family}/trial_{number:04d}/manual-stop.json"


def prune_trials(args):
    """Finalize explicitly stopped screening trials; never kill jobs or delete data."""
    import optuna
    reason = (args.reason or "").strip()
    if not reason:
        raise CampaignError("Manual pruning requires a nonempty --reason")
    try:
        numbers = [int(part) for part in (args.trials or "").split(",")]
    except ValueError as error:
        raise CampaignError("--trials must contain comma-separated trial numbers") from error
    if any(n < 0 for n in numbers) or len(set(numbers)) != len(numbers):
        raise CampaignError("Trial numbers must be distinct and nonnegative")
    output, family = args.output, args.family
    endpoint = state(output)["settings"]["endpoints"][0]
    with AllocationLock(output / ".locks/ask.lock"), ExitStack() as locks:
        study = study_for(output, family)
        trials = {trial.number: trial for trial in study.trials}
        if any(number not in trials for number in numbers):
            raise CampaignError("Requested trial does not exist; nothing was pruned")
        # Acquire every target before changing any trial. A live worker keeps
        # this lock throughout training, so scancel alone cannot race this step.
        for number in sorted(numbers):
            if not locks.enter_context(trial_lock(output, family, number)):
                raise CampaignError(f"Trial {number} is still locked by a worker; cancel its Slurm job and wait, then retry")
        actions = []
        for number in numbers:
            trial = trials[number]
            if trial.state.is_finished():
                actions.append((trial, None, None))
                continue
            target = result_path(output, family, endpoint, number)
            if target.exists():
                # The worker may have finished the endpoint just before scancel,
                # but been interrupted between saving its result and study.tell.
                row = json.loads(target.read_text())
                if (row["family"] != family or row["trial"] != number
                        or row["endpoint"] != endpoint or row["parameters"] != trial.params
                        or row["aggregation"]["contributing_epochs"] != stage_reference(output, endpoint)["contributing_epochs"]):
                    raise CampaignError(f"Trial {number} has an incompatible completed result; refusing to overwrite it")
                values = [float(row["metrics"][key]) for key, _ in OBJECTIVES]
                if not all(math.isfinite(value) for value in values):
                    raise CampaignError(f"Trial {number} has non-finite endpoint results")
                actions.append((trial, values, None))
                continue
            from scripts.optimize_node_edge_betti import JsonlMonitor
            run = output / f"runs/{family}/trial_{number:04d}"
            records = JsonlMonitor(run / "validation-metrics.jsonl").read_new()
            path = manual_stop_path(output, family, number)
            if path.exists():
                audit = json.loads(path.read_text())
                if (audit["family"] != family or audit["trial"] != number
                        or audit["requested_endpoint"] != endpoint or audit["state"] != "PRUNED"):
                    raise CampaignError("Existing manual-stop record has incompatible trial identity")
                if audit["reason"] != reason:
                    raise CampaignError("Refusing to change an existing manual-stop reason")
            else:
                audit = {"schema_version": 1, "family": family, "trial": number,
                         "state": "PRUNED", "previous_state": trial.state.name,
                         "reason": reason, "stopped_at": utc_now(),
                         "latest_validation_epoch": max((int(r["epoch"]) for r in records), default=None),
                         "parameters": trial.params or trial.system_attrs.get("fixed_params", {}),
                         "requested_endpoint": endpoint, "endpoint_comparison_eligible": False,
                         "slurm_job_id": trial.user_attrs.get("slurm_job_id"),
                         "checkpoint_retained": (run / "models/latest_checkpoint.pt").exists(),
                         "files_deleted": False}
            actions.append((trial, None, audit))
        for trial, values, audit in actions:
            if trial.state.is_finished():
                print(f"Trial {trial.number}: already {trial.state.name}; retained unchanged")
            elif values is not None:
                live = optuna.trial.Trial(study, trial._trial_id)
                live.set_user_attr("endpoint", endpoint)
                study.tell(trial.number, values=values)
                print(f"Trial {trial.number}: endpoint result already existed; finalized COMPLETE, not pruned")
            else:
                freeze(manual_stop_path(output, family, trial.number), audit)
                live = optuna.trial.Trial(study, trial._trial_id)
                live.set_user_attr("manual_stop", audit)
                study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
                print(f"Trial {trial.number}: PRUNED at last validation {audit['latest_validation_epoch']}; files retained")


def worker(args):
    import optuna
    saved = state(args.output)
    settings, family = saved["settings"], args.family
    endpoint = args.endpoint or settings["endpoints"][0]
    stage_reference(args.output, endpoint)
    if family == "h1" and not (args.output / "h1-plan.json").exists():
        raise CampaignError("Run h1-prepare before the optional ablation")
    study = study_for(args.output, family, int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if endpoint != settings["endpoints"][0]:
        plan = json.loads((args.output / "promotions" / family / f"epoch_{endpoint:04d}.json").read_text())
        for number in plan["trials"]:
            with trial_lock(args.output, family, number) as acquired:
                if acquired:
                    trial = next(t for t in study.trials if t.number == number)
                    incomplete = result_path(args.output, family, endpoint, number).with_suffix(".incomplete.json")
                    if incomplete.exists():
                        print(f"Trial {number}: early stopped; omitted from endpoint ranking")
                        continue
                    try:
                        evaluate_trial(args, family, trial, endpoint)
                    except CampaignError:
                        if incomplete.exists():
                            print(f"Trial {number}: early stopped; omitted from endpoint ranking")
                        else:
                            raise
        return
    h1_plan = json.loads((args.output / "h1-plan.json").read_text()) if family == "h1" else None
    total = settings["trials"] if family == "h0" else len(h1_plan["h1_weights"])
    # O_EXCL-free kernel locks are automatically released on Slurm termination.
    while True:
        chosen = None
        with AllocationLock(args.output / ".locks/ask.lock"):
            for item in study.trials:
                if item.state.name == "RUNNING":
                    if not args.resume:
                        continue
                    if manual_stop_path(args.output, family, item.number).exists():
                        # A durable stop record also prevents a restart if the
                        # manual finalizer was interrupted before study.tell.
                        continue
                    context = trial_lock(args.output, family, item.number)
                    if context.__enter__():
                        chosen = (optuna.trial.Trial(study, item._trial_id), context)
                        break
                    context.__exit__(None, None, None)
            if chosen is None:
                trials = study.trials
                if not any(t.state.name == "WAITING" for t in trials) and len(trials) >= total:
                    break
                trial = study.ask()
                stop_path = manual_stop_path(args.output, family, trial.number)
                if stop_path.exists():
                    trial.set_user_attr("manual_stop", json.loads(stop_path.read_text()))
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    continue
                if family == "h0":
                    trial.suggest_float("h0_weight", *settings["h0_range"], log=True)
                else:
                    trial.suggest_float("h0_weight", h1_plan["h0_weight"], h1_plan["h0_weight"])
                    trial.suggest_categorical("h1_weight", h1_plan["h1_weights"])
                context = trial_lock(args.output, family, trial.number)
                if not context.__enter__():
                    raise CampaignError("New trial is unexpectedly locked")
                chosen = trial, context
        trial, context = chosen
        try:
            trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
            trial.set_user_attr("resume_count", int(trial.user_attrs.get("resume_count", -1)) + 1)
            result = evaluate_trial(args, family, trial, endpoint)
        except KeyboardInterrupt:
            print(f"Trial {trial.number} interrupted; retained RUNNING for explicit resume", flush=True)
            raise
        except Exception as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {trial.number} FAILED: {error}", file=sys.stderr, flush=True)
        else:
            trial.set_user_attr("endpoint", endpoint)
            study.tell(trial, values=[result["metrics"][name] for name, _ in OBJECTIVES])
            print(f"Trial {trial.number} COMPLETE: {result['deltas']}", flush=True)
        finally:
            context.__exit__(None, None, None)


def promote(args):
    settings = state(args.output)["settings"]
    index = settings["endpoints"].index(args.endpoint)
    if index == 0:
        raise CampaignError("Cannot promote to the initial budget")
    numbers = [int(item) for item in args.trials.split(",")]
    if len(set(numbers)) != len(numbers):
        raise CampaignError("Duplicate promoted trials")
    required = settings["promotion_counts"][index - 1]
    if args.family == "h0" and len(numbers) != required:
        raise CampaignError(f"This H0 stage requires exactly {required} manually selected trials")
    if args.family == "h1" and not 1 <= len(numbers) <= len(json.loads((args.output / "h1-plan.json").read_text())["h1_weights"]):
        raise CampaignError("Invalid optional H1 promotion count")
    previous = settings["endpoints"][index - 1]
    for number in numbers:
        if not result_path(args.output, args.family, previous, number).is_file():
            raise CampaignError(f"Trial {number} has not completed the preceding budget")
    stage_reference(args.output, args.endpoint)
    freeze(args.output / "promotions" / args.family / f"epoch_{args.endpoint:04d}.json", {"from_endpoint": previous, "endpoint": args.endpoint, "trials": numbers})
    print(f"Frozen promotion to {args.endpoint}: {numbers}")


def dominates(left, right):
    signs = [1 if direction == "maximize" else -1 for _, direction in OBJECTIVES]
    pairs = [(sign * left[name], sign * right[name]) for (name, _), sign in zip(OBJECTIVES, signs)]
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def summary(args):
    settings = state(args.output)["settings"]
    if args.family == "h1" and not (args.output / "h1-plan.json").is_file():
        raise CampaignError("Run h1-prepare before requesting an H1 summary")
    for endpoint in settings["endpoints"]:
        directory = args.output / "results" / args.family / f"epoch_{endpoint:04d}"
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("trial_*.json")) if not p.name.endswith(".incomplete.json")]
        print(f"\nFAMILY={args.family} ENDPOINT={endpoint} complete={len(rows)}")
        parent = ablation_reference(args.output, endpoint) if args.family == "h1" and rows else None
        if parent:
            print(f"H0-only reference trial={parent['trial']} epochs={parent['aggregation']['contributing_epochs']}")
        flat = []
        for row in rows:
            pareto = not any(dominates(other["metrics"], row["metrics"]) for other in rows if other is not row)
            item = {"trial": row["trial"], "endpoint": endpoint, "pareto": pareto, **row["parameters"], **row["metrics"]}
            item["mean_train_seconds_per_epoch"] = row.get("mean_train_seconds_per_epoch")
            item.update({f"delta_{key}": value for key, value in row["deltas"].items()})
            item.update({f"control_{key}": value for key, value in row["control"].items()})
            item.update({f"percent_change_{key}": 100 * value / row["control"][key] if row["control"][key] else None for key, value in row["deltas"].items()})
            if parent:
                item["h0_reference_trial"] = parent["trial"]
                item.update({f"h0_reference_{key}": value for key, value in parent["metrics"].items()})
                item.update({f"delta_vs_h0_{key}": value - parent["metrics"][key]
                             for key, value in row["metrics"].items() if key in parent["metrics"]})
                item.update({f"percent_change_vs_h0_{key}": 100 * (value - parent["metrics"][key]) / parent["metrics"][key]
                             if parent["metrics"][key] else None
                             for key, value in row["metrics"].items() if key in parent["metrics"]})
            flat.append(item)
            print(f"trial={row['trial']:04d} params={row['parameters']} pareto={pareto} " + " ".join(f"d_{name}={row['deltas'][name]:+.6f}" for name, _ in OBJECTIVES))
            if parent:
                print("  versus H0-only: " + " ".join(
                    f"d_{name}={row['metrics'][name] - parent['metrics'][name]:+.6f}" for name, _ in OBJECTIVES))
        if flat:
            destination = args.output / "summaries" / args.family
            destination.mkdir(parents=True, exist_ok=True)
            keys = list(dict.fromkeys(key for row in flat for key in row))
            with (destination / f"epoch_{endpoint:04d}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flat)
        for path in sorted(directory.glob("trial_*.incomplete.json")):
            print("Excluded incomplete endpoint:", json.loads(path.read_text()))
    study = study_for(args.output, args.family)
    print("Study states:", {name: sum(t.state.name == name for t in study.trials) for name in ("WAITING", "RUNNING", "COMPLETE", "PRUNED", "FAIL")})
    for trial in study.trials:
        run = args.output / f"runs/{args.family}/trial_{trial.number:04d}"
        from scripts.optimize_node_edge_betti import JsonlMonitor
        records = JsonlMonitor(run / "validation-metrics.jsonl").read_new()
        print(f"trial={trial.number:04d} state={trial.state.name} latest_validation={records[-1]['epoch'] if records else None} checkpoint={(run / 'models/latest_checkpoint.pt').exists()} resumes={trial.user_attrs.get('resume_count', 0)} job={trial.user_attrs.get('slurm_job_id')} failure={trial.user_attrs.get('failure_reason', '-')}")
        if trial.user_attrs.get("manual_stop"):
            print("  manual_stop=" + trial.user_attrs["manual_stop"]["reason"])


def gradient_probe(output):
    """One small train batch, no optimizer updates; diagnose effective weights."""
    import math
    import random
    import numpy as np
    import torch
    from data.loaders import build_data_loaders
    from models import build_model
    from training.engine import _move_graph_batch
    from training.losses import build_criterion

    saved = state(output)
    config = concrete_config(output, "smoke", 0, {"h0_weight": .003, "h1_weight": .001}, saved["settings"]["endpoints"][0])
    config["runtime"].update(distributed=False, workers=0, device="cuda")
    config["data"].update(batch_size=2, validation_batch_size=2)
    config["data"]["datasets"]["synthetic_mri"].update(train_samples=32, validation_samples=16)
    random.seed(saved["settings"]["seed"])
    np.random.seed(saved["settings"]["seed"])
    torch.manual_seed(saved["settings"]["seed"])
    loader, _ = build_data_loaders(config, rank=0, world_size=1)
    model = build_model(config).cuda()
    model.load_state_dict(checkpoint(Path(saved["checkpoint"]))["net"])
    # Eval mode holds dropout/BN fixed while autograd remains active.
    model.eval()
    criterion = build_criterion(config, model).cuda()
    criterion.set_training_progress(saved["settings"]["branch_epoch"] + saved["settings"]["ramp_epochs"] + 1, 0.)
    prepared = _move_graph_batch(next(iter(loader)), torch.device("cuda"), config["training"]["input"], bool(config["loss"]["supervise_target_graphs"]))
    if prepared is None:
        raise CampaignError("Gradient probe has no supervised graphs")
    volumes, targets = prepared
    tokens, predictions, _ = model(volumes)
    losses = criterion(tokens, predictions, targets)
    terms = {
        "h0_weighted": losses["betti_h0_weighted"],
        "h1_weighted": losses["betti_h1_weighted"],
        "base_objective": losses["total"] - losses["betti_h0_weighted"] - losses["betti_h1_weighted"],
    }
    parameters = [p for p in model.parameters() if p.requires_grad]
    report = {"h0_weight": .003, "h1_weight": .001, "note": "Single fixed train batch; gradient norms are diagnostic, not stable dominance estimates."}
    for name, term in terms.items():
        gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        norm = sum(float(g.detach().double().square().sum().cpu()) for g in gradients if g is not None) ** .5
        value = float(term.detach().cpu())
        if not math.isfinite(norm) or not math.isfinite(value):
            raise CampaignError("Non-finite gradient probe")
        report[name] = {"loss": value, "parameter_gradient_l2": norm}
    atomic_json(output / "gpu-smoke/gradient-probe.json", report)
    print(json.dumps(report, indent=2), flush=True)


def smoke_config(output, number, label, parameters):
    settings = state(output)["settings"]
    config = concrete_config(output, "smoke", number, parameters, settings["branch_epoch"] + 3)
    config["experiment"]["name"] = f"gpu-smoke/v2/{label}"
    config["tracking"]["enabled"] = False
    config["training"]["early_stopping"]["enabled"] = False
    config["training"]["checkpoint"]["milestone_epochs"] = []
    config["evaluation"]["interval_epochs"] = 1
    # The metric loader deliberately ignores data.validation_samples. Cap its
    # independent selection too, or each smoke epoch evaluates the full split.
    config["evaluation"]["training_metrics"]["max_samples"] = 16
    config["data"]["datasets"]["synthetic_mri"].update(train_samples=32, validation_samples=16)
    for name in ("betti_h0", "betti_h1"):
        config["topology"][name]["ramp_epochs"] = 1
    validate_config(config)
    return config


def gpu_smoke(args):
    """Real DDP forward/backward and strict resume on a disposable tiny subset."""
    settings = state(args.output)["settings"]
    branch = settings["branch_epoch"]
    endpoint = branch + 3
    for number, (label, parameters) in enumerate((
        ("control", {"h0_weight": 0., "h1_weight": 0.}),
        ("h0", {"h0_weight": .003, "h1_weight": 0.}),
        ("h0-h1", {"h0_weight": .003, "h1_weight": .001}),
    )):
        config = smoke_config(args.output, number, label, parameters)
        run_name = config["experiment"]["name"]
        directory = args.output / run_name
        path = args.output / "gpu-smoke/v2" / f"{label}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        own = directory / "models/latest_checkpoint.pt"
        # Split the smoke into 1+2 epochs to exercise a strict scheduler resume.
        for stop in (branch + 1, endpoint):
            current = int(checkpoint(own)["epoch"]) if own.exists() else branch
            if current >= stop:
                continue
            config["training"]["stop_after_epoch"] = stop
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            code = run_command(training_command(args.output, path, run_name, own if own.exists() else Path(state(args.output)["checkpoint"]), not own.exists(), args.train_command), directory, poll=args.poll_interval)
            if code:
                raise CampaignError(f"GPU smoke {label} failed")
        if int(checkpoint(own)["epoch"]) != endpoint:
            raise CampaignError("GPU smoke checkpoint mismatch")
        aggregate_tail(_read_records(directory / "validation-metrics.jsonl"), 3)
        if not config["topology"]["betti_h1"]["enabled"]:
            performance = _read_records(directory / "performance.jsonl")
            if any(float(r["topology_betti_h1_selected_graphs"]) != 0 for r in performance):
                raise CampaignError("H1 unexpectedly active in H0-only GPU smoke")
    gradient_probe(args.output)
    atomic_json(args.output / "gpu-smoke/passed.json", {"passed": True, "smoke_version": 2, "gpus": settings["gpus"], "completed_at": utc_now(), "test_split_used": False})
    print("GPU smoke passed; this tiny subset is infrastructure validation only", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "promote", "summary", "h1-prepare", "gpu-smoke", "prune"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/experiments/focal_h0_study.yaml")
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--family", choices=("h0", "h1"), default="h0")
    parser.add_argument("--endpoint", type=int)
    parser.add_argument("--trials", help="Comma-separated manually selected trial IDs")
    parser.add_argument("--parent", type=int)
    parser.add_argument("--h1-weights", help="Comma-separated H1 grid, frozen separately from the H0 campaign")
    parser.add_argument("--reason", help="Audit reason for manually pruning stopped screening trials")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=.5)
    parser.add_argument("--train-command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        if args.action == "prepare":
            if not args.source_run or not args.reference_run:
                raise CampaignError("prepare requires --source-run and --reference-run")
            prepare(args)
            stage_reference(args.output, state(args.output)["settings"]["endpoints"][0])
            initialize_study(args.output)
        elif args.action == "worker":
            worker(args)
        elif args.action == "gpu-smoke":
            gpu_smoke(args)
        elif args.action == "promote":
            if args.endpoint is None or not args.trials:
                raise CampaignError("promote requires --endpoint and --trials")
            promote(args)
        elif args.action == "prune":
            prune_trials(args)
        elif args.action == "h1-prepare":
            if args.parent is None:
                raise CampaignError("h1-prepare requires --parent")
            ablation_prepare(args.output, args.parent,
                             None if args.h1_weights is None else args.h1_weights.split(","))
        else:
            summary(args)
    except CampaignError as error:
        parser.exit(1, f"H0 study error: {error}\n")


if __name__ == "__main__":
    main()
