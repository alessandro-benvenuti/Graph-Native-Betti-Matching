#!/usr/bin/env python3
"""Orchestrate control-relative staged Betti activation experiments.

The script intentionally keeps the shared prefix, three Optuna studies, and
selected final continuations independent.  No command reads the test split.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from configs import load_config, validate_config
from scripts.optimize_node_edge_betti import (
    AllocationLock,
    CampaignError,
    _read_records,
    _write_yaml,
    aggregate_tail,
    create_storage,
    deep_set,
    run_command,
    sampler_seed,
)
from training.checkpoint import load_runtime_state


ABSOLUTE_OBJECTIVES = (
    ("node_mAP", "maximize"),
    ("edge_mAP", "maximize"),
    ("beta0_absolute_error", "minimize"),
    ("beta1_absolute_error", "minimize"),
)
DELTA_METRIC_NAMES = {
    "node_mAP": "delta_node_mAP",
    "edge_mAP": "delta_edge_mAP",
    "beta0_absolute_error": "delta_beta0_error",
    "beta1_absolute_error": "delta_beta1_error",
}
DELTA_OBJECTIVES = tuple(
    (DELTA_METRIC_NAMES[name], direction) for name, direction in ABSOLUTE_OBJECTIVES
)
FINAL_METRICS = (
    "node_mAP", "edge_mAP", "node_f1", "edge_f1",
    "beta0_absolute_error", "beta1_absolute_error", "smd",
    "predicted_nodes", "target_nodes", "predicted_edges", "target_edges",
    "predicted_beta0", "target_beta0", "predicted_beta1", "target_beta1",
)
CHECKPOINT_KEYS = {
    "net", "optimizer", "scheduler", "scaler", "epoch", "iteration",
    "global_step", "runtime_states", "trainer_state", "training_config",
}
MARKER_NAME = ".staged-betti-activation.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["branch_epoch", "trial", "state"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True)
                if isinstance(row.get(key), (list, tuple, dict))
                else row.get(key)
                for key in keys
            })


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _merge_epoch_records(*histories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge metric histories, allowing later phases to own boundary epochs."""

    by_epoch: dict[int, dict[str, Any]] = {}
    for history in histories:
        for record in history:
            by_epoch[int(record["epoch"])] = dict(record)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _candidate_metric_history(args, staged, branch, source_run):
    """Build the node-focal -> Betti metric history for one selected trial."""

    endpoint = branch + staged["screening_epochs"]
    prefix_records = _read_records(
        args.output / "shared-prefix/validation-metrics.jsonl"
    )
    screening_records = _read_records(source_run / "validation-metrics.jsonl")
    prefix_phase = [
        {**record, "betti_active": 0.0, "betti_activation_epoch": branch}
        for record in prefix_records
        if int(record["epoch"]) <= branch
    ]
    betti_phase = [
        {**record, "betti_active": 1.0, "betti_activation_epoch": branch}
        for record in screening_records
        if branch < int(record["epoch"]) <= endpoint
    ]
    history = _merge_epoch_records(prefix_phase, betti_phase)
    if not history or int(history[-1]["epoch"]) != endpoint:
        raise CampaignError(
            f"candidate history does not reach screening endpoint {endpoint}"
        )
    return history


def _prefix_metric_history(args, branch):
    prefix_records = _read_records(
        args.output / "shared-prefix/validation-metrics.jsonl"
    )
    history = [
        {**record, "betti_active": 0.0, "betti_activation_epoch": branch}
        for record in prefix_records
        if int(record["epoch"]) <= branch
    ]
    if not history or int(history[-1]["epoch"]) != branch:
        raise CampaignError(f"shared prefix metric history does not reach epoch {branch}")
    return history


def load_staged(path: Path, environment=None):
    resolved = load_config(path, environment=environment)
    staged = resolved.pop("staged_betti", None)
    if not isinstance(staged, dict):
        raise CampaignError("configuration requires staged_betti")
    validate_config(resolved)
    branches = staged.get("branch_epochs")
    if not isinstance(branches, list) or not branches or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in branches
    ):
        raise CampaignError("staged_betti.branch_epochs must be positive integers")
    if branches != sorted(set(branches)):
        raise CampaignError("staged_betti.branch_epochs must be sorted and unique")
    screening_epochs = staged.get("screening_epochs")
    if not isinstance(screening_epochs, int) or screening_epochs <= 0:
        raise CampaignError("staged_betti.screening_epochs must be positive")
    if max(branches) + screening_epochs > int(resolved["training"]["epochs"]):
        raise CampaignError("screening endpoint exceeds training.epochs")
    trials = staged.get("trials_per_branch")
    if not isinstance(trials, int) or trials <= 0:
        raise CampaignError("staged_betti.trials_per_branch must be positive")
    aggregation = staged.get("metric_aggregation", {})
    if aggregation.get("name") != "tail_mean" or not isinstance(
        aggregation.get("observations"), int
    ) or aggregation["observations"] <= 0:
        raise CampaignError("staged metric aggregation must be a positive tail_mean")
    sampler = staged.get("sampler", {})
    if sampler.get("name") != "nsga2" or int(sampler.get("population_size", 0)) < 2:
        raise CampaignError("staged sampler must be NSGA-II with population >= 2")
    space = staged.get("search_space", {})
    expected = {
        "topology.betti_h0.weight",
        "topology.betti_h1.weight",
        "topology.betti_h1.false_positive_weight",
        "betti_ramp_epochs",
    }
    if set(space) != expected:
        raise CampaignError(
            "staged search space must contain only: " + ", ".join(sorted(expected))
        )
    if any(not isinstance(space[name], list) or not space[name] for name in expected):
        raise CampaignError("every staged search-space entry must have choices")
    if resolved["model"]["matcher"]["type"] != "hungarian":
        raise CampaignError("staged Betti requires Hungarian matching")
    if resolved["loss"]["node"]["classification"]["name"] != "focal":
        raise CampaignError("staged Betti fixes node loss to focal")
    if resolved["loss"]["edge"]["classification"]["name"] != "cross_entropy":
        raise CampaignError("staged Betti fixes edge loss to cross-entropy")
    if float(resolved["topology"]["complex"]["alpha"]) != 0.5:
        raise CampaignError("staged Betti fixes node-edge filtration alpha to 0.5")
    staged_gpus = int(os.environ.get("GNBM_STAGED_GPUS", "1"))
    global_batch_size = int(
        os.environ.get(
            "GNBM_STAGED_GLOBAL_BATCH_SIZE",
            str(resolved["data"]["batch_size"]),
        )
    )
    if staged_gpus not in {1, 2, 4}:
        raise CampaignError("GNBM_STAGED_GPUS must be 1, 2, or 4")
    if global_batch_size <= 0 or global_batch_size % staged_gpus:
        raise CampaignError(
            "GNBM_STAGED_GLOBAL_BATCH_SIZE must be positive and divisible by GPUs"
        )
    resolved["data"]["batch_size"] = global_batch_size // staged_gpus
    resolved["runtime"]["distributed"] = staged_gpus > 1
    validate_config(resolved)
    return resolved, staged


def _campaign_marker(output: Path, initial_weights: Path, config_path: Path, *, create=False):
    marker = output / MARKER_NAME
    if marker.is_file():
        payload = json.loads(marker.read_text())
        if payload.get("initial_weights") != str(initial_weights.resolve()):
            raise CampaignError("campaign initialization checkpoint changed")
        return payload
    if not create:
        raise CampaignError("staged campaign is not initialized; run prefix first")
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [item for item in output.iterdir() if item.name != MARKER_NAME]
    if unexpected:
        raise CampaignError(
            "refusing to reuse a non-empty output without a staged marker: "
            + str(output)
        )
    if not initial_weights.is_file():
        raise CampaignError("initial weights do not exist: " + str(initial_weights))
    payload = {
        "schema_version": 1,
        "campaign": "staged-betti-activation",
        "created_at": utc_now(),
        "initial_weights": str(initial_weights.resolve()),
        "initial_weights_sha256": _sha256(initial_weights),
        "config": str(config_path.resolve()),
        "execution": {
            "gpus": int(os.environ.get("GNBM_STAGED_GPUS", "1")),
            "global_batch_size": int(
                os.environ.get("GNBM_STAGED_GLOBAL_BATCH_SIZE", "32")
            ),
        },
        "test_split_used": False,
    }
    _atomic_json(marker, payload)
    return payload


def _checkpoint(path: Path, expected_epoch: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError("checkpoint is missing: " + str(path))
    try:
        payload = load_runtime_state(path)
    except Exception as error:
        raise CampaignError(f"cannot read full-state checkpoint {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CampaignError("checkpoint payload is not a mapping: " + str(path))
    missing = sorted(CHECKPOINT_KEYS - set(payload))
    if missing:
        raise CampaignError(
            f"checkpoint {path} is not fully resumable; missing: {', '.join(missing)}"
        )
    if expected_epoch is not None and int(payload["epoch"]) != int(expected_epoch):
        raise CampaignError(
            f"checkpoint {path} stores epoch {payload['epoch']}, expected {expected_epoch}"
        )
    if not payload["runtime_states"]:
        raise CampaignError("checkpoint has no RNG/runtime state: " + str(path))
    return dict(payload)


def _prefix_config(base, staged):
    config = copy.deepcopy(base)
    config["experiment"]["name"] = "shared-prefix"
    config["training"]["stop_after_epoch"] = None
    config["training"]["checkpoint"]["milestone_epochs"] = [
        *staged["branch_epochs"], int(config["training"]["epochs"])
    ]
    config["tracking"]["group"] = "staged-betti-activation"
    config["tracking"]["run_name"] = "shared-prefix-control"
    for name in ("betti_h0", "betti_h1"):
        config["topology"][name].update(
            enabled=False, log_only=True, weight=0.0, activation_epoch=None
        )
    validate_config(config)
    return config


def _command(args, config_path: Path, run_name: str, *, resume=None, initial=None):
    if args.train_command:
        command = [
            part.format(
                config=config_path,
                output=args.output,
                run_name=run_name,
                initial_weights=initial or "",
                resume_checkpoint=resume or "",
            )
            for part in args.train_command
        ]
    else:
        staged_gpus = int(os.environ.get("GNBM_STAGED_GPUS", "1"))
        if staged_gpus > 1:
            command = [
                sys.executable, "-u", "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1",
                "--nproc_per_node={}".format(staged_gpus),
                "train.py", "--distributed",
            ]
        else:
            command = [sys.executable, "-u", "train.py"]
        command.extend([
            "--config", str(config_path),
            "--output-dir", str(args.output), "--run-name", run_name,
        ])
    if resume is not None:
        command.extend(("--resume", str(resume)))
    elif initial is not None:
        command.extend(("--initial-weights", str(initial)))
    else:
        raise CampaignError("training command needs a resume or initialization checkpoint")
    return command


def _run_prefix_unlocked(args, base, staged):
    run_name = "shared-prefix"
    run_dir = args.output / run_name
    config_path = args.output / "configs/shared-prefix.yaml"
    config = _prefix_config(base, staged)
    _write_yaml(config_path, config)
    latest = run_dir / "models/latest_checkpoint.pt"
    complete = run_dir / ".complete.json"
    if complete.is_file():
        print("shared prefix already complete")
        return
    if run_dir.exists() and not latest.is_file():
        raise CampaignError("shared prefix exists but has no resumable latest checkpoint")
    resume = latest if latest.is_file() else None
    if resume is not None:
        checkpoint = _checkpoint(resume)
        stored_world_size = int(
            checkpoint.get("trainer_state", {}).get("world_size", 1)
        )
        requested_world_size = int(os.environ.get("GNBM_STAGED_GPUS", "1"))
        if stored_world_size != requested_world_size:
            raise CampaignError(
                "shared prefix checkpoint world_size={} cannot resume with {} GPUs; "
                "use a fresh output directory".format(
                    stored_world_size, requested_world_size
                )
            )
    command = _command(
        args, config_path, run_name, resume=resume,
        initial=None if resume else args.initial_weights,
    )
    code = run_command(command, run_dir, poll=args.poll_interval)
    if code != 0:
        raise CampaignError(f"shared prefix exited with status {code}")
    final_epoch = int(config["training"]["epochs"])
    for epoch in (*staged["branch_epochs"], final_epoch):
        _checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", epoch)
    history = _read_records(run_dir / "validation-metrics.jsonl")
    if not history or int(history[-1]["epoch"]) != final_epoch:
        raise CampaignError("shared prefix has no final validation record")
    _atomic_json(complete, {
        "completed_at": utc_now(), "epoch": final_epoch,
        "checkpoint": str((run_dir / "checkpoints" / f"epoch_{final_epoch:04d}.pt").resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(f"shared prefix COMPLETE at epoch {final_epoch}")


def run_prefix(args, base, staged):
    _campaign_marker(args.output, args.initial_weights, args.config, create=True)
    with AllocationLock(args.output / ".prefix.lock"):
        _run_prefix_unlocked(args, base, staged)


def _records_through(records, endpoint):
    selected = [record for record in records if int(record["epoch"]) <= endpoint]
    if not selected or int(selected[-1]["epoch"]) != endpoint:
        raise CampaignError(f"shared prefix has no validation record at epoch {endpoint}")
    return selected


def _prepare_controls_unlocked(args, base, staged, branch=None):
    prefix = args.output / "shared-prefix"
    if branch is None and not (prefix / ".complete.json").is_file():
        raise CampaignError("shared prefix is not complete")
    if branch is not None and branch not in staged["branch_epochs"]:
        raise CampaignError("branch is not configured: " + str(branch))
    history = _read_records(prefix / "validation-metrics.jsonl")
    manifest = prefix / "dataset-manifest.json"
    if not manifest.is_file():
        raise CampaignError("shared prefix dataset manifest is missing")
    manifest_sha = _sha256(manifest)
    observations = staged["metric_aggregation"]["observations"]
    expected_prefix = _prefix_config(base, staged)
    previous_iteration = -1
    optuna = _import_optuna()
    branches = staged["branch_epochs"] if branch is None else [branch]
    for branch_epoch in branches:
        snapshot = prefix / "checkpoints" / f"epoch_{branch_epoch:04d}.pt"
        payload = _checkpoint(snapshot, branch_epoch)
        stored = payload["training_config"]
        for key in ("model", "data", "loss"):
            if stored.get(key) != expected_prefix.get(key):
                raise CampaignError(
                    f"epoch-{branch_epoch} checkpoint has incompatible {key} configuration"
                )
        if stored.get("training", {}).get("optimizer") != expected_prefix["training"]["optimizer"]:
            raise CampaignError(f"epoch-{branch_epoch} checkpoint optimizer configuration changed")
        if stored.get("training", {}).get("scheduler") != expected_prefix["training"]["scheduler"]:
            raise CampaignError(f"epoch-{branch_epoch} checkpoint scheduler configuration changed")
        if int(payload["iteration"]) <= previous_iteration:
            raise CampaignError("shared-prefix checkpoint iterations are not increasing")
        previous_iteration = int(payload["iteration"])
        endpoint = branch_epoch + staged["screening_epochs"]
        aggregation = aggregate_tail(_records_through(history, endpoint), observations)
        directory = args.output / "screening" / f"branch_{branch_epoch:04d}"
        reference = directory / "control-reference.json"
        frozen = {
            "schema_version": 1,
            "branch_epoch": branch_epoch,
            "screening_endpoint": endpoint,
            "branch_checkpoint": str(snapshot.resolve()),
            "branch_checkpoint_sha256": _sha256(snapshot),
            "branch_iteration": int(payload["iteration"]),
            "control_run": str(prefix.resolve()),
            "aggregation": aggregation,
            "metrics": aggregation["metrics"],
            "dataset_manifest_sha256": manifest_sha,
            "test_split_used": False,
        }
        # Compare the persisted JSON representation, not Python container
        # identities.  aggregate_tail intentionally returns its objective
        # vector as a tuple, while JSON reloads that value as a list.
        frozen = json.loads(json.dumps(frozen, sort_keys=True))
        if reference.is_file() and json.loads(reference.read_text()) != frozen:
            raise CampaignError("refusing to change frozen branch control: " + str(reference))
        _atomic_json(reference, frozen)
        study = _study(optuna, directory, branch_epoch, staged)
        if not study.user_attrs.get("noise_duplicates_enqueued"):
            anchor_trials = [
                trial for trial in study.trials
                if trial.user_attrs.get("noise_duplicate_group") == "anchor"
            ]
            replicates = {
                trial.user_attrs.get("noise_duplicate_replicate")
                for trial in anchor_trials
            }
            if anchor_trials and (
                len(anchor_trials) != 2 or replicates != {0, 1}
            ):
                raise CampaignError(
                    "branch has an incomplete or invalid anchor duplicate pair"
                )
            if study.trials and not anchor_trials:
                raise CampaignError(
                    "cannot enqueue noise duplicates after branch trials exist"
                )
            anchor = {
                name: choices[(len(choices) - 1) // 2]
                for name, choices in staged["search_space"].items()
            }
            if not anchor_trials:
                for replicate in (0, 1):
                    study.enqueue_trial(
                        anchor,
                        user_attrs={
                            "noise_duplicate_group": "anchor",
                            "noise_duplicate_replicate": replicate,
                        },
                    )
            study.set_user_attr("noise_duplicates_enqueued", True)
        print(f"branch={branch_epoch} control_endpoint={endpoint} checkpoint=OK")


def prepare_controls(args, base, staged):
    _campaign_marker(args.output, args.initial_weights, args.config)
    with AllocationLock(args.output / ".prepare.lock"):
        _prepare_controls_unlocked(args, base, staged, branch=args.branch)


def _import_optuna():
    try:
        import optuna
    except ImportError as error:
        raise CampaignError("install requirements/optuna.txt") from error
    return optuna


def _study_name(branch: int) -> str:
    return f"staged-betti-activation-e{branch:04d}"


def _branch_dir(output: Path, branch: int) -> Path:
    return output / "screening" / f"branch_{branch:04d}"


def _sampler(optuna, staged):
    sampler_config = {
        "seed": int(staged["sampler"]["seed"]),
        "sampler": staged["sampler"],
    }
    return optuna.samplers.NSGAIISampler(
        seed=sampler_seed(sampler_config),
        population_size=int(staged["sampler"]["population_size"]),
    )


def _study(optuna, directory: Path, branch: int, staged):
    study = optuna.create_study(
        study_name=_study_name(branch),
        storage=create_storage(optuna, directory, "journal"),
        directions=[direction for _, direction in DELTA_OBJECTIVES],
        sampler=_sampler(optuna, staged),
        load_if_exists=True,
    )
    study.set_user_attr("absolute_metrics", list(ABSOLUTE_OBJECTIVES))
    study.set_user_attr("objectives", list(DELTA_OBJECTIVES))
    study.set_user_attr("branch_epoch", branch)
    study.set_user_attr("screening_epochs", staged["screening_epochs"])
    study.set_user_attr("requested_trials", staged["trials_per_branch"])
    return study


def _worker_generation():
    return (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or f"local-{os.getpid()}"
    )


def _allocate(optuna, study, maximum, lock, resume_running):
    with AllocationLock(lock):
        generation = _worker_generation()
        if resume_running:
            running = [
                trial for trial in study.get_trials(deepcopy=False)
                if trial.state == optuna.trial.TrialState.RUNNING
            ]
            stale = [
                trial for trial in running
                if trial.user_attrs.get("worker_generation") != generation
            ]
            if stale:
                frozen = min(stale, key=lambda item: item.number)
                trial = optuna.trial.Trial(study, frozen._trial_id)
                trial.set_user_attr("worker_generation", generation)
                return trial, True
            # A sibling in this resume generation is already recovering every
            # remaining RUNNING trial.  Do not allocate fresh work until those
            # identities become terminal.
            if running:
                return None
        trials = study.get_trials(deepcopy=False)
        waiting = any(
            trial.state == optuna.trial.TrialState.WAITING for trial in trials
        )
        if not waiting and len(trials) >= maximum:
            return None
        trial = study.ask()
        trial.set_user_attr("worker_generation", generation)
        return trial, False


def _trial_config(base, staged, branch, number, params, reference):
    config = copy.deepcopy(base)
    endpoint = branch + staged["screening_epochs"]
    run_name = f"screening/branch_{branch:04d}/runs/trial_{number:04d}"
    config["experiment"]["name"] = run_name
    config["tracking"]["run_name"] = f"branch-e{branch}-trial-{number:03d}"
    config["training"]["stop_after_epoch"] = endpoint
    config["training"]["checkpoint"]["milestone_epochs"] = []
    for name, value in params.items():
        if name == "betti_ramp_epochs":
            for topology_name in ("betti_h0", "betti_h1"):
                deep_set(config, f"topology.{topology_name}.ramp_epochs", value)
        else:
            deep_set(config, name, value)
    for topology_name in ("betti_h0", "betti_h1"):
        config["topology"][topology_name]["warmup_epochs"] = 0
        config["topology"][topology_name]["activation_epoch"] = branch
    config["staged_metadata"] = {
        "branch_epoch": branch,
        "starting_checkpoint": reference["branch_checkpoint"],
        "screening_endpoint": endpoint,
        "optuna_trial_number": number,
        "control_metrics": reference["metrics"],
        "resume_count": 0,
    }
    validate_config(config)
    return config


def _deltas(metrics, control):
    return {
        DELTA_METRIC_NAMES[name]: float(metrics[name]) - float(control[name])
        for name, _ in ABSOLUTE_OBJECTIVES
    }


def _execute_trial(args, base, staged, branch, trial, reference, *, resuming):
    directory = _branch_dir(args.output, branch)
    relative_name = f"screening/branch_{branch:04d}/runs/trial_{trial.number:04d}"
    run_dir = args.output / relative_name
    config_path = directory / "trial-configs" / f"trial_{trial.number:04d}.yaml"
    bootstrap_path = (
        directory / "metric-history" / f"trial_{trial.number:04d}.jsonl"
    )
    if resuming:
        if not config_path.is_file() or not run_dir.is_dir():
            raise CampaignError(f"trial {trial.number} has no resumable config/run")
        checkpoint = run_dir / "models/latest_checkpoint.pt"
        _checkpoint(checkpoint)
        resume_count = int(trial.user_attrs.get("resume_count", 0)) + 1
        trial.set_user_attr("resume_count", resume_count)
        trial.set_user_attr("resumed_from", str(checkpoint.resolve()))
        config = yaml.safe_load(config_path.read_text())
        config.setdefault("staged_metadata", {})["resume_count"] = resume_count
    else:
        params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in staged["search_space"].items()
        }
        if run_dir.exists():
            raise CampaignError("refusing to reuse trial directory: " + str(run_dir))
        config = _trial_config(base, staged, branch, trial.number, params, reference)
        checkpoint = Path(reference["branch_checkpoint"])
        trial.set_user_attr("run_dir", str(run_dir.resolve()))
        trial.set_user_attr("config_path", str(config_path.resolve()))
        trial.set_user_attr("branch_checkpoint", str(checkpoint.resolve()))
        trial.set_user_attr("resume_count", 0)
    _write_jsonl(bootstrap_path, _prefix_metric_history(args, branch))
    config["tracking"]["bootstrap_metrics_path"] = str(bootstrap_path.resolve())
    _write_yaml(config_path, config)
    trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
    trial.set_user_attr("sampler_seed", int(staged["sampler"]["seed"]) + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    trial.set_user_attr("started_at", utc_now())
    started = time.monotonic()
    command = _command(args, config_path, relative_name, resume=checkpoint)
    code = run_command(command, run_dir, poll=args.poll_interval)
    if code != 0:
        raise CampaignError(f"training exited with status {code}")
    records = _read_records(run_dir / "validation-metrics.jsonl")
    aggregation = aggregate_tail(
        records, staged["metric_aggregation"]["observations"]
    )
    endpoint = branch + staged["screening_epochs"]
    if aggregation["final_epoch"] != endpoint:
        raise CampaignError(
            f"trial ended at epoch {aggregation['final_epoch']}, expected {endpoint}"
        )
    screening_history = [
        {
            **record,
            "betti_active": 1.0,
            "betti_activation_epoch": branch,
        }
        for record in records
    ]
    _write_jsonl(
        run_dir / "stitched-validation-metrics.jsonl",
        _merge_epoch_records(_prefix_metric_history(args, branch), screening_history),
    )
    manifest = run_dir / "dataset-manifest.json"
    if not manifest.is_file() or _sha256(manifest) != reference["dataset_manifest_sha256"]:
        raise CampaignError("trial dataset manifest differs from shared control")
    latest = run_dir / "models/latest_checkpoint.pt"
    _checkpoint(latest, endpoint)
    deltas = _deltas(aggregation["metrics"], reference["metrics"])
    trial.set_user_attr("aggregation", aggregation)
    trial.set_user_attr("absolute_metrics", aggregation["metrics"])
    trial.set_user_attr("control_metrics", reference["metrics"])
    trial.set_user_attr("control_deltas", deltas)
    trial.set_user_attr("final_epoch", endpoint)
    trial.set_user_attr("duration_seconds", time.monotonic() - started)
    trial.set_user_attr("training_completed", True)
    trial.set_user_attr("finished_at", utc_now())
    return tuple(deltas[name] for name, _ in DELTA_OBJECTIVES)


def run_screen_worker(args, base, staged):
    if args.branch not in staged["branch_epochs"]:
        raise CampaignError("branch is not configured: " + str(args.branch))
    _campaign_marker(args.output, args.initial_weights, args.config)
    directory = _branch_dir(args.output, args.branch)
    reference_path = directory / "control-reference.json"
    if not reference_path.is_file():
        raise CampaignError("branch control is missing; run prepare")
    reference = json.loads(reference_path.read_text())
    optuna = _import_optuna()
    study = _study(optuna, directory, args.branch, staged)
    attempts = 0
    maximum = int(staged["trials_per_branch"])
    while args.worker_trials <= 0 or attempts < args.worker_trials:
        allocation = _allocate(
            optuna, study, maximum, directory / ".trial-allocation.lock",
            args.resume_running,
        )
        if allocation is None:
            break
        trial, resuming = allocation
        attempts += 1
        try:
            values = _execute_trial(
                args, base, staged, args.branch, trial, reference,
                resuming=resuming,
            )
        except KeyboardInterrupt:
            trial.set_user_attr("failure_reason", "interrupted by SIGTERM/SIGINT")
            trial.set_user_attr("training_completed", False)
            trial.set_user_attr("interrupted_at", utc_now())
            raise
        except Exception as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            trial.set_user_attr("training_completed", False)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {trial.number} failed: {error}", file=sys.stderr)
        else:
            study.tell(trial, values=values)
            _atomic_json(Path(trial.user_attrs["run_dir"]) / ".complete.json", {
                "completed_at": utc_now(), "trial": trial.number,
                "branch_epoch": args.branch,
                "epoch": trial.user_attrs["final_epoch"],
            })
            print(f"Trial {trial.number} COMPLETE deltas={values}")
    print(f"branch={args.branch} attempts={attempts} trials={len(study.trials)}")


def _dominates(left, right):
    no_worse = True
    better = False
    for name, direction in DELTA_OBJECTIVES:
        a, b = float(left[name]), float(right[name])
        if direction == "maximize":
            no_worse &= a >= b
            better |= a > b
        else:
            no_worse &= a <= b
            better |= a < b
    return bool(no_worse and better)


def _study_rows(study, branch):
    rows = []
    for trial in sorted(study.trials, key=lambda item: item.number):
        attrs = trial.user_attrs
        absolute = attrs.get("absolute_metrics", {})
        deltas = attrs.get("control_deltas", {})
        row = {
            "branch_epoch": branch, "trial": trial.number,
            "state": trial.state.name, "pareto": False,
            "final_epoch": attrs.get("final_epoch"),
            "resume_count": attrs.get("resume_count", 0),
            "checkpoint_available": bool(
                attrs.get("run_dir")
                and (Path(attrs["run_dir"]) / "models/latest_checkpoint.pt").is_file()
            ),
            "config_path": attrs.get("config_path"), "run_dir": attrs.get("run_dir"),
            "failure_reason": attrs.get("failure_reason"),
            "noise_duplicate_group": attrs.get("noise_duplicate_group"),
            "noise_duplicate_replicate": attrs.get("noise_duplicate_replicate"),
            **absolute, **deltas, **trial.params,
        }
        rows.append(row)
    complete = [
        row for row in rows if row["state"] == "COMPLETE"
        and all(row.get(name) is not None for name, _ in DELTA_OBJECTIVES)
    ]
    for row in complete:
        row["pareto"] = not any(
            _dominates(other, row) for other in complete if other is not row
        )
    return rows


def summarize(args, base, staged):
    _campaign_marker(args.output, args.initial_weights, args.config)
    optuna = _import_optuna()
    all_rows = []
    for branch in staged["branch_epochs"]:
        directory = _branch_dir(args.output, branch)
        if not (directory / "study.journal").exists():
            continue
        study = _study(optuna, directory, branch, staged)
        all_rows.extend(_study_rows(study, branch))
    noise_ranges = {}
    for branch in staged["branch_epochs"]:
        duplicates = [
            row for row in all_rows
            if row["branch_epoch"] == branch
            and row["state"] == "COMPLETE"
            and row.get("noise_duplicate_group") == "anchor"
        ]
        noise_ranges[branch] = {}
        for name, _ in DELTA_OBJECTIVES:
            values = [float(row[name]) for row in duplicates if row.get(name) is not None]
            noise_ranges[branch][name] = (
                max(values) - min(values) if len(values) >= 2 else None
            )
        for row in [item for item in all_rows if item["branch_epoch"] == branch]:
            for name, _ in DELTA_OBJECTIVES:
                noise = noise_ranges[branch][name]
                row[name + "_vs_noise"] = (
                    "not_estimated" if noise is None or row.get(name) is None
                    else "inconclusive" if abs(float(row[name])) <= noise
                    else "larger_than_duplicate_noise"
                )
    summaries = args.output / "summaries"
    pareto = [row for row in all_rows if row["pareto"]]
    suggestions = {}
    for branch in staged["branch_epochs"]:
        front = [row for row in pareto if row["branch_epoch"] == branch]
        if not front:
            suggestions[branch] = {"balanced": None, "h1_oriented": None}
            continue
        ranges = {
            name: (
                min(float(row[name]) for row in front),
                max(float(row[name]) for row in front),
            )
            for name, _ in DELTA_OBJECTIVES
        }
        def distance(row):
            total = 0.0
            for name, direction in DELTA_OBJECTIVES:
                low, high = ranges[name]
                if low == high:
                    score = 1.0
                elif direction == "maximize":
                    score = (float(row[name]) - low) / (high - low)
                else:
                    score = (high - float(row[name])) / (high - low)
                total += (1.0 - score) ** 2
            return math.sqrt(total), row["trial"]
        suggestions[branch] = {
            "balanced": min(front, key=distance)["trial"],
            "h1_oriented": min(
                front,
                key=lambda row: (float(row["delta_beta1_error"]), row["trial"]),
            )["trial"],
            "note": "descriptive suggestions; predictive trade-offs require manual review",
        }
    _atomic_json(summaries / "candidate-suggestions.json", suggestions)
    _write_csv(summaries / "control-relative-comparison.csv", all_rows)
    _write_csv(summaries / "screening-pareto.csv", pareto)
    final_rows = _final_rows(args, staged)
    _write_csv(summaries / "final-epoch-500-comparison.csv", final_rows)
    complete = sum(row["state"] == "COMPLETE" for row in all_rows)
    strict_useful = sorted({
        int(row["branch_epoch"]) for row in final_rows
        if float(row.get("delta_node_mAP", -math.inf)) >= 0.0
        and float(row.get("delta_edge_mAP", -math.inf)) >= 0.0
        and (
            float(row.get("delta_beta0_error", math.inf)) < 0.0
            or float(row.get("delta_beta1_error", math.inf)) < 0.0
        )
    })
    topology_survives = [
        row for row in final_rows
        if float(row.get("delta_beta0_error", math.inf)) < 0.0
        or float(row.get("delta_beta1_error", math.inf)) < 0.0
    ]
    useful_answer = (
        "pending final continuations"
        if not final_rows
        else "strict evidence at branch epoch(s) " + ", ".join(map(str, strict_useful))
        if strict_useful
        else "no selected continuation improved topology while preserving both mAP metrics"
    )
    persistence_answer = (
        "pending"
        if not final_rows
        else "yes for " + ", ".join(
            f"branch {row['branch_epoch']} trial {row['trial']}" for row in topology_survives
        )
        if topology_survives
        else "no topology improvement survived in the completed selections"
    )
    report = [
        "# Staged Betti activation report", "",
        f"Generated: {utc_now()}", "",
        "## Screening status", "",
        f"- Complete trials: {complete}/{len(staged['branch_epochs']) * staged['trials_per_branch']}",
        f"- Pareto trials: {len(pareto)}", "",
        "Cross-branch Pareto objectives are matched-control deltas: node/edge mAP are maximized; beta0/beta1 error deltas are minimized.", "",
        "## Scientific questions", "",
        f"1. Useful activation stage: {useful_answer}.",
        "2. Best H0/H1/FP trade-off: the Pareto front has no unique scalar winner; use the reviewed balanced and H1-oriented selections below.",
        f"3. Persistence through epoch 500: {persistence_answer}.",
        "4. Edge suppression check: each final row reports predicted/target edge counts and their matched-control deltas; topology gains accompanied only by a large negative predicted-edge delta require caution.",
        "5. Run-to-run noise: every screening metric is labelled against the duplicated anchor configuration; effects within that range are `inconclusive`.", "",
        "## Duplicate-run noise ranges", "",
    ]
    for branch in staged["branch_epochs"]:
        report.append(f"- branch {branch}: {json.dumps(noise_ranges[branch], sort_keys=True)}")
    report.append("")
    report.extend(["## Descriptive candidate suggestions", ""])
    for branch in staged["branch_epochs"]:
        report.append(f"- branch {branch}: {json.dumps(suggestions[branch], sort_keys=True)}")
    report.extend([
        "",
        "Suggestions do not authorize continuation: inspect predictive deltas and duplicate-run noise before selecting at most two candidates per branch.",
        "",
        "## Fixed loss behavior", "",
        "H1 `matched_mean` sums false-positive cycle terms, so its magnitude depends on the number of false cycles. This campaign does not alter that normalization.",
        "",
    ])
    if final_rows:
        report.extend(["## Final continuations", ""])
        for row in final_rows:
            report.append(
                f"- branch {row['branch_epoch']}, trial {row['trial']}: "
                f"{row['betti_supervised_epochs']} Betti-supervised epochs"
            )
    (summaries / "report.md").write_text("\n".join(report) + "\n")
    is_smoke = staged["branch_epochs"] == [2, 4, 6]
    if is_smoke:
        expected = len(staged["branch_epochs"]) * staged["trials_per_branch"]
        resumed = [row for row in all_rows if int(row.get("resume_count", 0)) > 0]
        smoke_validation = {
            "passed": complete == expected and bool(resumed) and bool(final_rows),
            "complete_trials": complete,
            "expected_trials": expected,
            "resumed_trials": [
                {"branch_epoch": row["branch_epoch"], "trial": row["trial"]}
                for row in resumed
            ],
            "final_continuations": len(final_rows),
            "wandb_mode": base["tracking"].get("mode"),
            "validated_at": utc_now(),
        }
        _atomic_json(summaries / "smoke-validation.json", smoke_validation)
        print("smoke_validation_passed=" + str(smoke_validation["passed"]))
    print(f"screening complete={complete} pareto={len(pareto)} finals={len(final_rows)}")


def _trial_by_number(args, staged, branch, number):
    optuna = _import_optuna()
    study = _study(optuna, _branch_dir(args.output, branch), branch, staged)
    matches = [trial for trial in study.trials if trial.number == number]
    if not matches or matches[0].state.name != "COMPLETE":
        raise CampaignError(f"branch {branch} trial {number} is not COMPLETE")
    return matches[0]


def _run_final_unlocked(args, base, staged):
    if args.branch not in staged["branch_epochs"]:
        raise CampaignError("invalid final branch")
    trial = _trial_by_number(args, staged, args.branch, args.trial)
    source_config = Path(trial.user_attrs["config_path"])
    source_run = Path(trial.user_attrs["run_dir"])
    name = f"branch-e{args.branch}-final-{args.trial:04d}"
    relative_name = "final-continuations/" + name
    run_dir = args.output / relative_name
    config_path = args.output / "final-continuations/configs" / (name + ".yaml")
    bootstrap_path = (
        args.output / "final-continuations/metric-history" / (name + ".jsonl")
    )
    complete = run_dir / ".complete.json"
    if complete.is_file():
        print(name + " already complete")
        return
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text())
    else:
        config = yaml.safe_load(source_config.read_text())
        config["experiment"]["name"] = relative_name
        config["tracking"]["run_name"] = f"branch-e{args.branch}-final-{args.trial}"
        config["training"]["stop_after_epoch"] = None
        prefix_records = _read_records(
            args.output / "shared-prefix/validation-metrics.jsonl"
        )
        final_epoch = int(config["training"]["epochs"])
        final_controls = [
            record for record in prefix_records
            if int(record["epoch"]) == final_epoch
        ]
        if len(final_controls) != 1:
            raise CampaignError("shared prefix has no unique final control metrics")
        config["staged_metadata"].update({
            "final_continuation": True,
            "source_screening_trial": args.trial,
            "betti_supervised_epochs": int(config["training"]["epochs"]) - args.branch,
            "control_metrics": final_controls[0],
        })
    candidate_history = _candidate_metric_history(
        args, staged, args.branch, source_run
    )
    _write_jsonl(bootstrap_path, candidate_history)
    config["tracking"]["bootstrap_metrics_path"] = str(bootstrap_path.resolve())
    _write_yaml(config_path, config)
    own_latest = run_dir / "models/latest_checkpoint.pt"
    checkpoint = own_latest if own_latest.is_file() else source_run / "models/latest_checkpoint.pt"
    payload = _checkpoint(checkpoint)
    if int(payload["epoch"]) < args.branch + staged["screening_epochs"]:
        raise CampaignError("final continuation checkpoint predates screening completion")
    code = run_command(
        _command(args, config_path, relative_name, resume=checkpoint),
        run_dir, poll=args.poll_interval,
    )
    if code != 0:
        raise CampaignError(f"final continuation exited with status {code}")
    final_epoch = int(config["training"]["epochs"])
    _checkpoint(own_latest, final_epoch)
    records = _read_records(run_dir / "validation-metrics.jsonl")
    if not records or int(records[-1]["epoch"]) != final_epoch:
        raise CampaignError("final continuation has no epoch-500 metrics")
    continued_history = [
        {
            **record,
            "betti_active": 1.0,
            "betti_activation_epoch": args.branch,
        }
        for record in records
    ]
    _write_jsonl(
        run_dir / "stitched-validation-metrics.jsonl",
        _merge_epoch_records(candidate_history, continued_history),
    )
    completion = {
        "completed_at": utc_now(), "branch_epoch": args.branch,
        "trial": args.trial, "epoch": final_epoch,
        "betti_supervised_epochs": final_epoch - args.branch,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(complete, completion)
    registry = args.output / "final-continuations/selections.jsonl"
    existing = []
    if registry.is_file():
        existing = [json.loads(line) for line in registry.read_text().splitlines() if line]
    if not any(item["branch_epoch"] == args.branch and item["trial"] == args.trial for item in existing):
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**completion, "run_dir": str(run_dir.resolve())}, sort_keys=True) + "\n")
    print(name + " COMPLETE")


def run_final(args, base, staged):
    _campaign_marker(args.output, args.initial_weights, args.config)
    name = f"branch-e{args.branch}-final-{args.trial:04d}"
    selection_file = args.output / "final-continuations/selection-plan.jsonl"
    selection_lock = args.output / "final-continuations/.locks/selection.lock"
    with AllocationLock(selection_lock):
        selections = []
        if selection_file.is_file():
            selections = [
                json.loads(line)
                for line in selection_file.read_text().splitlines()
                if line.strip()
            ]
        selected = any(
            item["branch_epoch"] == args.branch and item["trial"] == args.trial
            for item in selections
        )
        branch_count = sum(
            item["branch_epoch"] == args.branch for item in selections
        )
        if not selected and branch_count >= 2:
            raise CampaignError(
                f"branch {args.branch} already has the maximum two selected candidates"
            )
        if not selected:
            role = "manual"
            suggestion_path = args.output / "summaries/candidate-suggestions.json"
            if suggestion_path.is_file():
                suggestion = json.loads(suggestion_path.read_text()).get(
                    str(args.branch), {}
                )
                if suggestion.get("balanced") == args.trial:
                    role = "balanced"
                elif suggestion.get("h1_oriented") == args.trial:
                    role = "h1_oriented"
            selection_file.parent.mkdir(parents=True, exist_ok=True)
            with selection_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "branch_epoch": args.branch,
                    "trial": args.trial,
                    "role": role,
                    "selected_at": utc_now(),
                }, sort_keys=True) + "\n")
    lock = args.output / "final-continuations/.locks" / (name + ".lock")
    with AllocationLock(lock):
        _run_final_unlocked(args, base, staged)


def run_final_test(args, base, staged):
    """Evaluate exactly one explicitly chosen completed model on test data."""
    _campaign_marker(args.output, args.initial_weights, args.config)
    if args.branch not in staged["branch_epochs"]:
        raise CampaignError("invalid final-test branch")
    name = f"branch-e{args.branch}-final-{args.trial:04d}"
    run_dir = args.output / "final-continuations" / name
    if not (run_dir / ".complete.json").is_file():
        raise CampaignError("selected final continuation is not complete")
    config_path = args.output / "final-continuations/configs" / (name + ".yaml")
    checkpoint = run_dir / "models/latest_checkpoint.pt"
    _checkpoint(checkpoint, int(base["training"]["epochs"]))
    evaluation = args.output / "test-evaluation"
    selection_path = evaluation / "final-selection.json"
    selection = {
        "branch_epoch": args.branch,
        "trial": args.trial,
        "run": name,
        "checkpoint": str(checkpoint.resolve()),
    }
    lock = args.output / "test-evaluation.lock"
    with AllocationLock(lock):
        if selection_path.is_file():
            existing = json.loads(selection_path.read_text())
            if existing != selection:
                raise CampaignError(
                    "test split was already assigned to a different final model"
                )
            if (evaluation / ".complete.json").is_file():
                print("final test evaluation already complete")
                return
        else:
            _atomic_json(selection_path, selection)
        command = [
            sys.executable, "-u", "evaluate.py",
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(evaluation),
            "--dataset", "synthetic_mri",
            "--split", "test",
        ]
        return_code = run_command(command, evaluation, poll=args.poll_interval)
        if return_code != 0:
            raise CampaignError(
                f"one-time test evaluation exited with status {return_code}"
            )
        summary = evaluation / "summary.json"
        if not summary.is_file():
            raise CampaignError("test evaluation produced no summary")
        _atomic_json(evaluation / ".complete.json", {
            **selection, "completed_at": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })


def _final_rows(args, staged):
    registry = args.output / "final-continuations/selections.jsonl"
    if not registry.is_file():
        return []
    prefix_records = _read_records(args.output / "shared-prefix/validation-metrics.jsonl")
    final_epoch = int(max(record["epoch"] for record in prefix_records))
    controls = [record for record in prefix_records if int(record["epoch"]) == final_epoch]
    if len(controls) != 1:
        raise CampaignError("shared prefix needs exactly one final control record")
    control = controls[0]
    rows = []
    for item in [json.loads(line) for line in registry.read_text().splitlines() if line]:
        run_dir = Path(item["run_dir"])
        records = _read_records(run_dir / "validation-metrics.jsonl")
        finals = [record for record in records if int(record["epoch"]) == final_epoch]
        if len(finals) != 1:
            continue
        metrics = finals[0]
        row = {
            "branch_epoch": item["branch_epoch"], "trial": item["trial"],
            "betti_supervised_epochs": final_epoch - int(item["branch_epoch"]),
            "run_dir": str(run_dir),
        }
        for name in FINAL_METRICS:
            row[name] = metrics.get(name)
            row["control_" + name] = control.get(name)
            if metrics.get(name) is not None and control.get(name) is not None:
                row[DELTA_METRIC_NAMES.get(name, "delta_" + name)] = (
                    float(metrics[name]) - float(control[name])
                )
        rows.append(row)
    return rows


def status(args, base, staged):
    marker = args.output / MARKER_NAME
    print("campaign_marker=" + ("OK" if marker.is_file() else "MISSING"))
    test_selection = args.output / "test-evaluation/final-selection.json"
    if test_selection.is_file():
        selection = json.loads(test_selection.read_text())
        print(
            "test_selection=branch:{} trial:{} complete={}".format(
                selection["branch_epoch"], selection["trial"],
                (args.output / "test-evaluation/.complete.json").is_file(),
            )
        )
    else:
        print("test_selection=UNUSED")
    prefix_latest = args.output / "shared-prefix/models/latest_checkpoint.pt"
    if prefix_latest.is_file():
        try:
            epoch = _checkpoint(prefix_latest)["epoch"]
        except CampaignError as error:
            print("prefix_checkpoint=INVALID " + str(error))
        else:
            completion_path = args.output / "shared-prefix/.complete.json"
            completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
            print(
                f"prefix_latest_epoch={epoch} complete={completion_path.is_file()} "
                f"job={completion.get('slurm_job_id') or '-'}"
            )
    optuna = _import_optuna()
    for branch in staged["branch_epochs"]:
        directory = _branch_dir(args.output, branch)
        if not (directory / "study.journal").is_file():
            print(f"branch={branch} study=NOT_STARTED")
            continue
        study = _study(optuna, directory, branch, staged)
        counts = {name: 0 for name in ("COMPLETE", "RUNNING", "FAIL")}
        for trial in study.trials:
            counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
            run_dir = trial.user_attrs.get("run_dir")
            checkpoint = Path(run_dir) / "models/latest_checkpoint.pt" if run_dir else None
            latest_epoch = None
            if checkpoint and checkpoint.is_file():
                try:
                    latest_epoch = _checkpoint(checkpoint)["epoch"]
                except CampaignError:
                    latest_epoch = "INVALID"
            print(
                f"branch={branch} trial={trial.number} state={trial.state.name} "
                f"latest_epoch={latest_epoch or '-'} checkpoint={bool(checkpoint and checkpoint.is_file())} "
                f"resume_count={trial.user_attrs.get('resume_count', 0)} "
                f"job={trial.user_attrs.get('slurm_job_id') or '-'}"
            )
        print(
            f"branch={branch} complete={counts.get('COMPLETE', 0)} "
            f"running={counts.get('RUNNING', 0)} failed={counts.get('FAIL', 0)} "
            f"total={len(study.trials)}/{staged['trials_per_branch']}"
        )


def print_preflight(args, base, staged):
    data = base["data"]["datasets"]["synthetic_mri"]
    print("Staged Betti activation preflight")
    print("  output:", args.output)
    print("  initialization:", args.initial_weights)
    print("  dataset:", data["root"])
    print("  samples: train={} validation={}".format(data["train_samples"], data["validation_samples"]))
    print("  branch epochs:", staged["branch_epochs"])
    print("  screening endpoints:", [value + staged["screening_epochs"] for value in staged["branch_epochs"]])
    print("  trials per branch:", staged["trials_per_branch"])
    gpus = int(os.environ.get("GNBM_STAGED_GPUS", "1"))
    print(
        "  execution: gpus={} distributed={} per_gpu_batch={} global_batch={}".format(
            gpus,
            base["runtime"]["distributed"],
            base["data"]["batch_size"],
            base["data"]["batch_size"] * gpus,
        )
    )
    print("  objectives:", list(DELTA_OBJECTIVES))
    print("  matcher: hungarian; node loss: focal; edge loss: cross_entropy; alpha: 0.5")
    sampling = base["topology"]["sampling"]
    print(
        "  topology sampling: enabled={} common={} h0={} h1={} validation=full".format(
            sampling["enabled"],
            sampling["max_graphs_per_rank"],
            sampling["betti_h0_max_graphs_per_rank"],
            sampling["betti_h1_max_graphs_per_rank"],
        )
    )
    print("  test split: UNUSED")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("preflight", "prefix", "prepare", "screen", "summarize", "final", "test", "status"))
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--initial-weights", type=Path, required=True)
    result.add_argument("--branch", type=int)
    result.add_argument("--trial", type=int)
    result.add_argument("--worker-trials", type=int, default=0)
    result.add_argument("--resume-running", action="store_true")
    result.add_argument("--poll-interval", type=float, default=2.0)
    result.add_argument("--train-command", nargs=argparse.REMAINDER)
    return result


def main():
    args = parser().parse_args()
    args.output = args.output.expanduser().resolve()
    args.initial_weights = args.initial_weights.expanduser().resolve()
    try:
        base, staged = load_staged(args.config)
        if args.mode in {"screen", "final", "test"} and args.branch is None:
            raise CampaignError("--branch is required")
        if args.mode in {"final", "test"} and args.trial is None:
            raise CampaignError("--trial is required")
        if args.mode == "preflight":
            print_preflight(args, base, staged)
        elif args.mode == "prefix":
            run_prefix(args, base, staged)
        elif args.mode == "prepare":
            prepare_controls(args, base, staged)
        elif args.mode == "screen":
            run_screen_worker(args, base, staged)
        elif args.mode == "summarize":
            summarize(args, base, staged)
        elif args.mode == "final":
            run_final(args, base, staged)
        elif args.mode == "test":
            run_final_test(args, base, staged)
        else:
            status(args, base, staged)
    except KeyboardInterrupt:
        print("Interrupted; active trial remains RUNNING and resumable", file=sys.stderr)
        raise SystemExit(130)
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit("Staged Betti campaign error: " + str(error)) from error


if __name__ == "__main__":
    main()
