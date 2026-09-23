#!/usr/bin/env python3
"""Run a paired, resumable Optuna search around the existing train.py process."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from configs import validate_config
from configs.loader import _expand_environment, _load_with_defaults


METRICS = (
    "node_mAP", "edge_mAP", "node_f1", "edge_f1",
    "beta0_absolute_error", "beta1_absolute_error", "smd",
)
CONSTRAINT_METRICS = ("node_f1", "edge_f1", "node_mAP", "edge_mAP")


class CampaignError(RuntimeError):
    pass


def deep_set(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise CampaignError(f"Cannot set {dotted}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def load_campaign(path: Path, environment: Mapping[str, str] | None = None):
    """Resolve defaults/environment, remove controller metadata, validate training config."""
    merged = _load_with_defaults(path, set())
    resolved = _expand_environment(
        merged, os.environ if environment is None else environment, "config"
    )
    optuna_config = resolved.pop("optuna", None)
    if not isinstance(optuna_config, dict):
        raise CampaignError("campaign configuration requires an optuna mapping")
    validate_optuna_config(optuna_config)
    validate_config(resolved)
    return resolved, optuna_config


def validate_optuna_config(config: Mapping[str, Any]) -> None:
    required = {"study_name", "direction", "n_trials", "objective", "search_space"}
    missing = sorted(required - set(config))
    if missing:
        raise CampaignError("missing optuna keys: " + ", ".join(missing))
    if config["direction"] != "minimize":
        raise CampaignError("only optuna.direction=minimize is supported")
    if not isinstance(config["n_trials"], int) or config["n_trials"] <= 0:
        raise CampaignError("optuna.n_trials must be a positive integer")
    objective = config["objective"]
    tolerances = objective.get("tolerances", {})
    for name in CONSTRAINT_METRICS:
        if name not in tolerances or float(tolerances[name]) < 0:
            raise CampaignError(f"missing/non-negative tolerance required for {name}")
    for key in ("penalty_scale", "epsilon"):
        if float(objective.get(key, 0)) <= 0:
            raise CampaignError(f"optuna.objective.{key} must be positive")
    space = config["search_space"]
    for key in (
        "topology.betti_h0.weight", "topology.betti_h1.weight",
        "topology.betti_h1.false_positive_weight", "betti_warmup_epochs",
        "betti_ramp_epochs",
    ):
        if not isinstance(space.get(key), list) or not space[key]:
            raise CampaignError(f"search space {key} must be a non-empty list")
    if any(float(value) <= 0 for value in space["topology.betti_h1.weight"]):
        raise CampaignError("H1 must remain active in every trial")
    if config.get("sampler", {}).get("name", "tpe") != "tpe":
        raise CampaignError("sampler.name must be tpe")


def apply_parameters(config: Mapping[str, Any], parameters: Mapping[str, Any]):
    concrete = copy.deepcopy(dict(config))
    for name, value in parameters.items():
        if name == "betti_warmup_epochs":
            deep_set(concrete, "topology.betti_h0.warmup_epochs", value)
            deep_set(concrete, "topology.betti_h1.warmup_epochs", value)
        elif name == "betti_ramp_epochs":
            deep_set(concrete, "topology.betti_h0.ramp_epochs", value)
            deep_set(concrete, "topology.betti_h1.ramp_epochs", value)
        else:
            deep_set(concrete, name, value)
    validate_config(concrete)
    return concrete


def _finite_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name in METRICS:
        value = record.get(name)
        if value is None:
            raise CampaignError(f"validation record is missing {name}")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise CampaignError(f"validation metric {name} is not numeric") from error
        if not math.isfinite(number):
            raise CampaignError(f"validation metric {name} is not finite")
        result[name] = number
    return result


def compute_objective(record, control, objective_config):
    metrics = _finite_metrics(record)
    reference = _finite_metrics(control)
    epsilon = float(objective_config["epsilon"])
    topology_score = 0.5 * metrics["beta0_absolute_error"] / max(
        reference["beta0_absolute_error"], epsilon
    ) + 0.5 * metrics["beta1_absolute_error"] / max(
        reference["beta1_absolute_error"], epsilon
    )
    tolerances = objective_config["tolerances"]
    violations = {
        name: max(0.0, reference[name] - float(tolerances[name]) - metrics[name])
        for name in CONSTRAINT_METRICS
    }
    penalty = float(objective_config["penalty_scale"]) * sum(violations.values())
    return {
        "objective": topology_score + penalty,
        "topology_score": topology_score,
        "penalty": penalty,
        "feasible": not any(value > 0 for value in violations.values()),
        "violations": violations,
        "total_violation": sum(violations.values()),
        "metrics": metrics,
    }


class JsonlMonitor:
    """Incrementally read complete JSONL records, retaining partial final lines."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.pending = ""

    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            raise CampaignError(f"validation history was truncated: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        text = self.pending + chunk
        lines = text.splitlines(keepends=True)
        self.pending = ""
        records = []
        for line in lines:
            if not line.endswith(("\n", "\r")):
                self.pending = line
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise CampaignError(f"malformed validation JSONL in {self.path}: {error}") from error
            if not isinstance(value, dict):
                raise CampaignError(f"validation JSONL record is not an object: {self.path}")
            records.append(value)
        return records


class ControllerLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error:
            raise CampaignError(
                f"controller lock exists: {self.path}; do not run concurrent SQLite controllers"
            ) from error
        os.write(self.fd, f"pid={os.getpid()}\n".encode())
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_payload(config, initial_weights: str) -> dict[str, Any]:
    checkpoint = Path(initial_weights).expanduser().resolve()
    if not checkpoint.is_file():
        raise CampaignError(f"initial checkpoint does not exist: {checkpoint}")
    return {
        "schema_version": 1,
        "initial_weights": str(checkpoint),
        "initial_weights_sha256": _file_sha256(checkpoint),
        "experiment_seed": config["experiment"]["seed"],
        "data": config["data"],
        "optimizer": config["training"]["optimizer"],
        "scheduler": config["training"]["scheduler"],
        "base_loss": config["loss"],
        "model": config["model"],
    }


def compatibility_digest(config, initial_weights: str) -> str:
    encoded = json.dumps(
        compatibility_payload(config, initial_weights), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    monitor = JsonlMonitor(path)
    records = monitor.read_new()
    if monitor.pending.strip():
        raise CampaignError(f"incomplete final JSONL record in {path}")
    return records


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)


def _terminate(process: subprocess.Popen, grace: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_command(command: list[str], run_dir: Path, on_record=None, poll=0.2):
    monitor = JsonlMonitor(run_dir / "validation-metrics.jsonl")
    process = subprocess.Popen(command)
    interrupted = False

    def stop_child(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        _terminate(process)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, stop_child)
    try:
        while process.poll() is None:
            for record in monitor.read_new():
                if on_record is not None:
                    on_record(record, process)
            time.sleep(poll)
        for record in monitor.read_new():
            if on_record is not None:
                on_record(record, process)
    except BaseException:
        _terminate(process)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if interrupted:
        raise KeyboardInterrupt
    return process.returncode


def _training_command(args, config_path: Path, run_name: str) -> list[str]:
    if args.train_command:
        return [part.format(config=config_path, output=args.output, run_name=run_name,
                            initial_weights=args.initial_weights)
                for part in args.train_command]
    return [sys.executable, "-u", "train.py", "--config", str(config_path),
            "--output-dir", str(args.output), "--run-name", run_name,
            "--initial-weights", args.initial_weights]


def run_control(args, base, optuna_config) -> None:
    output = args.output
    reference_path = output / "control-reference.json"
    if reference_path.exists():
        raise CampaignError(f"control reference already exists: {reference_path}")
    run_name = "control"
    run_dir = output / run_name
    if run_dir.exists():
        raise CampaignError(f"control run directory already exists: {run_dir}")
    control = copy.deepcopy(base)
    control["experiment"]["name"] = run_name
    for name in ("betti_h0", "betti_h1"):
        control["topology"][name].update(enabled=False, log_only=True, weight=0.0)
    config_path = output / "configs" / "control.yaml"
    _write_yaml(config_path, control)
    command = _training_command(args, config_path, run_name)
    if run_command(command, run_dir, poll=args.poll_interval) != 0:
        raise CampaignError("control training process failed")
    records = _read_records(run_dir / "validation-metrics.jsonl")
    if not records:
        raise CampaignError("control produced no validation metrics")
    metrics = _finite_metrics(records[-1])
    manifest_path = run_dir / "dataset-manifest.json"
    if not manifest_path.is_file():
        raise CampaignError("control did not produce dataset-manifest.json")
    reference = {
        "schema_version": 1,
        "selection": "last_validation_epoch",
        "epoch": int(records[-1]["epoch"]),
        "metrics": metrics,
        "compatibility_digest": compatibility_digest(base, args.initial_weights),
        "compatibility": compatibility_payload(base, args.initial_weights),
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "run_dir": str(run_dir.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(f"Frozen control reference: {reference_path}")


def _import_optuna():
    try:
        import optuna
    except ImportError as error:
        raise CampaignError(
            "Optuna is not installed; install requirements/optuna.txt in this environment"
        ) from error
    return optuna


def build_sampler_pruner(optuna, config):
    seed = int(config.get("seed", 364505))
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner_config = config.get("pruner", {"name": "median"})
    if pruner_config.get("name") == "none":
        pruner = optuna.pruners.NopPruner()
    elif pruner_config.get("name") == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_config.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_config.get("n_warmup_steps", 3)),
            interval_steps=int(pruner_config.get("interval_steps", 1)),
        )
    else:
        raise CampaignError("pruner.name must be median or none")
    return sampler, pruner


def recover_stale_trials(optuna, study) -> None:
    stale = [trial for trial in study.get_trials(deepcopy=False)
             if trial.state == optuna.trial.TrialState.RUNNING]
    for trial in stale:
        study._storage.set_trial_state_values(
            trial._trial_id, optuna.trial.TrialState.FAIL
        )
        print(f"Marked stale RUNNING trial {trial.number} as FAIL; artifacts retained")


def _set_result_attributes(trial, result, epoch):
    trial.set_user_attr("best_objective_epoch", int(epoch))
    trial.set_user_attr("best_objective_value", result["objective"])
    trial.set_user_attr("topology_score", result["topology_score"])
    trial.set_user_attr("penalty", result["penalty"])
    trial.set_user_attr("feasible", result["feasible"])
    trial.set_user_attr("constraint_violations", result["violations"])
    for name, value in result["metrics"].items():
        trial.set_user_attr(name, value)


def optimize(args, base, config) -> None:
    optuna = _import_optuna()
    output = args.output
    reference_path = output / "control-reference.json"
    if not reference_path.is_file():
        raise CampaignError("control-reference.json is missing; run control first")
    reference = json.loads(reference_path.read_text())
    expected = compatibility_digest(base, args.initial_weights)
    if reference.get("compatibility_digest") != expected:
        raise CampaignError("control reference is incompatible with config/checkpoint")
    sampler, pruner = build_sampler_pruner(optuna, config)
    storage = f"sqlite:///{(output / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=config["study_name"], storage=storage, direction="minimize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    recover_stale_trials(optuna, study)
    space = config["search_space"]
    control_metrics = reference["metrics"]

    def objective(trial):
        parameters = {name: trial.suggest_categorical(name, values)
                      for name, values in space.items()}
        run_name = f"runs/trial_{trial.number:04d}"
        run_dir = output / run_name
        if run_dir.exists():
            raise CampaignError(f"refusing to reuse existing trial directory: {run_dir}")
        concrete = apply_parameters(base, parameters)
        concrete["experiment"]["name"] = run_name
        config_path = output / "trial-configs" / f"trial_{trial.number:04d}.yaml"
        _write_yaml(config_path, concrete)
        trial.set_user_attr("run_dir", str(run_dir.resolve()))
        trial.set_user_attr("config_path", str(config_path.resolve()))
        trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
        warmup = int(parameters["betti_warmup_epochs"])
        best = None
        best_edge = None
        last = None
        history = []
        report_step = 0

        def observe(record, process):
            nonlocal best, best_edge, last, report_step
            epoch = int(record.get("epoch", -1))
            metrics = _finite_metrics(record)
            last = {"epoch": epoch, "metrics": metrics}
            if best_edge is None or metrics["edge_mAP"] > best_edge["metrics"]["edge_mAP"]:
                best_edge = last
            if epoch <= warmup:
                history.append({"epoch": epoch, "reported": False, "metrics": metrics})
                return
            result = compute_objective(record, control_metrics, config["objective"])
            history.append({"epoch": epoch, "reported": True, **result})
            # Optuna's warm-up is counted in post-Betti validation observations,
            # not raw training epochs (which may start well after epoch 3).
            trial.report(result["objective"], step=report_step)
            report_step += 1
            if best is None or result["objective"] < best["result"]["objective"]:
                best = {"epoch": epoch, "result": result}
                _set_result_attributes(trial, result, epoch)
            if trial.should_prune():
                trial.set_user_attr("pruned_epoch", epoch)
                _terminate(process)
                raise optuna.TrialPruned(f"pruned at validation epoch {epoch}")

        try:
            return_code = run_command(
                _training_command(args, config_path, run_name), run_dir,
                on_record=observe, poll=args.poll_interval,
            )
        finally:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "optuna-history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n"
            )
        if return_code != 0:
            raise CampaignError(f"training process exited with status {return_code}")
        if best is None:
            raise CampaignError("trial produced no post-warm-up validation record")
        manifest = run_dir / "dataset-manifest.json"
        if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != reference["dataset_manifest_sha256"]:
            raise CampaignError("trial dataset manifest differs from paired control")
        trial.set_user_attr("best_edge_mAP_epoch", best_edge["epoch"] if best_edge else None)
        trial.set_user_attr("last_epoch", last["epoch"] if last else None)
        trial.set_user_attr("selected_epoch_feasible", best["result"]["feasible"])
        return best["result"]["objective"]

    remaining = max(0, int(config["n_trials"]) - len([
        t for t in study.trials if t.state in {
            optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED
        }
    ]))
    if remaining:
        study.optimize(objective, n_trials=remaining, catch=(CampaignError,))
    else:
        print("Requested completed/pruned trial count already reached; no trials launched")
    print(f"Study stored at {output / 'study.sqlite3'}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("control", "optimize"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--train-command", nargs=argparse.REMAINDER,
        help="test hook; command tokens may use {config}, {output}, {run_name}, {initial_weights}",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output = args.output.expanduser().resolve()
    try:
        base, config = load_campaign(args.config)
        with ControllerLock(args.output / ".controller.lock"):
            if args.mode == "control":
                run_control(args, base, config)
            else:
                optimize(args, base, config)
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Optuna campaign error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
