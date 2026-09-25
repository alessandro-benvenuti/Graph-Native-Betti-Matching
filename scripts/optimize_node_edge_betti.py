#!/usr/bin/env python3
"""Run fixed-tail, four-objective Betti studies around the existing train.py."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from configs import validate_config
from configs.loader import _expand_environment, _load_with_defaults

OBJECTIVES = (("node_mAP", "maximize"), ("edge_mAP", "maximize"),
              ("beta0_absolute_error", "minimize"),
              ("beta1_absolute_error", "minimize"))
PRIMARY_METRICS = tuple(metric for metric, _ in OBJECTIVES)
KNOWN_DIAGNOSTICS = (
    "node_mAR", "edge_mAR", "node_precision", "node_recall", "node_f1",
    "edge_precision", "edge_recall", "edge_f1", "smd", "target_beta0",
    "predicted_beta0", "target_beta1", "predicted_beta1", "target_nodes",
    "predicted_nodes", "node_count_absolute_error", "target_edges",
    "predicted_edges", "edge_count_absolute_error",
)
METRICS = PRIMARY_METRICS + KNOWN_DIAGNOSTICS
LEGACY_KEYS = {"direction", "objective", "pruner"}


class CampaignError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_set(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts, current = dotted.split("."), mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise CampaignError(f"Cannot set {dotted}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def validate_optuna_config(config: Mapping[str, Any]) -> None:
    legacy = sorted(LEGACY_KEYS.intersection(config))
    if legacy:
        raise CampaignError(
            "legacy scalar Optuna fields are unsupported ({}); migrate to objectives, "
            "metric_aggregation, and sampler.name=nsga2".format(", ".join(legacy))
        )
    required = {"study_name", "n_trials", "seed", "objectives",
                "metric_aggregation", "sampler", "search_space"}
    missing = sorted(required - set(config))
    if missing:
        raise CampaignError("missing optuna keys: " + ", ".join(missing))
    if not isinstance(config["n_trials"], int) or config["n_trials"] <= 0:
        raise CampaignError("optuna.n_trials must be a positive integer")
    actual = tuple((item.get("metric"), item.get("direction"))
                   for item in config["objectives"])
    if actual != OBJECTIVES:
        raise CampaignError(f"objectives must be exactly {list(OBJECTIVES)}")
    aggregation = config["metric_aggregation"]
    if aggregation.get("name") != "tail_mean":
        raise CampaignError("only metric_aggregation.name=tail_mean is supported")
    observations = aggregation.get("observations")
    if not isinstance(observations, int) or observations <= 0:
        raise CampaignError("metric_aggregation.observations must be positive")
    sampler = config["sampler"]
    if sampler.get("name") != "nsga2":
        raise CampaignError("sampler.name must be nsga2")
    if not isinstance(sampler.get("population_size"), int) or sampler["population_size"] < 2:
        raise CampaignError("sampler.population_size must be at least 2")
    space = config["search_space"]
    for key in ("topology.betti_h0.weight", "topology.betti_h1.weight",
                "topology.betti_h1.false_positive_weight", "betti_warmup_epochs",
                "betti_ramp_epochs"):
        if not isinstance(space.get(key), list) or not space[key]:
            raise CampaignError(f"search space {key} must be a non-empty list")
    if any(float(value) <= 0 for value in space["topology.betti_h1.weight"]):
        raise CampaignError("H1 must remain active in every trial")


def load_campaign(path: Path, environment: Mapping[str, str] | None = None):
    merged = _load_with_defaults(path, set())
    resolved = _expand_environment(
        merged, os.environ if environment is None else environment, "config"
    )
    optuna_config = resolved.pop("optuna", None)
    resolved.pop("study_b_proposal", None)
    if not isinstance(optuna_config, dict):
        raise CampaignError("campaign configuration requires an optuna mapping")
    validate_optuna_config(optuna_config)
    validate_config(resolved)
    if resolved["model"]["matcher"]["type"] != "hungarian":
        raise CampaignError("Betti Pareto studies require model.matcher.type=hungarian")
    return resolved, optuna_config


def apply_parameters(config: Mapping[str, Any], parameters: Mapping[str, Any]):
    concrete = copy.deepcopy(dict(config))
    for name, value in parameters.items():
        if name == "betti_warmup_epochs":
            deep_set(concrete, "topology.betti_h0.warmup_epochs", value)
            deep_set(concrete, "topology.betti_h1.warmup_epochs", value)
        elif name == "betti_ramp_epochs":
            deep_set(concrete, "topology.betti_h0.ramp_epochs", value)
            deep_set(concrete, "topology.betti_h1.ramp_epochs", value)
        elif name == "edge_loss":
            if value == "cross_entropy":
                deep_set(concrete, "loss.edge.classification.name", "cross_entropy")
                deep_set(concrete, "loss.edge.balancing.mode", "ratio_upsample")
            elif isinstance(value, str) and value.startswith("focal_gamma_"):
                try:
                    gamma = float(value.removeprefix("focal_gamma_"))
                except ValueError as error:
                    raise CampaignError(f"invalid edge_loss choice: {value}") from error
                deep_set(concrete, "loss.edge.classification.name", "focal")
                deep_set(concrete, "loss.edge.classification.focal_gamma", gamma)
                deep_set(concrete, "loss.edge.balancing.mode", "none")
            else:
                raise CampaignError(f"invalid edge_loss choice: {value}")
        else:
            deep_set(concrete, name, value)
    validate_config(concrete)
    if concrete["model"]["matcher"]["type"] != "hungarian":
        raise CampaignError("search parameters may not change the Hungarian matcher")
    return concrete


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"validation metric {name} is not numeric") from error
    if not math.isfinite(number):
        raise CampaignError(f"validation metric {name} is not finite")
    return number


def aggregate_tail(records: Sequence[Mapping[str, Any]], observations: int) -> dict[str, Any]:
    if len(records) < observations:
        raise CampaignError(
            f"tail_mean_{observations} requires {observations} validation observations; found {len(records)}"
        )
    tail, epochs = list(records[-observations:]), []
    for record in tail:
        if "epoch" not in record:
            raise CampaignError("validation record is missing epoch")
        epochs.append(int(record["epoch"]))
    metric_names = sorted(set(tail[0]) - {"epoch", "iteration"})
    for required in PRIMARY_METRICS:
        if required not in metric_names:
            raise CampaignError(f"validation record is missing {required}")
    means = {}
    for name in metric_names:
        values = []
        for record in tail:
            if name not in record:
                raise CampaignError(f"tail validation record is missing {name}")
            values.append(_finite_number(record[name], name))
        means[name] = sum(values) / len(values)
    return {"method": f"tail_mean_{observations}", "name": "tail_mean",
            "observations": observations, "contributing_epochs": epochs,
            "final_epoch": int(records[-1]["epoch"]), "metrics": means,
            "objectives": tuple(means[name] for name in PRIMARY_METRICS)}


class JsonlMonitor:
    def __init__(self, path: Path): self.path, self.offset, self.pending = path, 0, ""
    def read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists(): return []
        if self.path.stat().st_size < self.offset:
            raise CampaignError(f"validation history was truncated: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset); chunk, self.offset = handle.read(), handle.tell()
        lines, records = (self.pending + chunk).splitlines(keepends=True), []
        self.pending = ""
        for line in lines:
            if not line.endswith(("\n", "\r")):
                self.pending = line; continue
            if not line.strip(): continue
            try: value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"malformed validation JSONL in {self.path}: {error}") from error
            if not isinstance(value, dict): raise CampaignError("validation JSONL record must be an object")
            records.append(value)
        return records


class ControllerLock:
    def __init__(self, path: Path): self.path, self.fd = path, None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try: self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error: raise CampaignError(f"controller lock exists: {self.path}") from error
        os.write(self.fd, f"pid={os.getpid()}\n".encode()); return self
    def __exit__(self, *_):
        if self.fd is not None: os.close(self.fd)
        try: self.path.unlink()
        except FileNotFoundError: pass


class AllocationLock:
    def __init__(self, path: Path): self.path, self.handle = path, None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX); return self
    def __exit__(self, *_):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN); self.handle.close()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def compatibility_payload(config, initial_weights: str) -> dict[str, Any]:
    checkpoint = Path(initial_weights).expanduser().resolve()
    if not checkpoint.is_file(): raise CampaignError(f"initial checkpoint does not exist: {checkpoint}")
    return {"schema_version": 2, "initial_weights": str(checkpoint),
            "initial_weights_sha256": _file_sha256(checkpoint),
            "experiment_seed": config["experiment"]["seed"], "data": config["data"],
            "optimizer": config["training"]["optimizer"], "scheduler": config["training"]["scheduler"],
            "base_loss": config["loss"], "model": config["model"],
            "training_epochs": config["training"]["epochs"],
            "evaluation_interval": config["evaluation"]["interval_epochs"]}


def compatibility_digest(config, initial_weights: str) -> str:
    return hashlib.sha256(json.dumps(compatibility_payload(config, initial_weights),
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    monitor = JsonlMonitor(path); records = monitor.read_new()
    if monitor.pending.strip(): raise CampaignError(f"incomplete final JSONL record in {path}")
    return records


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: yaml.safe_dump(dict(value), handle, sort_keys=False)


def _terminate(process: subprocess.Popen, grace: float = 10.0) -> None:
    if process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=grace)
    except subprocess.TimeoutExpired: process.kill(); process.wait()


def run_command(command: list[str], run_dir: Path, on_record=None, poll=0.2):
    monitor, process, interrupted = JsonlMonitor(run_dir / "validation-metrics.jsonl"), subprocess.Popen(command), False
    def stop_child(_signum, _frame):
        nonlocal interrupted
        interrupted = True; _terminate(process)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous: signal.signal(sig, stop_child)
    try:
        while process.poll() is None:
            for record in monitor.read_new():
                if on_record: on_record(record, process)
            time.sleep(poll)
        for record in monitor.read_new():
            if on_record: on_record(record, process)
    except BaseException: _terminate(process); raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
    if interrupted: raise KeyboardInterrupt
    return process.returncode


def _training_command(args, config_path: Path, run_name: str) -> list[str]:
    if args.train_command:
        return [part.format(config=config_path, output=args.output, run_name=run_name,
                            initial_weights=args.initial_weights) for part in args.train_command]
    return [sys.executable, "-u", "train.py", "--config", str(config_path),
            "--output-dir", str(args.output), "--run-name", run_name,
            "--initial-weights", args.initial_weights]


def run_control(args, base, config) -> None:
    reference_path, run_dir = args.output / "control-reference.json", args.output / "control"
    if reference_path.exists() or run_dir.exists(): raise CampaignError("control artifacts already exist; frozen controls are immutable")
    control = copy.deepcopy(base); control["experiment"]["name"] = "control"
    for name in ("betti_h0", "betti_h1"): control["topology"][name].update(enabled=False, log_only=True, weight=0.0)
    config_path = args.output / "configs/control.yaml"; _write_yaml(config_path, control); records = []
    return_code = run_command(_training_command(args, config_path, "control"), run_dir,
                              lambda record, _process: records.append(record), args.poll_interval)
    if return_code != 0: raise CampaignError(f"control process exited with status {return_code}")
    if not records: records = _read_records(run_dir / "validation-metrics.jsonl")
    aggregation = aggregate_tail(records, config["metric_aggregation"]["observations"])
    manifest = run_dir / "dataset-manifest.json"
    if not manifest.is_file(): raise CampaignError("control did not produce dataset-manifest.json")
    reference = {"schema_version": 2, "aggregation": aggregation,
                 "metrics": aggregation["metrics"], "objectives": list(aggregation["objectives"]),
                 "training_completed": True, "process_return_code": return_code,
                 "compatibility_digest": compatibility_digest(base, args.initial_weights),
                 "compatibility": compatibility_payload(base, args.initial_weights),
                 "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 "run_dir": str(run_dir.resolve()), "config_path": str(config_path.resolve()),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    reference_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(f"Frozen {aggregation['method']} control reference: {reference_path}")


def _import_optuna():
    try: import optuna
    except ImportError as error: raise CampaignError("install requirements/optuna.txt in this environment") from error
    return optuna


def build_sampler(optuna, config):
    return optuna.samplers.NSGAIISampler(seed=int(config["seed"]),
        population_size=int(config["sampler"]["population_size"]))


def create_storage(optuna, output: Path, kind: str):
    if kind == "sqlite": return f"sqlite:///{(output / 'study.sqlite3').resolve()}"
    if kind != "journal": raise CampaignError("storage must be sqlite or journal")
    journal_cls = getattr(optuna.storages, "JournalStorage", None)
    try:
        from optuna.storages.journal import JournalFileBackend as file_cls
    except ImportError:
        file_cls = getattr(optuna.storages, "JournalFileStorage", None)
    if journal_cls is None or file_cls is None: raise CampaignError("installed Optuna lacks JournalStorage/JournalFileStorage")
    return journal_cls(file_cls(str((output / "study.journal").resolve())))


def create_study(optuna, output, config, storage_kind):
    study = optuna.create_study(study_name=config["study_name"],
        storage=create_storage(optuna, output, storage_kind),
        directions=[direction for _, direction in OBJECTIVES],
        sampler=build_sampler(optuna, config), load_if_exists=True)
    study.set_user_attr("objectives", list(OBJECTIVES))
    study.set_user_attr("metric_aggregation", config["metric_aggregation"])
    study.set_user_attr("sampler", config["sampler"])
    study.set_user_attr("requested_trials", config["n_trials"])
    return study


def recover_stale_trials(optuna, study) -> int:
    stale = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.RUNNING]
    for trial in stale:
        study._storage.set_trial_user_attr(trial._trial_id, "failure_reason", "stale RUNNING trial recovered explicitly")
        study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.FAIL)
    return len(stale)


def execute_trial(args, base, config, reference, trial):
    started = time.monotonic(); trial.set_user_attr("started_at", utc_now())
    parameters = {name: trial.suggest_categorical(name, values) for name, values in config["search_space"].items()}
    run_name, run_dir = f"runs/trial_{trial.number:04d}", args.output / f"runs/trial_{trial.number:04d}"
    if run_dir.exists(): raise CampaignError(f"refusing to reuse existing trial directory: {run_dir}")
    concrete = apply_parameters(base, parameters); concrete["experiment"]["name"] = run_name
    config_path = args.output / f"trial-configs/trial_{trial.number:04d}.yaml"; _write_yaml(config_path, concrete)
    trial.set_user_attr("run_dir", str(run_dir.resolve())); trial.set_user_attr("config_path", str(config_path.resolve()))
    trial.set_user_attr("slurm_job_id", os.environ.get("SLURM_JOB_ID")); records = []
    try:
        return_code = run_command(_training_command(args, config_path, run_name), run_dir,
                                  lambda record, _process: records.append(record), args.poll_interval)
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "optuna-validation-history.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    if return_code != 0: raise CampaignError(f"training process exited with status {return_code}")
    if not records: records = _read_records(run_dir / "validation-metrics.jsonl")
    aggregation = aggregate_tail(records, config["metric_aggregation"]["observations"])
    manifest = run_dir / "dataset-manifest.json"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != reference["dataset_manifest_sha256"]:
        raise CampaignError("trial dataset manifest differs from paired control")
    trial.set_user_attr("aggregation", aggregation); trial.set_user_attr("training_completed", True)
    trial.set_user_attr("process_return_code", return_code); trial.set_user_attr("final_epoch", aggregation["final_epoch"])
    trial.set_user_attr("duration_seconds", time.monotonic() - started); trial.set_user_attr("finished_at", utc_now())
    for name, value in aggregation["metrics"].items(): trial.set_user_attr(name, value)
    return aggregation["objectives"]


def _allocate_trial(study, maximum: int, lock_path: Path):
    with AllocationLock(lock_path):
        if len(study.get_trials(deepcopy=False)) >= maximum: return None
        return study.ask()


def run_worker(args, base, config) -> None:
    optuna = _import_optuna(); reference_path = args.output / "control-reference.json"
    if not reference_path.is_file(): raise CampaignError("control-reference.json is missing; run control first")
    reference = json.loads(reference_path.read_text())
    if reference.get("compatibility_digest") != compatibility_digest(base, args.initial_weights):
        raise CampaignError("control reference is incompatible with config/checkpoint")
    study = create_study(optuna, args.output, config, args.storage)
    maximum, attempts = args.max_trials or int(config["n_trials"]), 0
    while args.worker_trials <= 0 or attempts < args.worker_trials:
        trial = _allocate_trial(study, maximum, args.output / ".trial-allocation.lock")
        if trial is None: break
        attempts += 1
        try: values = execute_trial(args, base, config, reference, trial)
        except KeyboardInterrupt:
            trial.set_user_attr("failure_reason", "interrupted by SIGTERM/SIGINT")
            study.tell(trial, state=optuna.trial.TrialState.FAIL); raise
        except Exception as error:
            trial.set_user_attr("failure_reason", f"{type(error).__name__}: {error}")
            trial.set_user_attr("training_completed", False)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {trial.number} failed: {error}", file=sys.stderr)
        else:
            study.tell(trial, values=values)
            print(f"Trial {trial.number} COMPLETE values={tuple(round(v, 6) for v in values)}")
    print(f"worker attempts={attempts} global_trials={len(study.trials)} storage={args.storage}")


def print_preflight(base, config, args) -> None:
    dataset, workers = base["data"]["datasets"]["synthetic_mri"], int(os.environ.get("GNBM_OPTUNA_WORKERS", "1"))
    print("Pareto campaign preflight")
    print(f"  initialization: {args.initial_weights}\n  dataset: {dataset['root']}")
    print(f"  samples: train={dataset['train_samples']} validation={dataset['validation_samples']}")
    print(f"  epochs: {base['training']['epochs']} aggregation={config['metric_aggregation']}")
    print(f"  objectives: {list(OBJECTIVES)}")
    print(f"  sampler: NSGA-II population={config['sampler']['population_size']} seed={config['seed']}")
    print(f"  trials={config['n_trials']} workers={workers} estimated_gpu_jobs={workers}")
    print(f"  test split: UNUSED\n  matcher: {base['model']['matcher']['type']}")
    print(f"  edge loss: {base['loss']['edge']['classification']['name']}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("control", "worker", "recover", "preflight"))
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-weights", required=True); parser.add_argument("--storage", choices=("sqlite", "journal"), default="sqlite")
    parser.add_argument("--max-trials", type=int); parser.add_argument("--worker-trials", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=2.0); parser.add_argument("--train-command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args(); args.output = args.output.expanduser().resolve()
    try:
        base, config = load_campaign(args.config)
        if args.mode == "preflight": print_preflight(base, config, args); return 0
        if args.mode == "recover":
            optuna = _import_optuna(); study = create_study(optuna, args.output, config, args.storage)
            print(f"recovered stale trials: {recover_stale_trials(optuna, study)}"); return 0
        if args.storage == "sqlite":
            with ControllerLock(args.output / ".controller.lock"):
                run_control(args, base, config) if args.mode == "control" else run_worker(args, base, config)
        elif args.mode == "control":
            with ControllerLock(args.output / ".control.lock"): run_control(args, base, config)
        else: run_worker(args, base, config)
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Optuna campaign error: {error}") from error
    return 0


if __name__ == "__main__": raise SystemExit(main())
