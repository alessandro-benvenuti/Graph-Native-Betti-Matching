"""Optional Weights & Biases experiment tracking."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


class WandbTracker:
    """Small adapter keeping W&B concerns out of the training engine."""

    def __init__(self, run):
        self.run = run
        self.run.define_metric("train/iteration")
        self.run.define_metric("train/*", step_metric="train/iteration")
        self.run.define_metric("validation/epoch")
        self.run.define_metric("validation/*", step_metric="validation/epoch")
        self.run.define_metric("metrics/epoch")
        self.run.define_metric("metrics/*", step_metric="metrics/epoch")
        self.run.define_metric("stopping/epoch")
        self.run.define_metric("stopping/*", step_metric="stopping/epoch")
        self.run.define_metric(
            "validation/total",
            step_metric="validation/epoch",
            summary="min",
            overwrite=True,
        )
        for name, summary in (
            ("node_mAP", "max"),
            ("node_mAR", "max"),
            ("edge_mAP", "max"),
            ("edge_mAR", "max"),
            ("node_precision", "max"),
            ("node_recall", "max"),
            ("node_f1", "max"),
            ("edge_precision", "max"),
            ("edge_recall", "max"),
            ("edge_f1", "max"),
            ("beta0_absolute_error", "min"),
            ("beta1_absolute_error", "min"),
            ("smd", "min"),
        ):
            self.run.define_metric(
                "metrics/" + name,
                step_metric="metrics/epoch",
                summary=summary,
                overwrite=True,
            )

    def log_training(self, metrics, *, iteration: int, epoch: int, learning_rate: float):
        payload = {
            "train/iteration": int(iteration),
            "train/epoch": int(epoch),
            "train/learning_rate": float(learning_rate),
        }
        payload.update({"train/" + name: float(value) for name, value in metrics.items()})
        self.run.log(payload)

    def log_validation(self, metrics, *, iteration: int, epoch: int):
        payload = {
            "validation/iteration": int(iteration),
            "validation/epoch": int(epoch),
        }
        payload.update(
            {"validation/" + name: float(value) for name, value in metrics.items()}
        )
        self.run.log(payload)

    def log_metrics(self, metrics, *, iteration: int, epoch: int):
        payload = {
            "metrics/iteration": int(iteration),
            "metrics/epoch": int(epoch),
        }
        payload.update(
            {
                "metrics/" + name: float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        self.run.log(payload)

    def log_selection(self, record):
        self.run.summary["checkpoints/" + record["metric"]] = {
            key: record[key]
            for key in ("epoch", "iteration", "metric", "mode", "value", "checkpoint")
        }

    def log_stopping(self, state):
        self.run.log({
            "stopping/epoch": state["last_epoch"],
            "stopping/best": state["best"],
            "stopping/best_epoch": state["best_epoch"],
            "stopping/stale_epochs": state["last_epoch"] - state["best_epoch"],
            "stopping/patience_epochs": state["config"]["patience_epochs"],
            "stopping/min_epochs": state["config"].get("min_epochs", 0),
            "stopping/stop": int(state["stopped"]),
        })

    def finish(self, exit_code: int = 0):
        self.run.finish(exit_code=int(exit_code))


def _read_run_id(path: Path):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read W&B run metadata: {path}") from error
    run_id = payload.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f"W&B run metadata has no valid id: {path}")
    return run_id


def build_tracker(
    config: Mapping,
    run_dir: Path,
    *,
    resume: bool,
    launch_metadata: Mapping | None = None,
):
    """Initialize W&B and preserve its run ID for checkpoint resumes."""

    settings = config["tracking"]
    if not settings["enabled"]:
        print("W&B tracking is disabled by configuration")
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed. "
            "Install the Jean-Zay environment again or set tracking.enabled=false."
        ) from error

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "wandb-run.json"
    run_id = _read_run_id(metadata_path) if resume else None
    mode = os.environ.get("WANDB_MODE") or settings.get("mode")
    project = os.environ.get("WANDB_PROJECT") or settings["project"]
    entity = os.environ.get("WANDB_ENTITY") or settings.get("entity")
    group = os.environ.get("WANDB_RUN_GROUP") or settings.get("group")
    tracked_config = dict(config)
    if launch_metadata:
        tracked_config["launch"] = dict(launch_metadata)

    run = wandb.init(
        project=project,
        entity=entity,
        name=config["experiment"]["name"],
        group=group,
        tags=tuple(settings["tags"]),
        config=tracked_config,
        dir=str(run_dir),
        id=run_id,
        resume="allow" if run_id is not None else None,
        mode=mode,
        save_code=bool(settings["save_code"]),
    )
    metadata = {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": run.entity,
        "url": getattr(run, "url", None),
        "mode": mode or "online",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "W&B run id={} mode={} url={}".format(
            run.id,
            metadata["mode"],
            metadata["url"] or "local",
        )
    )
    return WandbTracker(run)


__all__ = ["WandbTracker", "build_tracker"]
