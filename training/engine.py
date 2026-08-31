"""Dependency-light training and validation loops for RelationFormer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist

from .checkpoint import (
    alias_training_checkpoint,
    capture_runtime_state,
    load_runtime_state,
    save_runtime_state,
    save_training_checkpoint,
)
from .evaluation import evaluate_model
from .early_stopping import EarlyStopping


def _write_json(path, payload):
    """Replace small provenance records only after checkpoint writes succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _move_graph_batch(batch, device, input_name, supervise_target_graphs):
    images, segmentations, nodes, edges, _, domains = batch
    volumes = images if input_name == "image" else segmentations
    domains = domains.to(device=device, non_blocking=True)
    if supervise_target_graphs:
        keep = torch.arange(len(nodes), device=device)
    else:
        keep = torch.nonzero(domains == 0, as_tuple=False).flatten()
    if keep.numel() == 0:
        return None
    indices = keep.detach().cpu().tolist()
    return (
        volumes[keep.cpu()].to(device=device, dtype=torch.float32, non_blocking=True),
        {
            "nodes": [nodes[index].to(device=device) for index in indices],
            "edges": [edges[index].to(device=device) for index in indices],
        },
    )


def train_step(
    model,
    criterion,
    optimizer,
    scheduler,
    batch,
    config: Mapping,
    device,
    *,
    epoch: int,
    iteration: int,
    total_iterations: int,
):
    prepared = _move_graph_batch(
        batch,
        device,
        config["training"]["input"],
        bool(config["loss"]["supervise_target_graphs"]),
    )
    if prepared is None:
        return None
    volumes, targets = prepared
    criterion.set_training_progress(
        epoch, 100.0 * max(0, iteration - 1) / max(1, total_iterations - 1)
    )
    model.train()
    optimizer.zero_grad()
    tokens, predictions, _ = model(volumes)
    losses = criterion(tokens, predictions, targets)
    if not bool(torch.isfinite(losses["total"].detach()).item()):
        raise FloatingPointError("non-finite total training loss")
    losses["total"].backward()
    optimizer.step()
    scheduler.step()
    return losses


@torch.no_grad()
def evaluate_loss(model, criterion, loader, config: Mapping, device):
    model.eval()
    totals = {}
    batches = 0
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
        tokens, predictions, _ = model(volumes)
        losses = criterion(tokens, predictions, targets)
        for name, value in losses.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        validation_criterion,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        config,
        device,
        tracker=None,
        metric_loader=None,
        *,
        evaluation_model=None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.evaluation_model = evaluation_model or model
        self.criterion = criterion
        self.validation_criterion = validation_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = torch.device(device)
        self.tracker = tracker
        self.metric_loader = metric_loader
        self.rank = int(rank)
        self.world_size = int(world_size)

    @property
    def is_primary(self):
        return self.rank == 0

    def _distributed(self):
        return self.world_size > 1 and dist.is_initialized()

    def _reduce_epoch_totals(self, sums, batches):
        if not self._distributed():
            return sums, batches
        names = sorted(sums)
        values = torch.tensor(
            [*(sums[name] for name in names), float(batches)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return (
            {name: float(values[index].item()) for index, name in enumerate(names)},
            int(values[-1].item()),
        )

    def _save_checkpoints(self, paths, epoch, iteration, trainer_state):
        if not paths:
            return
        paths = list(dict.fromkeys(Path(path) for path in paths))
        generator = getattr(self.train_loader, "generator", None)
        local_state = capture_runtime_state(generator)
        if self._distributed():
            # Do not use gather_object with an NCCL process group here.  Object
            # collectives create an extra CUDA-side communication path and have
            # proved capable of hanging at the first checkpoint on Jean-Zay.
            # The run directory is shared by all ranks, so exchange these small
            # per-rank states through atomic files and keep NCCL for tensors.
            runtime_directory = paths[0].parent / ".runtime_states"
            runtime_path = runtime_directory / "rank={}.pt".format(self.rank)
            save_runtime_state(runtime_path, local_state)
            dist.barrier()
            gathered = None
            if self.is_primary:
                gathered = [
                    load_runtime_state(
                        runtime_directory / "rank={}.pt".format(rank)
                    )
                    for rank in range(self.world_size)
                ]
        else:
            gathered = [local_state]
        if self.is_primary:
            save_training_checkpoint(
                paths[0],
                self.evaluation_model,
                self.optimizer,
                self.scheduler,
                epoch,
                iteration,
                runtime_states=gathered,
                trainer_state=trainer_state,
            )
            for path in paths[1:]:
                alias_training_checkpoint(paths[0], path)
            if self._distributed():
                for rank in range(self.world_size):
                    (runtime_directory / "rank={}.pt".format(rank)).unlink(
                        missing_ok=True
                    )
                try:
                    runtime_directory.rmdir()
                except OSError:
                    pass

    def fit(self, start_epoch=0, start_iteration=0, trainer_state=None):
        epochs = int(self.config["training"]["epochs"])
        total_iterations = epochs * len(self.train_loader)
        global_iteration = int(start_iteration)
        output = (
            Path(self.config["experiment"]["output_dir"])
            / self.config["experiment"]["name"]
            / "models"
        )
        interval = int(
            self.config["training"]["checkpoint"]["interval_epochs"]
        )
        latest_interval = int(
            self.config["training"]["checkpoint"].get(
                "latest_interval_epochs", interval
            )
        )
        validation_interval = int(self.config["evaluation"]["interval_epochs"])
        policy = self.config["training"]["checkpoint"]["policy"]
        metric_config = self.config["evaluation"]["training_metrics"]
        metric_enabled = bool(metric_config["enabled"])
        if metric_enabled and self.metric_loader is None and self.is_primary:
            raise ValueError(
                "evaluation.training_metrics.enabled requires a metric loader"
            )
        selection_metric = str(metric_config["selection_metric"])
        selection_mode = str(metric_config["selection_mode"])
        save_metric_checkpoint = bool(metric_config["save_best_checkpoint"])
        trainer_state = dict(trainer_state or {})
        for key, current in (("selection_metric", selection_metric), ("selection_mode", selection_mode)):
            if key in trainer_state and trainer_state[key] != current:
                raise ValueError("Checkpoint selection configuration changed during resume: " + key)
        stopping = EarlyStopping(
            self.config["training"].get("early_stopping"),
            trainer_state.get("early_stopping"),
        )
        if stopping.stopped:
            if self.is_primary:
                print("Early stopping was already reached in the resumed checkpoint; no training performed.")
            return
        save_f1 = bool(metric_config.get("save_f1_checkpoints", False))
        best_f1 = dict(trainer_state.get("best_f1", {}))
        best_metric = trainer_state.get(
            "best_metric",
            -float("inf") if selection_mode == "max" else float("inf"),
        )
        metric_record_path = output.parent / "best-metric.json"
        if "best_metric" not in trainer_state and metric_record_path.is_file():
            record = json.loads(metric_record_path.read_text(encoding="utf-8"))
            if (
                record.get("metric") == selection_metric
                and record.get("mode") == selection_mode
                and record.get("epoch", 0) <= start_epoch
            ):
                best_metric = float(record["value"])
        best_validation = float(trainer_state.get("best_validation", float("inf")))

        for epoch in range(int(start_epoch) + 1, epochs + 1):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            sums = {}
            batches = 0
            for batch in self.train_loader:
                global_iteration += 1
                losses = train_step(
                    self.model,
                    self.criterion,
                    self.optimizer,
                    self.scheduler,
                    batch,
                    self.config,
                    self.device,
                    epoch=epoch,
                    iteration=global_iteration,
                    total_iterations=total_iterations,
                )
                if losses is None:
                    continue
                scalars = {}
                for name, value in losses.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        scalar = float(value.detach().cpu())
                        scalars[name] = scalar
                        sums[name] = sums.get(name, 0.0) + scalar
                if self.tracker is not None and self.is_primary:
                    self.tracker.log_training(
                        scalars,
                        iteration=global_iteration,
                        epoch=epoch,
                        learning_rate=self.optimizer.param_groups[0]["lr"],
                    )
                batches += 1
            sums, batches = self._reduce_epoch_totals(sums, batches)
            means = {name: value / max(1, batches) for name, value in sums.items()}
            if self.is_primary:
                print(
                    "epoch={}/{} total={:.6f} lr={:.8g}".format(
                        epoch,
                        epochs,
                        means.get("total", float("nan")),
                        self.optimizer.param_groups[0]["lr"],
                    )
                )

            save_best_validation = False
            save_best_metric = False
            save_node_f1 = False
            save_edge_f1 = False
            stop_training = False
            pending_records = []
            monitored_metrics = {}
            if epoch % validation_interval == 0 or epoch == epochs:
                if self._distributed():
                    dist.barrier()
                if self.is_primary:
                    validation = evaluate_loss(
                        self.evaluation_model,
                        self.validation_criterion,
                        self.validation_loader,
                        self.config,
                        self.device,
                    )
                    print(
                        "validation epoch={} total={:.6f}".format(
                            epoch, validation.get("total", float("nan"))
                        )
                    )
                    if self.tracker is not None:
                        self.tracker.log_validation(
                            validation,
                            iteration=global_iteration,
                            epoch=epoch,
                        )
                    validation_total = validation.get("total")
                    monitored_metrics["validation_total"] = validation_total
                    if (
                        policy in {"best_only", "interval_and_best"}
                        and validation_total is not None
                        and validation_total < best_validation
                    ):
                        best_validation = validation_total
                        save_best_validation = True

                if metric_enabled and self.is_primary:
                    task_metrics, _ = evaluate_model(
                        self.evaluation_model,
                        self.metric_loader,
                        self.config,
                        self.device,
                        output_dir=None,
                        max_visualizations=0,
                        export_predictions=False,
                    )
                    selected_value = task_metrics.get(selection_metric)
                    monitored_metrics.update(task_metrics)
                    print(
                        "metrics epoch={} node_mAP={:.6f} edge_mAP={:.6f} "
                        "beta0_abs={:.6f} beta1_abs={:.6f} smd={:.6f} "
                        "node_F1={:.6f} edge_F1={:.6f}".format(
                            epoch,
                            task_metrics.get("node_mAP", float("nan")),
                            task_metrics.get("edge_mAP", float("nan")),
                            task_metrics.get("beta0_absolute_error", float("nan")),
                            task_metrics.get("beta1_absolute_error", float("nan")),
                            task_metrics.get("smd", float("nan")),
                            task_metrics.get("node_f1", float("nan")),
                            task_metrics.get("edge_f1", float("nan")),
                        )
                    )
                    if self.tracker is not None:
                        self.tracker.log_metrics(
                            task_metrics,
                            iteration=global_iteration,
                            epoch=epoch,
                        )
                    history_path = output.parent / "validation-metrics.jsonl"
                    history_path.parent.mkdir(parents=True, exist_ok=True)
                    history_metrics = {
                        name: (
                            value
                            if not isinstance(value, float) or math.isfinite(value)
                            else None
                        )
                        for name, value in task_metrics.items()
                    }
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "epoch": int(epoch),
                                    "iteration": int(global_iteration),
                                    **history_metrics,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    if selected_value is not None and math.isfinite(
                        float(selected_value)
                    ):
                        selected_value = float(selected_value)
                        improved = (
                            selected_value > best_metric
                            if selection_mode == "max"
                            else selected_value < best_metric
                        )
                        if improved:
                            best_metric = selected_value
                            if save_metric_checkpoint:
                                save_best_metric = True
                                pending_records.append((
                                    metric_record_path, {
                                            "checkpoint": "models/best_metric_checkpoint.pt",
                                            "epoch": int(epoch),
                                            "iteration": int(global_iteration),
                                            "metric": selection_metric,
                                            "mode": selection_mode,
                                            "value": selected_value,
                                        }
                                ))

                    if save_f1:
                        for name in ("node_f1", "edge_f1"):
                            value = task_metrics.get(name)
                            if value is None or not math.isfinite(float(value)):
                                raise FloatingPointError("Missing/non-finite checkpoint metric: " + name)
                            if name in best_f1 and value <= best_f1[name]["value"]:
                                continue
                            record = {
                                "checkpoint": "models/best_{}_checkpoint.pt".format(name),
                                "epoch": epoch, "iteration": global_iteration,
                                "metric": name, "mode": "max", "value": float(value),
                                "node_threshold": self.config["evaluation"]["node_threshold"],
                                "edge_threshold": self.config["evaluation"]["edge_threshold"],
                                "protocol": dict(self.config["evaluation"]["protocol"]),
                                "f1_iou_threshold": self.config["evaluation"]["protocol"].get("f1_iou_threshold", 0.5),
                                "f1_aggregation": "micro_all_retained_detections",
                                "metrics": history_metrics,
                            }
                            best_f1[name] = record
                            pending_records.append((
                                output.parent / ("best-" + name.replace("_", "-") + ".json"), record
                            ))
                            if name == "node_f1":
                                save_node_f1 = True
                            else:
                                save_edge_f1 = True

                if self.is_primary:
                    stop_training = stopping.update(epoch, monitored_metrics)
                    if stopping.config["enabled"]:
                        print("patience epoch={} monitor={} best={:.6f} best_epoch={} stale_epochs={}/{} stop={}".format(
                            epoch, stopping.config["monitor"], stopping.best,
                            stopping.best_epoch, epoch - stopping.best_epoch,
                            stopping.config["patience_epochs"], stop_training,
                        ))

                if self._distributed():
                    flags = torch.tensor(
                        [save_best_validation, save_best_metric, save_node_f1, save_edge_f1, stop_training],
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    dist.broadcast(flags, src=0)
                    save_best_validation, save_best_metric, save_node_f1, save_edge_f1, stop_training = (
                        bool(value) for value in flags.tolist()
                    )

            paths = []
            if save_best_validation:
                paths.append(output / "best_checkpoint.pt")
            if save_best_metric:
                paths.append(output / "best_metric_checkpoint.pt")
            if save_node_f1:
                paths.append(output / "best_node_f1_checkpoint.pt")
            if save_edge_f1:
                paths.append(output / "best_edge_f1_checkpoint.pt")
            if policy in {"interval", "interval_and_best"} and (
                epoch % interval == 0 or epoch == epochs
            ):
                paths.append(output / "checkpoint_epoch={}.pt".format(epoch))
            if policy != "none" and (
                epoch % latest_interval == 0 or epoch == epochs or stop_training
            ):
                paths.append(output / "latest_checkpoint.pt")
            self._save_checkpoints(
                paths,
                epoch,
                global_iteration,
                {
                    "best_validation": best_validation,
                    "best_metric": best_metric,
                    "selection_metric": selection_metric,
                    "selection_mode": selection_mode,
                    "best_f1": best_f1,
                    "early_stopping": stopping.state_dict(),
                    "world_size": self.world_size,
                },
            )
            if self.is_primary:
                for path, record in pending_records:
                    _write_json(path, record)
                    if hasattr(self.tracker, "log_selection"):
                        self.tracker.log_selection(record)
                if stopping.config["enabled"]:
                    _write_json(output.parent / "early-stopping.json", stopping.state_dict())
                    if stopping.last_epoch == epoch and hasattr(self.tracker, "log_stopping"):
                        self.tracker.log_stopping(stopping.state_dict())
                if stop_training or epoch == epochs:
                    _write_json(output.parent / "training-status.json", {
                        "reason": "early_stopping" if stop_training else "max_epochs",
                        "epoch": epoch, "iteration": global_iteration,
                        "max_epochs": epochs,
                    })
            if self._distributed():
                dist.barrier()
            if stop_training:
                break
__all__ = ["Trainer", "evaluate_loss", "train_step"]
