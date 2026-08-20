"""Dependency-light training and validation loops for RelationFormer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .checkpoint import save_training_checkpoint
from .evaluation import evaluate_model


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
    ):
        self.model = model
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

    def fit(self, start_epoch=0, start_iteration=0):
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
        validation_interval = int(self.config["evaluation"]["interval_epochs"])
        policy = self.config["training"]["checkpoint"]["policy"]
        metric_config = self.config["evaluation"]["training_metrics"]
        metric_enabled = bool(metric_config["enabled"])
        if metric_enabled and self.metric_loader is None:
            raise ValueError(
                "evaluation.training_metrics.enabled requires a metric loader"
            )
        selection_metric = str(metric_config["selection_metric"])
        selection_mode = str(metric_config["selection_mode"])
        save_metric_checkpoint = bool(metric_config["save_best_checkpoint"])
        best_metric = -float("inf") if selection_mode == "max" else float("inf")
        metric_record_path = output.parent / "best-metric.json"
        if metric_record_path.is_file():
            record = json.loads(metric_record_path.read_text(encoding="utf-8"))
            if (
                record.get("metric") == selection_metric
                and record.get("mode") == selection_mode
            ):
                best_metric = float(record["value"])
        best_validation = float("inf")
        if int(start_epoch) > 0 and policy in {"best_only", "interval_and_best"}:
            resumed_validation = evaluate_loss(
                self.model,
                self.validation_criterion,
                self.validation_loader,
                self.config,
                self.device,
            )
            best_validation = resumed_validation.get("total", best_validation)

        for epoch in range(int(start_epoch) + 1, epochs + 1):
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
                if self.tracker is not None:
                    self.tracker.log_training(
                        scalars,
                        iteration=global_iteration,
                        epoch=epoch,
                        learning_rate=self.optimizer.param_groups[0]["lr"],
                    )
                batches += 1
            means = {name: value / max(1, batches) for name, value in sums.items()}
            print(
                "epoch={}/{} total={:.6f} lr={:.8g}".format(
                    epoch,
                    epochs,
                    means.get("total", float("nan")),
                    self.optimizer.param_groups[0]["lr"],
                )
            )

            if epoch % validation_interval == 0 or epoch == epochs:
                validation = evaluate_loss(
                    self.model,
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
                if (
                    policy in {"best_only", "interval_and_best"}
                    and validation_total is not None
                    and validation_total < best_validation
                ):
                    best_validation = validation_total
                    save_training_checkpoint(
                        output / "best_checkpoint.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        global_iteration,
                    )

                if metric_enabled:
                    task_metrics, _ = evaluate_model(
                        self.model,
                        self.metric_loader,
                        self.config,
                        self.device,
                        output_dir=None,
                        max_visualizations=0,
                        export_predictions=False,
                    )
                    selected_value = task_metrics.get(selection_metric)
                    print(
                        "metrics epoch={} node_mAP={:.6f} edge_mAP={:.6f} "
                        "beta0_abs={:.6f} beta1_abs={:.6f} smd={:.6f}".format(
                            epoch,
                            task_metrics.get("node_mAP", float("nan")),
                            task_metrics.get("edge_mAP", float("nan")),
                            task_metrics.get("beta0_absolute_error", float("nan")),
                            task_metrics.get("beta1_absolute_error", float("nan")),
                            task_metrics.get("smd", float("nan")),
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
                                save_training_checkpoint(
                                    output / "best_metric_checkpoint.pt",
                                    self.model,
                                    self.optimizer,
                                    self.scheduler,
                                    epoch,
                                    global_iteration,
                                )
                                metric_record_path.write_text(
                                    json.dumps(
                                        {
                                            "checkpoint": "models/best_metric_checkpoint.pt",
                                            "epoch": int(epoch),
                                            "iteration": int(global_iteration),
                                            "metric": selection_metric,
                                            "mode": selection_mode,
                                            "value": selected_value,
                                        },
                                        indent=2,
                                        sort_keys=True,
                                    )
                                    + "\n",
                                    encoding="utf-8",
                                )

            if policy in {"interval", "interval_and_best"} and (
                epoch % interval == 0 or epoch == epochs
            ):
                save_training_checkpoint(
                    output / "checkpoint_epoch={}.pt".format(epoch),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    global_iteration,
                )
__all__ = ["Trainer", "evaluate_loss", "train_step"]
