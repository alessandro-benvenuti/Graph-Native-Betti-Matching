"""Dataset-level inference, metric aggregation, and optional BN calibration."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch

from .inference import infer_graphs
from .metrics import evaluate_graph, summarize_metrics
from .visualization import save_graph_comparison


def _evaluation_volumes(batch, input_name, device):
    images, segmentations = batch[0], batch[1]
    volumes = images if input_name == "image" else segmentations
    return volumes.to(device=device, dtype=torch.float32, non_blocking=True)


def _disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) or "Dropout" in module.__class__.__name__:
            module.eval()


@torch.no_grad()
def calibrate_batch_norm(model, loader, config: Mapping, device, *, batches: int):
    """Update only BatchNorm running statistics using evaluation-domain inputs."""

    if int(batches) <= 0:
        return 0
    model.train()
    _disable_dropout(model)
    consumed = 0
    for batch in loader:
        model(_evaluation_volumes(batch, config["training"]["input"], device))
        consumed += 1
        if consumed >= int(batches):
            break
    model.eval()
    return consumed


def _tensor_values(values):
    return torch.as_tensor(values).detach().cpu().tolist()


def _prediction_record(sample_id, source_sample_id, prediction, metrics):
    return {
        "sample_id": sample_id,
        "source_sample_id": source_sample_id,
        "nodes_dhw": _tensor_values(prediction["nodes"]),
        "node_boxes_center_size_dhw": _tensor_values(prediction["boxes"]),
        "node_scores": _tensor_values(prediction["node_scores"]),
        "query_ids": _tensor_values(prediction["query_ids"]),
        "edges": _tensor_values(prediction["edges"]),
        "edge_scores": _tensor_values(prediction["edge_scores"]),
        "metrics": metrics,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_per_patch_metrics(path, records):
    """Write a compact table while keeping JSON as the lossless graph export."""

    metric_names = sorted(
        set().union(*(record["metrics"].keys() for record in records))
    )
    field_names = ["sample_id", "source_sample_id"] + metric_names
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            row = {
                "sample_id": record["sample_id"],
                "source_sample_id": record["source_sample_id"],
            }
            for name in metric_names:
                value = record["metrics"].get(name)
                row[name] = (
                    value
                    if not isinstance(value, float) or math.isfinite(value)
                    else ""
                )
            writer.writerow(row)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    config: Mapping,
    device,
    *,
    output_dir: Optional[Path] = None,
    max_visualizations: int = 0,
    export_predictions: bool = True,
):
    """Evaluate one model and return ``(summary, per_sample_rows)``."""

    model.eval()
    relation = model.module.relation_embed if hasattr(model, "module") else model.relation_embed
    decoder = config["model"]["decoder"]
    evaluation = config["evaluation"]
    protocol = evaluation.get("protocol", {})
    rows = []
    records = []
    node_detection_states = []
    edge_detection_states = []
    sample_index = 0
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    dataset_records = getattr(getattr(loader, "dataset", None), "records", ())

    for batch in loader:
        volumes = _evaluation_volumes(
            batch, config["training"]["input"], device
        )
        tokens, predictions, _ = model(volumes)
        graphs = infer_graphs(
            tokens,
            predictions,
            relation,
            object_queries=int(decoder["object_queries"]),
            relation_tokens=int(decoder["relation_tokens"]),
            node_threshold=evaluation.get("node_threshold"),
            edge_threshold=evaluation.get("edge_threshold"),
        )
        for local_index, graph in enumerate(graphs):
            sample_id = "sample_{:06d}".format(sample_index)
            source_sample_id = (
                str(dataset_records[sample_index].sample_id)
                if sample_index < len(dataset_records)
                else None
            )
            metrics, node_state, edge_state = evaluate_graph(
                graph,
                batch[2][local_index],
                batch[3][local_index],
                protocol=protocol,
                return_detection_states=True,
            )
            rows.append(metrics)
            node_detection_states.append(node_state)
            edge_detection_states.append(edge_state)
            if export_predictions:
                records.append(
                    _prediction_record(sample_id, source_sample_id, graph, metrics)
                )
            if output_dir is not None and sample_index < int(max_visualizations):
                save_graph_comparison(
                    batch[1][local_index],
                    batch[2][local_index],
                    batch[3][local_index],
                    graph,
                    output_dir / "plots" / (sample_id + ".png"),
                    title=sample_id,
                )
            sample_index += 1

    summary = summarize_metrics(
        rows,
        folds=int(protocol.get("folds", 5)),
        node_detection_states=node_detection_states,
        edge_detection_states=edge_detection_states,
        iou_thresholds=protocol.get("iou_thresholds"),
    )
    if output_dir is not None:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")
        if export_predictions:
            with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(records), handle, indent=2)
                handle.write("\n")
            _write_per_patch_metrics(output_dir / "per-patch-metrics.csv", records)
    return summary, rows


__all__ = ["calibrate_batch_norm", "evaluate_model"]
