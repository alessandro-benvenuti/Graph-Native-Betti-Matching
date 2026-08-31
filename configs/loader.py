"""Small, dependency-light loader for the repository YAML configuration schema."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, MutableMapping, Optional, Set

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot be resolved or is internally invalid."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_with_defaults(path: Path, active: Set[Path]) -> Dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if resolved_path in active:
        chain = " -> ".join(str(item) for item in (*active, resolved_path))
        raise ConfigError(f"Cyclic configuration defaults: {chain}")
    if not resolved_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, MutableMapping):
        raise ConfigError(f"Top-level YAML value must be a mapping: {resolved_path}")

    raw = dict(loaded)
    defaults = raw.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list) or not all(
        isinstance(item, str) for item in defaults
    ):
        raise ConfigError(f"'defaults' must be a string list in {resolved_path}")

    merged: Dict[str, Any] = {}
    next_active = set(active)
    next_active.add(resolved_path)
    for default in defaults:
        parent = _load_with_defaults(resolved_path.parent / default, next_active)
        merged = _deep_merge(merged, parent)
    return _deep_merge(merged, raw)


def _expand_environment(value: Any, environment: Mapping[str, str], location: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, environment, f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_environment(item, environment, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    missing = sorted(
        {name for name in _ENV_PATTERN.findall(value) if name not in environment}
    )
    if missing:
        raise ConfigError(
            f"Missing environment variable(s) {', '.join(missing)} at {location}"
        )
    return _ENV_PATTERN.sub(lambda match: environment[match.group(1)], value)


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{location} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{location} must be a positive integer") from error
    if parsed <= 0 or parsed != value:
        raise ConfigError(f"{location} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, location: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{location} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{location} must be a non-negative integer") from error
    if parsed < 0 or parsed != value:
        raise ConfigError(f"{location} must be a non-negative integer")
    return parsed


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ConfigError("experiment must be a mapping")
    _non_negative_int(experiment.get("seed"), "experiment.seed")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ConfigError("runtime must be a mapping")
    _non_negative_int(runtime.get("workers"), "runtime.workers")
    if not isinstance(runtime.get("distributed", False), bool):
        raise ConfigError("runtime.distributed must be a boolean")

    tracking = config.get("tracking")
    if not isinstance(tracking, Mapping):
        raise ConfigError("tracking must be a mapping")
    if not isinstance(tracking.get("enabled"), bool):
        raise ConfigError("tracking.enabled must be a boolean")
    project = tracking.get("project")
    if not isinstance(project, str) or not project.strip():
        raise ConfigError("tracking.project must be a non-empty string")
    for name in ("entity", "group"):
        value = tracking.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"tracking.{name} must be null or a non-empty string")
    mode = tracking.get("mode")
    if mode not in {None, "online", "offline", "disabled", "shared"}:
        raise ConfigError(
            "tracking.mode must be null, online, offline, disabled, or shared"
        )
    tags = tracking.get("tags")
    if not isinstance(tags, list) or not all(
        isinstance(tag, str) and tag.strip() for tag in tags
    ):
        raise ConfigError("tracking.tags must be a list of non-empty strings")
    if not isinstance(tracking.get("save_code"), bool):
        raise ConfigError("tracking.save_code must be a boolean")

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ConfigError("data must be a mapping")
    if data.get("spatial_dims") != 3:
        raise ConfigError("The supported model requires data.spatial_dims=3")
    image_size = data.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 3:
        raise ConfigError("data.image_size must contain three spatial dimensions")
    for index, size in enumerate(image_size):
        _positive_int(size, f"data.image_size[{index}]")
    _positive_int(data.get("batch_size"), "data.batch_size")
    if "validation_batch_size" in data:
        _positive_int(
            data.get("validation_batch_size"), "data.validation_batch_size"
        )

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ConfigError("model must be a mapping")
    try:
        encoder = model["encoder"]
        decoder = model["decoder"]
    except (KeyError, TypeError) as error:
        raise ConfigError("model encoder/decoder configuration is incomplete") from error
    _positive_int(model.get("num_classes"), "model.num_classes")
    if encoder.get("type") != "se_resnet":
        raise ConfigError("model.encoder.type must be se_resnet")
    _positive_int(encoder.get("input_channels"), "model.encoder.input_channels")
    depths = encoder.get("depths")
    strides = encoder.get("strides")
    if not isinstance(depths, list) or len(depths) != 4:
        raise ConfigError("model.encoder.depths must contain four stages")
    if not isinstance(strides, list) or len(strides) != 4:
        raise ConfigError("model.encoder.strides must contain four stages")
    for index, depth in enumerate(depths):
        _positive_int(depth, f"model.encoder.depths[{index}]")
    for index, stride in enumerate(strides):
        _positive_int(stride, f"model.encoder.strides[{index}]")
    if decoder.get("type") != "deformable_detr":
        raise ConfigError("model.decoder.type must be deformable_detr")
    hidden_dim = _positive_int(
        decoder.get("hidden_dim"), "model.decoder.hidden_dim"
    )
    attention_heads = _positive_int(
        decoder.get("attention_heads"), "model.decoder.attention_heads"
    )
    if hidden_dim % attention_heads:
        raise ConfigError(
            "model.decoder.hidden_dim must be divisible by attention_heads"
        )
    if attention_heads not in {6, 26}:
        raise ConfigError("3D deformable attention supports 6 or 26 heads")
    try:
        decoder_dropout = float(decoder.get("dropout"))
    except (TypeError, ValueError) as error:
        raise ConfigError("model.decoder.dropout must be numeric") from error
    if not 0.0 <= decoder_dropout < 1.0:
        raise ConfigError("model.decoder.dropout must lie in [0,1)")
    if decoder.get("activation") not in {"relu", "gelu", "glu"}:
        raise ConfigError("model.decoder.activation must be relu, gelu, or glu")
    for name in (
        "encoder_layers",
        "decoder_layers",
        "feedforward_dim",
        "decoder_points",
        "encoder_points",
        "object_queries",
    ):
        _positive_int(decoder.get(name), f"model.decoder.{name}")
    if decoder.get("feature_levels") != 1:
        raise ConfigError("The supported 3D baseline requires one feature level")
    for name in ("relation_tokens", "dummy_tokens"):
        _non_negative_int(decoder.get(name), f"model.decoder.{name}")
    if not isinstance(decoder.get("relation_attention"), bool):
        raise ConfigError("model.decoder.relation_attention must be a boolean")
    if decoder.get("relation_attention") and decoder.get("relation_tokens", 0) == 0:
        raise ConfigError("relation_attention requires at least one relation token")
    if not isinstance(decoder.get("use_cuda_extension"), bool):
        raise ConfigError("model.decoder.use_cuda_extension must be a boolean")

    datasets = data.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ConfigError("data.datasets must contain at least one dataset")
    valid_names = {"plants", "synthetic_mri"}
    valid_roles = {"source", "target"}
    target_count = 0
    for name, dataset in datasets.items():
        location = f"data.datasets.{name}"
        if name not in valid_names:
            raise ConfigError(
                f"Unsupported dataset '{name}'; supported datasets: {sorted(valid_names)}"
            )
        if not isinstance(dataset, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        role = dataset.get("role")
        if role not in valid_roles:
            raise ConfigError(f"{location}.role must be 'source' or 'target'")
        target_count += int(role == "target")
        root = dataset.get("root")
        if not isinstance(root, str) or not root.strip():
            raise ConfigError(f"{location}.root must be a non-empty path")
        if "train_samples" not in dataset:
            raise ConfigError(f"{location}.train_samples is required")
        if dataset["train_samples"] is not None:
            _positive_int(dataset["train_samples"], f"{location}.train_samples")
        if (
            "validation_samples" in dataset
            and dataset["validation_samples"] is not None
        ):
            _positive_int(
                dataset["validation_samples"], f"{location}.validation_samples"
            )
        if name == "plants":
            if dataset.get("coordinate_order_on_disk") != ["y", "x"]:
                raise ConfigError(
                    f"{location}.coordinate_order_on_disk must be [y, x]"
                )
        elif name == "synthetic_mri":
            if dataset.get("coordinate_space_on_disk") not in {"normalized", "voxel"}:
                raise ConfigError(
                    f"{location}.coordinate_space_on_disk must be normalized or voxel"
                )
            try:
                float(dataset["foreground_mean"])
            except (KeyError, TypeError, ValueError) as error:
                raise ConfigError(f"{location}.foreground_mean must be numeric") from error
            selection = dataset.get("sample_cap_selection", "first")
            if selection not in {"first", "seeded_random"}:
                raise ConfigError(
                    f"{location}.sample_cap_selection must be first or seeded_random"
                )
            if selection == "seeded_random":
                _non_negative_int(
                    dataset.get("sample_cap_seed"),
                    f"{location}.sample_cap_seed",
                )

    if target_count == 0:
        raise ConfigError("At least one target dataset is required")

    mixed_sampling = data.get("mixed_sampling")
    if not isinstance(mixed_sampling, Mapping):
        raise ConfigError("data.mixed_sampling must be a mapping")
    if not isinstance(mixed_sampling.get("balance_source_target"), bool):
        raise ConfigError(
            "data.mixed_sampling.balance_source_target must be a boolean"
        )

    augmentation = config.get("augmentation")
    if not isinstance(augmentation, Mapping):
        raise ConfigError("augmentation must be a mapping")
    normalization = augmentation.get("normalization_denominator")
    if normalization != "axis_size":
        raise ConfigError(
            "augmentation.normalization_denominator must be 'axis_size'"
        )
    try:
        plants_augmentation = augmentation["plants"]
        projection_depth = _positive_int(
            plants_augmentation["projection_depth"],
            "augmentation.plants.projection_depth",
        )
        padding = _non_negative_int(
            plants_augmentation["padding"], "augmentation.plants.padding"
        )
        flip_probability = float(
            plants_augmentation["flip"]["probability_per_axis"]
        )
        mri_augmentation = augmentation["synthetic_mri"]
        zoom_range = mri_augmentation["zoom"]["range"]
        noise_range = mri_augmentation["gaussian_noise"]["std_range"]
        noise_probability = float(
            mri_augmentation["gaussian_noise"]["probability"]
        )
        clamp_range = mri_augmentation["intensity_clamp"]
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError("augmentation configuration is incomplete or invalid") from error
    if projection_depth % 2 == 0:
        raise ConfigError("augmentation.plants.projection_depth must be odd")
    if 2 * padding >= min(int(value) for value in image_size):
        raise ConfigError("augmentation.plants.padding leaves no inner image volume")
    if not 0.0 <= flip_probability <= 1.0:
        raise ConfigError("Plants flip probability must lie in [0,1]")
    if (
        not isinstance(zoom_range, list)
        or len(zoom_range) != 2
        or float(zoom_range[0]) <= 0
        or float(zoom_range[1]) < float(zoom_range[0])
    ):
        raise ConfigError("SyntheticMRI zoom range is invalid")
    if (
        not isinstance(noise_range, list)
        or len(noise_range) != 2
        or float(noise_range[0]) < 0
        or float(noise_range[1]) < float(noise_range[0])
    ):
        raise ConfigError("SyntheticMRI Gaussian-noise std range is invalid")
    if not 0.0 <= noise_probability <= 1.0:
        raise ConfigError("SyntheticMRI Gaussian-noise probability must lie in [0,1]")
    if clamp_range is not None and (
        not isinstance(clamp_range, list)
        or len(clamp_range) != 2
        or float(clamp_range[1]) <= float(clamp_range[0])
    ):
        raise ConfigError("SyntheticMRI intensity clamp must be null or an increasing pair")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a mapping")
    _positive_int(training.get("epochs"), "training.epochs")
    if training.get("input") not in {"image", "segmentation"}:
        raise ConfigError("training.input must be image or segmentation")
    optimizer = training.get("optimizer")
    scheduler = training.get("scheduler")
    if not isinstance(optimizer, Mapping) or optimizer.get("name") != "adamw":
        raise ConfigError("training.optimizer.name must be adamw")
    if not isinstance(scheduler, Mapping) or scheduler.get("name") != "polynomial":
        raise ConfigError("training.scheduler.name must be polynomial")
    checkpoint = training.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ConfigError("training.checkpoint must be a mapping")
    if checkpoint.get("policy") not in {
        "none",
        "best_only",
        "interval",
        "interval_and_best",
    }:
        raise ConfigError(
            "training.checkpoint.policy must be none, best_only, interval, "
            "or interval_and_best"
        )
    _positive_int(
        checkpoint.get("interval_epochs"),
        "training.checkpoint.interval_epochs",
    )
    if "latest_interval_epochs" in checkpoint:
        _positive_int(
            checkpoint.get("latest_interval_epochs"),
            "training.checkpoint.latest_interval_epochs",
        )

    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        raise ConfigError("loss must be a mapping")
    if not isinstance(loss.get("supervise_target_graphs"), bool):
        raise ConfigError("loss.supervise_target_graphs must be a boolean")
    try:
        node_classification = loss["node"]["classification"]
        edge = loss["edge"]
        classification_name = edge["classification"]["name"]
        candidates = edge["candidates"]
        balancing = edge["balancing"]
    except (KeyError, TypeError) as error:
        raise ConfigError("loss.edge configuration is incomplete") from error
    for location, classification, allowed in (
        (
            "loss.node.classification",
            node_classification,
            {"weighted_cross_entropy", "focal"},
        ),
        (
            "loss.edge.classification",
            edge["classification"],
            {"cross_entropy", "focal"},
        ),
    ):
        if classification.get("name") not in allowed:
            raise ConfigError(
                f"{location}.name must be one of {sorted(allowed)}"
            )
        class_weights = classification.get("class_weights")
        if (
            not isinstance(class_weights, list)
            or len(class_weights) != 2
            or any(float(value) < 0 for value in class_weights)
            or sum(float(value) for value in class_weights) <= 0
        ):
            raise ConfigError(f"{location}.class_weights must contain two non-negative values with a positive sum")
        if float(classification.get("focal_gamma", -1)) < 0:
            raise ConfigError(f"{location}.focal_gamma must be non-negative")
        curriculum = classification.get("curriculum")
        if not isinstance(curriculum, Mapping):
            raise ConfigError(f"{location}.curriculum must be a mapping")
        if not isinstance(curriculum.get("enabled"), bool):
            raise ConfigError(f"{location}.curriculum.enabled must be a boolean")
        start = float(curriculum.get("start_percent", -1))
        end = float(curriculum.get("end_percent", -1))
        if not 0 <= start <= end <= 100:
            raise ConfigError(
                f"{location}.curriculum must satisfy 0 <= start <= end <= 100"
            )
    mode = balancing.get("mode")
    if classification_name == "focal" and mode != "none":
        raise ConfigError("Focal edge classification requires balancing.mode=none")
    if mode not in {"none", "ratio_upsample"}:
        raise ConfigError("loss.edge.balancing.mode must be none or ratio_upsample")
    if mode == "ratio_upsample":
        if float(balancing.get("positive_to_negative_ratio", 0.0)) <= 0:
            raise ConfigError("ratio_upsample requires a positive class ratio")
        if float(balancing.get("tolerance", -1.0)) < 0:
            raise ConfigError("ratio_upsample requires a non-negative tolerance")
    maximum = candidates.get("max_per_graph")
    if maximum is not None:
        _positive_int(maximum, "loss.edge.candidates.max_per_graph")
    if candidates.get("positive_cap") is not None:
        raise ConfigError("loss.edge.candidates.positive_cap must remain null")
    include_unmatched = candidates.get("include_unmatched")
    if not isinstance(include_unmatched, bool):
        raise ConfigError("loss.edge.candidates.include_unmatched must be a boolean")
    threshold = float(candidates.get("unmatched_object_threshold", -1))
    if not 0 <= threshold <= 1:
        raise ConfigError(
            "loss.edge.candidates.unmatched_object_threshold must lie in [0,1]"
        )
    for name in (
        "max_active_unmatched",
        "max_unmatched_pairs_per_graph",
        "unmatched_warmup_epochs",
        "unmatched_ramp_epochs",
    ):
        _non_negative_int(candidates.get(name), f"loss.edge.candidates.{name}")
    if float(candidates.get("unmatched_weight", -1)) < 0:
        raise ConfigError("loss.edge.candidates.unmatched_weight must be non-negative")

    topology = config.get("topology")
    if not isinstance(topology, Mapping):
        raise ConfigError("topology must be a mapping")
    for name in ("betti_h0", "betti_h1"):
        topology_loss = topology.get(name)
        location = f"topology.{name}"
        if not isinstance(topology_loss, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        for flag in ("enabled", "log_only", "normalize"):
            if not isinstance(topology_loss.get(flag), bool):
                raise ConfigError(f"{location}.{flag} must be a boolean")
        for count_name in ("warmup_epochs", "ramp_epochs"):
            _non_negative_int(
                topology_loss.get(count_name), f"{location}.{count_name}"
            )
        value_names = ["weight", "diagonal_factor"]
        if name == "betti_h0":
            value_names.append("unmatched_weight")
        else:
            value_names.extend(
                ("false_positive_weight", "false_negative_weight")
            )
        for value_name in value_names:
            if float(topology_loss.get(value_name, -1)) < 0:
                raise ConfigError(f"{location}.{value_name} must be non-negative")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ConfigError("evaluation must be a mapping")
    _positive_int(evaluation.get("interval_epochs"), "evaluation.interval_epochs")
    for name in ("node_threshold", "edge_threshold"):
        value = evaluation.get(name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ConfigError(f"evaluation.{name} must be null or lie in [0,1]")
    _non_negative_int(
        evaluation.get("bn_calibration_batches"),
        "evaluation.bn_calibration_batches",
    )
    training_metrics = evaluation.get("training_metrics")
    if not isinstance(training_metrics, Mapping):
        raise ConfigError("evaluation.training_metrics must be a mapping")
    for name in ("enabled", "save_best_checkpoint"):
        if not isinstance(training_metrics.get(name), bool):
            raise ConfigError(
                f"evaluation.training_metrics.{name} must be a boolean"
            )
    metric_dataset = training_metrics.get("dataset")
    if metric_dataset not in datasets:
        raise ConfigError(
            "evaluation.training_metrics.dataset must name a configured dataset"
        )
    metric_maximum = training_metrics.get("max_samples")
    if metric_maximum is not None:
        _positive_int(
            metric_maximum, "evaluation.training_metrics.max_samples"
        )
    selection_metric = training_metrics.get("selection_metric")
    if not isinstance(selection_metric, str) or not selection_metric:
        raise ConfigError(
            "evaluation.training_metrics.selection_metric must be a non-empty string"
        )
    if training_metrics.get("selection_mode") not in {"min", "max"}:
        raise ConfigError(
            "evaluation.training_metrics.selection_mode must be min or max"
        )
    if training_metrics["save_best_checkpoint"] and not training_metrics["enabled"]:
        raise ConfigError(
            "evaluation.training_metrics.save_best_checkpoint requires enabled=true"
        )
    protocol = evaluation.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ConfigError("evaluation.protocol must be a mapping")
    f1_iou = protocol.get("f1_iou_threshold", 0.5)
    if isinstance(f1_iou, bool) or not isinstance(f1_iou, (int, float)) or not 0 < f1_iou <= 1:
        raise ConfigError("evaluation.protocol.f1_iou_threshold must lie in (0,1]")
    save_f1 = training_metrics.get("save_f1_checkpoints", False)
    if not isinstance(save_f1, bool):
        raise ConfigError("evaluation.training_metrics.save_f1_checkpoints must be boolean")
    if save_f1 and (not training_metrics["enabled"] or checkpoint["policy"] == "none"):
        raise ConfigError("F1 checkpoints require training metrics and a checkpoint policy")
    stopping = training.get("early_stopping", {})
    if not isinstance(stopping, Mapping):
        raise ConfigError("training.early_stopping must be a mapping")
    if not isinstance(stopping.get("enabled", False), bool):
        raise ConfigError("training.early_stopping.enabled must be boolean")
    _positive_int(stopping.get("patience_epochs", 50), "training.early_stopping.patience_epochs")
    minimum_epochs = _non_negative_int(
        stopping.get("min_epochs", 0), "training.early_stopping.min_epochs"
    )
    if stopping.get("enabled", False) and minimum_epochs > training["epochs"]:
        raise ConfigError("training.early_stopping.min_epochs cannot exceed training.epochs")
    delta = stopping.get("min_delta", 0.0)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ConfigError("training.early_stopping.min_delta must be finite and non-negative")
    monitor_modes = {
        "edge_mAP": "max", "node_mAP": "max", "edge_f1": "max", "node_f1": "max",
        "validation_total": "min", "beta0_absolute_error": "min",
        "beta1_absolute_error": "min", "smd": "min",
    }
    monitor = stopping.get("monitor", "edge_mAP")
    if monitor not in monitor_modes or stopping.get("mode", "max") != monitor_modes[monitor]:
        raise ConfigError("Unsupported early-stopping monitor or incorrect min/max mode")
    if stopping.get("enabled", False):
        if checkpoint["policy"] == "none":
            raise ConfigError("Early stopping requires checkpoints for the final resume state")
        if monitor != "validation_total" and not training_metrics["enabled"]:
            raise ConfigError("Metric-based early stopping requires training metrics")
    thresholds = protocol.get("iou_thresholds")
    if (
        not isinstance(thresholds, list)
        or not thresholds
        or any(not 0.0 <= float(value) <= 1.0 for value in thresholds)
        or any(float(left) >= float(right) for left, right in zip(thresholds, thresholds[1:]))
    ):
        raise ConfigError(
            "evaluation.protocol.iou_thresholds must be a strictly increasing list in [0,1]"
        )
    for name in ("max_detections", "smd_points", "smd_iterations", "folds"):
        _positive_int(protocol.get(name), f"evaluation.protocol.{name}")
    for name in ("target_node_size", "edge_half_width", "smd_epsilon"):
        if float(protocol.get(name, 0.0)) <= 0:
            raise ConfigError(f"evaluation.protocol.{name} must be positive")


def load_config(
    path: Path | str,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Load defaults, resolve ``${VARIABLE}`` references, and validate a config."""

    merged = _load_with_defaults(Path(path), set())
    resolved = _expand_environment(
        merged,
        os.environ if environment is None else environment,
        "config",
    )
    validate_config(resolved)
    return resolved


__all__ = ["ConfigError", "load_config", "validate_config"]
