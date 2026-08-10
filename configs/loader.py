"""Small, dependency-light loader for the repository YAML configuration schema."""

from __future__ import annotations

import copy
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

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ConfigError("data must be a mapping")
    image_size = data.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 3:
        raise ConfigError("data.image_size must contain three spatial dimensions")
    for index, size in enumerate(image_size):
        _positive_int(size, f"data.image_size[{index}]")
    _positive_int(data.get("batch_size"), "data.batch_size")

    datasets = data.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ConfigError("data.datasets must contain at least one dataset")
    valid_names = {"plants", "synthetic_mri"}
    valid_roles = {"source", "target"}
    target_count = 0
    source_count = 0
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
        source_count += int(role == "source")
        target_count += int(role == "target")
        root = dataset.get("root")
        if not isinstance(root, str) or not root.strip():
            raise ConfigError(f"{location}.root must be a non-empty path")
        _positive_int(dataset.get("train_samples"), f"{location}.train_samples")
        if "validation_samples" in dataset:
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

    if target_count == 0:
        raise ConfigError("At least one target dataset is required")
    domain_adaptation = config.get("domain_adaptation", {})
    if not isinstance(domain_adaptation, Mapping):
        raise ConfigError("domain_adaptation must be a mapping")
    if bool(domain_adaptation.get("enabled")) and source_count == 0:
        raise ConfigError("Domain adaptation requires at least one source dataset")

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
    if (
        not isinstance(clamp_range, list)
        or len(clamp_range) != 2
        or float(clamp_range[1]) <= float(clamp_range[0])
    ):
        raise ConfigError("SyntheticMRI intensity clamp is invalid")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ConfigError("training must be a mapping")
    _positive_int(training.get("epochs"), "training.epochs")

    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        raise ConfigError("loss must be a mapping")
    try:
        edge = loss["edge"]
        classification_name = edge["classification"]["name"]
        candidates = edge["candidates"]
        balancing = edge["balancing"]
    except (KeyError, TypeError) as error:
        raise ConfigError("loss.edge configuration is incomplete") from error
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
