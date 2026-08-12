#!/usr/bin/env python3
"""Compare legacy and refactored RelationFormer outputs on one fixed volume.

The two repositories are evaluated in separate Python processes. This avoids
collisions between their identically named ``models`` packages and compiled
``MultiScaleDeformableAttention`` extensions.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Mapping

import torch


OUTPUT_KEYS = ("tokens", "pred_logits", "pred_nodes", "projected_features")
LOSS_KEYS = ("loss/class", "loss/nodes", "loss/boxes", "loss/cardinality", "loss/edges", "loss/total")
INFERENCE_KEYS = (
    "inference/nodes",
    "inference/boxes",
    "inference/node_scores",
    "inference/edges",
    "inference/edge_scores",
)
REMOVED_PREFIXES = (
    "backbone_domain_discriminator.",
    "instance_domain_discriminator.",
)


def _namespace(value):
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def legacy_config(config: Mapping) -> SimpleNamespace:
    """Translate the explicit model schema to fields read by legacy builders."""

    model_config = config["model"]
    encoder = model_config["encoder"]
    decoder = model_config["decoder"]
    matcher = model_config.get("matcher", {"class_cost": 2.0, "node_cost": 5.0})
    loss = config.get(
        "loss",
        {
            "enabled": ["boxes", "class", "cardinality", "nodes", "edges"],
            "weights": {"boxes": 3.0, "class": 4.0, "cardinality": 0.8, "nodes": 2.0, "edges": 6.0},
            "edge": {
                "candidates": {"max_per_graph": 9999},
                "balancing": {"positive_to_negative_ratio": 0.15, "tolerance": 0.02},
            },
        },
    )
    return _namespace(
        {
            "DATA": {"DIM": 3, "MIXED": False},
            "MODEL": {
                "ENCODER": {
                    "IN_CHANS": encoder["input_channels"],
                    "DEPTHS": encoder["depths"],
                    "STRIDES": encoder["strides"],
                },
                "DECODER": {
                    "HIDDEN_DIM": decoder["hidden_dim"],
                    "NHEADS": decoder["attention_heads"],
                    "ENC_LAYERS": decoder["encoder_layers"],
                    "DEC_LAYERS": decoder["decoder_layers"],
                    "DIM_FEEDFORWARD": decoder["feedforward_dim"],
                    "DROPOUT": decoder["dropout"],
                    "ACTIVATION": decoder["activation"],
                    "NUM_FEATURE_LEVELS": decoder["feature_levels"],
                    "DEC_N_POINTS": decoder["decoder_points"],
                    "ENC_N_POINTS": decoder["encoder_points"],
                    "OBJ_TOKEN": decoder["object_queries"],
                    "RLN_TOKEN": decoder["relation_tokens"],
                    "DUMMY_TOKEN": decoder["dummy_tokens"],
                    "RLN_ATTN": decoder["relation_attention"],
                },
                "MATCHER": {
                    "NAME": "Hungarian",
                    "C_CLASS": matcher["class_cost"],
                    "C_NODE": matcher["node_cost"],
                },
            },
            "TRAIN": {
                "LOSSES": [
                    "cards" if name == "cardinality" else name
                    for name in loss["enabled"]
                ],
                "W_BBOX": loss["weights"]["boxes"],
                "W_CLASS": loss["weights"]["class"],
                "W_CARD": loss["weights"]["cardinality"],
                "W_NODE": loss["weights"]["nodes"],
                "W_EDGE": loss["weights"]["edges"],
                "W_DOMAIN": 0.0,
                "EDGE_SAMPLE_RATIO": loss["edge"]["balancing"]["positive_to_negative_ratio"],
                "EDGE_SAMPLE_RATIO_INTERVAL": loss["edge"]["balancing"]["tolerance"],
                "HARD_NEGATIVE_MINING": False,
            },
        }
    )


def _load_torch_file(path: Path):
    options = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "mmap" in parameters:
        options["mmap"] = True
    if "weights_only" in parameters:
        options["weights_only"] = False
    return torch.load(str(path), **options)


def _extract_state(checkpoint) -> dict:
    candidate = checkpoint
    if isinstance(checkpoint, Mapping):
        for container in ("net", "model", "state_dict"):
            if isinstance(checkpoint.get(container), Mapping):
                candidate = checkpoint[container]
                break
    if not isinstance(candidate, Mapping):
        raise ValueError("checkpoint does not contain a model state mapping")

    state = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if not any(key.startswith(prefix) for prefix in REMOVED_PREFIXES):
            state[key] = value
    return state


def _canonical_inference_tensor(name: str, value, shape):
    """Normalize legacy NumPy/Tensor graph outputs for parity comparison."""

    tensor = torch.as_tensor(value).reshape(shape).detach().cpu()
    return tensor.long() if name == "edges" else tensor.float()


def _legacy_meshgrid_wrapper(original_meshgrid):
    """Adapt an old ij-default meshgrid implementation to the newer API."""

    def compatible_meshgrid(*tensors, **kwargs):
        indexing = kwargs.pop("indexing", None)
        if indexing not in (None, "ij"):
            raise ValueError(
                "legacy torch.meshgrid supports only ij indexing, not {}".format(
                    indexing
                )
            )
        return original_meshgrid(*tensors, **kwargs)

    return compatible_meshgrid


def _install_legacy_torch_compatibility() -> None:
    """Bridge API differences without changing the legacy tensor operations."""

    probe = torch.arange(2, dtype=torch.float32)
    try:
        torch.div(probe, 2, rounding_mode="trunc")
    except TypeError:
        original_div = torch.div

        def compatible_div(input_tensor, other, *args, **kwargs):
            rounding_mode = kwargs.pop("rounding_mode", None)
            result = original_div(input_tensor, other, *args, **kwargs)
            if rounding_mode == "trunc":
                return torch.trunc(result)
            if rounding_mode == "floor":
                return torch.floor(result)
            if rounding_mode is not None:
                raise ValueError("unsupported rounding_mode: {}".format(rounding_mode))
            return result

        torch.div = compatible_div

    try:
        torch.meshgrid(probe, probe, indexing="ij")
    except TypeError:
        torch.meshgrid = _legacy_meshgrid_wrapper(torch.meshgrid)


def _prepare_repository_imports(repository: Path) -> None:
    repository = repository.resolve()
    os.chdir(str(repository))
    sys.path.insert(0, str(repository))


def _execute_with_postponed_annotations(source: str, filename: str, namespace: dict):
    """Execute legacy source while leaving its type annotations unevaluated."""

    compatible_source = "from __future__ import annotations\n" + source
    exec(compile(compatible_source, filename, "exec"), namespace)


def _load_legacy_losses_module(repository: Path):
    """Import legacy losses on Python 3.8 without modifying the old checkout."""

    import training

    path = repository / "training" / "losses.py"
    module = ModuleType("training.losses")
    module.__file__ = str(path)
    module.__package__ = "training"
    module.__loader__ = None
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module.__name__] = module
    _execute_with_postponed_annotations(
        path.read_text(encoding="utf-8"), str(path), module.__dict__
    )
    setattr(training, "losses", module)
    return module


def _install_unused_legacy_nms_stub() -> None:
    """Avoid importing legacy NMS when parity explicitly disables NMS.

    Magnolia's old PyTorch does not expose ``torch.cuda.amp.autocast``, which
    the legacy NMS module imports eagerly. ``relation_infer(...,
    apply_nms=False)`` never calls NMS, so a fail-fast stub preserves the
    evaluated path without changing the original checkout.
    """

    import boxes
    from boxes import box_ops

    module = ModuleType("boxes.nms")

    def disabled_nms(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("Legacy NMS is unavailable in no-NMS parity mode")

    module.nms = disabled_nms
    module.box_ops = box_ops
    sys.modules[module.__name__] = module
    setattr(boxes, "nms", module)


def _run_worker(args: argparse.Namespace) -> None:
    repository = Path(args.repository).resolve()
    _prepare_repository_imports(repository)
    with Path(args.config_snapshot).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if args.worker == "legacy":
        _install_legacy_torch_compatibility()
        from models import build_model

        model = build_model(legacy_config(config))
    else:
        from models import build_model

        model = build_model(config)

    checkpoint = _load_torch_file(Path(args.checkpoint))
    state = _extract_state(checkpoint)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()

    device = torch.device(args.device)
    model.to(device).eval()
    inputs = _load_torch_file(Path(args.input)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        result = model(inputs)
    if args.worker == "legacy":
        tokens, predictions, projected_features = result[:3]
    else:
        tokens, predictions, projected_features = result

    outputs = {
        "tokens": tokens.detach().cpu(),
        "pred_logits": predictions["pred_logits"].detach().cpu(),
        "pred_nodes": predictions["pred_nodes"].detach().cpu(),
        "projected_features": projected_features.detach().cpu(),
    }
    if args.compare_inference:
        decoder = config["model"]["decoder"]
        if args.worker == "legacy":
            _install_unused_legacy_nms_stub()
            from training.inference import relation_infer

            graph_batch = relation_infer(
                tokens,
                predictions,
                model,
                decoder["object_queries"],
                decoder["relation_tokens"],
                apply_nms=False,
            )
            graph = {
                "nodes": graph_batch["pred_nodes"][0],
                "boxes": graph_batch["pred_boxes"][0],
                "node_scores": graph_batch["pred_boxes_score"][0],
                "edges": graph_batch["pred_rels"][0],
                "edge_scores": graph_batch["pred_rels_score"][0],
            }
        else:
            from training.evaluation.inference import infer_graphs

            graph = infer_graphs(
                tokens,
                predictions,
                model.relation_embed,
                object_queries=decoder["object_queries"],
                relation_tokens=decoder["relation_tokens"],
            )[0]
        shapes = {
            "nodes": (-1, 3),
            "boxes": (-1, 6),
            "node_scores": (-1,),
            "edges": (-1, 2),
            "edge_scores": (-1,),
        }
        for name, shape in shapes.items():
            # Legacy relation_infer creates an empty edge tensor without a
            # dtype, so the no-edge case is float while non-empty edge indices
            # are integer. Canonical graph edge indices are always int64.
            outputs["inference/" + name] = _canonical_inference_tensor(
                name, graph[name], shape
            )
    if args.compare_losses:
        targets = _load_torch_file(Path(args.targets))
        targets = {
            "nodes": [value.to(device) for value in targets["nodes"]],
            "edges": [value.to(device) for value in targets["edges"]],
            "domains": targets["domains"].to(device),
        }
        torch.manual_seed(int(args.seed))
        if args.worker == "legacy":
            from models.matcher import build_matcher

            legacy_losses = _load_legacy_losses_module(repository)

            translated = legacy_config(config)
            criterion = legacy_losses.SetCriterion(
                translated,
                build_matcher(translated, dims=3),
                model.relation_embed,
                dims=3,
                num_edge_samples=config["loss"]["edge"]["candidates"]["max_per_graph"],
                edge_sampling_mode=legacy_losses.EDGE_SAMPLING_MODE.UP,
            ).to(device)
            losses = criterion(tokens, predictions, targets, None, None)
        else:
            from training.losses import build_criterion

            criterion = build_criterion(config, model).to(device)
            losses = criterion(tokens, predictions, targets)
        legacy_to_new = {"cards": "cardinality"}
        for name in ("class", "nodes", "boxes", "cards", "cardinality", "edges", "total"):
            if name in losses:
                outputs["loss/" + legacy_to_new.get(name, name)] = losses[name].detach().cpu()
    torch.save(outputs, str(Path(args.output)))


def compare_outputs(reference: Mapping, observed: Mapping, rtol: float, atol: float, keys=OUTPUT_KEYS):
    results = {}
    for key in keys:
        expected = reference[key]
        actual = observed[key]
        if expected.shape != actual.shape:
            results[key] = {
                "compatible": False,
                "reason": "shape {} != {}".format(tuple(expected.shape), tuple(actual.shape)),
            }
            continue
        difference = (expected - actual).abs()
        compatible = (
            torch.allclose(expected, actual, rtol=rtol, atol=atol)
            if expected.is_floating_point()
            else torch.equal(expected, actual)
        )
        results[key] = {
            "compatible": bool(compatible),
            "max_abs": float(difference.max()) if difference.numel() else 0.0,
            "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        }
    return results


def _load_config_snapshot(repository: Path, config_path: Path) -> Mapping:
    _prepare_repository_imports(repository)
    from configs import load_config

    config = load_config(
        config_path,
        environment={
            "GNBM_OUTPUT_DIR": "/tmp/gnbm-forward-parity",
            "PLANTS_DATASET": "/unused/plants",
            "SYNTHETIC_MRI_DATASET": "/unused/synthetic-mri",
        },
    )
    return config


def _run_comparison(args: argparse.Namespace) -> int:
    repository = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    if not (legacy_root / "models" / "relationformer.py").is_file():
        raise FileNotFoundError("legacy 3d repository not found at {}".format(legacy_root))
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint))

    with tempfile.TemporaryDirectory(prefix="gnbm-forward-parity-") as directory:
        artifacts = Path(directory)
        input_path = artifacts / "input.pt"
        target_path = artifacts / "targets.pt"
        config_snapshot_path = artifacts / "config.json"
        legacy_output = artifacts / "legacy.pt"
        refactored_output = artifacts / "refactored.pt"

        config = _load_config_snapshot(repository, config_path)
        with config_snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        elements = args.input_size ** 3
        inputs = torch.linspace(-0.5, 0.5, steps=elements, dtype=torch.float32)
        torch.save(inputs.reshape(1, 1, args.input_size, args.input_size, args.input_size), str(input_path))
        torch.save(
            {
                "nodes": [
                    torch.tensor(
                        [[0.15, 0.20, 0.25], [0.40, 0.35, 0.55], [0.70, 0.65, 0.45], [0.82, 0.78, 0.74]],
                        dtype=torch.float32,
                    )
                ],
                "edges": [torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)],
                "domains": torch.zeros(1, dtype=torch.long),
            },
            str(target_path),
        )

        for implementation, root, output in (
            ("legacy", legacy_root, legacy_output),
            ("refactored", repository, refactored_output),
        ):
            print("Running {} model...".format(implementation), flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                implementation,
                "--repository",
                str(root),
                "--checkpoint",
                str(checkpoint),
                "--config-snapshot",
                str(config_snapshot_path),
                "--input",
                str(input_path),
                "--output",
                str(output),
                "--targets",
                str(target_path),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
            if args.compare_losses:
                command.append("--compare-losses")
            if args.compare_inference:
                command.append("--compare-inference")
            subprocess.run(command, check=True, cwd=str(root))

        reference = _load_torch_file(legacy_output)
        observed = _load_torch_file(refactored_output)
        keys = OUTPUT_KEYS
        if args.compare_losses:
            keys += LOSS_KEYS
        if args.compare_inference:
            keys += INFERENCE_KEYS
        results = compare_outputs(reference, observed, args.rtol, args.atol, keys=keys)
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0 if all(item["compatible"] for item in results.values()) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", help="path to the original 3d repository")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "pretrain_mixed.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--compare-losses", action="store_true")
    parser.add_argument("--compare-inference", action="store_true")
    parser.add_argument("--seed", type=int, default=364505)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--worker", choices=("legacy", "refactored"), help=argparse.SUPPRESS)
    parser.add_argument("--repository", help=argparse.SUPPRESS)
    parser.add_argument("--config-snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--input", help=argparse.SUPPRESS)
    parser.add_argument("--targets", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        _run_worker(args)
        return 0
    if not args.legacy_root:
        raise SystemExit("--legacy-root is required")
    return _run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
