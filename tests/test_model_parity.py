"""CPU-safe unit tests for the cross-repository parity harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_model_forward.py"
SPEC = importlib.util.spec_from_file_location("compare_model_forward", SCRIPT)
parity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parity)


class ModelParityHarnessTests(unittest.TestCase):
    def test_both_worker_processes_complete_with_full_config_snapshot(self) -> None:
        fake_models = '''
import torch
from torch import nn


class FakeModel(nn.Module):
    def __init__(self, legacy):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.legacy = legacy

    def forward(self, inputs):
        value = inputs.mean() * self.weight
        tokens = value.reshape(1, 1, 1)
        predictions = {
            "pred_logits": tokens.repeat(1, 1, 2),
            "pred_nodes": tokens.repeat(1, 1, 6),
        }
        features = inputs * self.weight
        result = (tokens, predictions, features)
        return result + (None, None, None) if self.legacy else result


def build_model(config):
    legacy = not isinstance(config, dict)
    if legacy:
        assert config.DATA.DIM == 3
        assert config.MODEL.DECODER.ENC_LAYERS == 6
    else:
        assert config["data"]["spatial_dims"] == 3
        assert config["model"]["decoder"]["encoder_layers"] == 6
    return FakeModel(legacy)
'''
        with tempfile.TemporaryDirectory(prefix="gnbm-parity-test-") as directory:
            root = Path(directory)
            repository = root / "repository"
            models = repository / "models"
            models.mkdir(parents=True)
            (models / "__init__.py").write_text(fake_models, encoding="utf-8")

            config_path = root / "config.json"
            checkpoint_path = root / "checkpoint.pt"
            input_path = root / "input.pt"
            config_path.write_text(
                json.dumps(
                    {
                        "data": {"spatial_dims": 3},
                        "model": self._model_config(),
                    }
                ),
                encoding="utf-8",
            )
            torch.save({"net": {"weight": torch.tensor(2.0)}}, checkpoint_path)
            torch.save(torch.ones((1, 1, 2, 2, 2)), input_path)

            for worker in ("legacy", "refactored"):
                output_path = root / (worker + ".pt")
                subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--worker",
                        worker,
                        "--repository",
                        str(repository),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--config-snapshot",
                        str(config_path),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--device",
                        "cpu",
                    ],
                    check=True,
                    cwd=str(repository),
                )
                output = parity._load_torch_file(output_path)
                self.assertEqual(tuple(output["tokens"].shape), (1, 1, 1))
                self.assertEqual(float(output["tokens"].item()), 2.0)

    def test_legacy_meshgrid_wrapper_removes_supported_ij_keyword(self) -> None:
        calls = []

        def old_meshgrid(*tensors, **kwargs):
            calls.append((tensors, kwargs))
            return "mesh"

        wrapped = parity._legacy_meshgrid_wrapper(old_meshgrid)
        result = wrapped(torch.arange(2), torch.arange(3), indexing="ij")

        self.assertEqual(result, "mesh")
        self.assertEqual(calls[0][1], {})
        with self.assertRaisesRegex(ValueError, "only ij"):
            wrapped(torch.arange(2), torch.arange(3), indexing="xy")

    def test_legacy_annotations_are_postponed_without_editing_source(self) -> None:
        namespace = {}
        parity._execute_with_postponed_annotations(
            "def function(value: MissingType | None = None):\n    return value\n",
            "legacy.py",
            namespace,
        )

        self.assertEqual(
            namespace["function"].__annotations__["value"],
            "MissingType | None",
        )

    def test_worker_implementation_is_stored_in_worker_argument(self) -> None:
        arguments = parity._parser().parse_args(
            [
                "--worker",
                "legacy",
                "--repository",
                "/legacy",
                "--checkpoint",
                "/checkpoint.pt",
                "--config-snapshot",
                "/config.json",
                "--input",
                "/input.pt",
                "--output",
                "/output.pt",
            ]
        )

        self.assertEqual(arguments.worker, "legacy")

    def test_legacy_config_preserves_effective_architecture_fields(self) -> None:
        config = {
            "data": {"spatial_dims": 3},
            "model": self._model_config(),
        }
        legacy = parity.legacy_config(config)
        self.assertEqual(legacy.MODEL.ENCODER.STRIDES, [1, 2, 2, 2])
        self.assertEqual(legacy.MODEL.DECODER.ENC_LAYERS, 6)
        self.assertFalse(legacy.DATA.MIXED)

    @staticmethod
    def _model_config():
        return {
            "encoder": {
                "input_channels": 1,
                "depths": [4, 4, 16, 4],
                "strides": [1, 2, 2, 2],
            },
            "decoder": {
                "hidden_dim": 552,
                "attention_heads": 6,
                "encoder_layers": 6,
                "decoder_layers": 4,
                "feedforward_dim": 1280,
                "dropout": 0.0,
                "activation": "relu",
                "feature_levels": 1,
                "decoder_points": 4,
                "encoder_points": 4,
                "object_queries": 120,
                "relation_tokens": 2,
                "dummy_tokens": 0,
                "relation_attention": True,
                "use_cuda_extension": True,
            },
        }

    def test_comparison_reports_numerical_and_shape_differences(self) -> None:
        reference = {key: torch.ones((2, 3)) for key in parity.OUTPUT_KEYS}
        observed = {key: value.clone() for key, value in reference.items()}
        observed["pred_nodes"][0, 0] += 1.0
        observed["projected_features"] = torch.ones((3, 2))

        results = parity.compare_outputs(reference, observed, rtol=0.0, atol=0.0)

        self.assertTrue(results["tokens"]["compatible"])
        self.assertFalse(results["pred_nodes"]["compatible"])
        self.assertEqual(results["pred_nodes"]["max_abs"], 1.0)
        self.assertFalse(results["projected_features"]["compatible"])
        self.assertIn("shape", results["projected_features"]["reason"])

    def test_loss_keys_can_be_compared_with_the_same_reporter(self) -> None:
        keys = ("loss/class", "loss/total")
        reference = {key: torch.tensor(2.0) for key in keys}
        observed = {key: value.clone() for key, value in reference.items()}

        results = parity.compare_outputs(
            reference, observed, rtol=0.0, atol=0.0, keys=keys
        )

        self.assertEqual(set(results), set(keys))
        self.assertTrue(all(item["compatible"] for item in results.values()))


if __name__ == "__main__":
    unittest.main()
