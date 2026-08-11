"""CPU-safe tests for the ported 3D model components."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from configs import load_config
from models.checkpoint import (
    checkpoint_compatibility,
    extract_model_state,
    load_checkpoint_file,
)
from models.deformable_transformer import DeformableTransformer
from models.ops.modules import MSDeformAttn
from models.position_encoding import PositionEmbeddingSine3D
from models.relationformer import RelationFormer


MONAI_AVAILABLE = importlib.util.find_spec("monai") is not None


def _baseline_config():
    repository = Path(__file__).resolve().parents[1]
    return load_config(
        repository / "configs" / "pretrain_mixed.yaml",
        environment={
            "GNBM_OUTPUT_DIR": "/outputs",
            "PLANTS_DATASET": "/plants",
            "SYNTHETIC_MRI_DATASET": "/synthetic",
        },
    )


class ModelConfigurationTests(unittest.TestCase):
    def test_config_records_effective_legacy_architecture(self) -> None:
        config = _baseline_config()
        self.assertEqual(config["model"]["encoder"]["strides"], [1, 2, 2, 2])
        self.assertEqual(config["model"]["decoder"]["encoder_layers"], 6)
        self.assertEqual(config["model"]["decoder"]["decoder_layers"], 4)
        self.assertEqual(config["model"]["decoder"]["hidden_dim"], 552)
        self.assertEqual(config["model"]["decoder"]["relation_tokens"], 2)


class PositionEncodingTests(unittest.TestCase):
    def test_sine_encoding_shape_dtype_and_determinism(self) -> None:
        encoding = PositionEmbeddingSine3D(channels=50)
        mask = torch.zeros((2, 3, 4, 5), dtype=torch.bool)
        first = encoding(mask)
        second = encoding(mask)
        self.assertEqual(tuple(first.shape), (2, 50, 3, 4, 5))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_encoding_rejects_non_boolean_or_non_volume_masks(self) -> None:
        encoding = PositionEmbeddingSine3D(channels=12)
        with self.assertRaisesRegex(ValueError, "boolean"):
            encoding(torch.zeros((1, 2, 3, 4)))
        with self.assertRaisesRegex(ValueError, "shape"):
            encoding(torch.zeros((1, 2, 3), dtype=torch.bool))


class DeformableAttentionTests(unittest.TestCase):
    def test_cpu_fallback_has_expected_shape_and_gradients(self) -> None:
        torch.manual_seed(7)
        attention = MSDeformAttn(
            d_model=48,
            n_levels=1,
            n_heads=6,
            n_points=2,
            use_cuda_extension=False,
        )
        query = torch.randn((2, 3, 48), requires_grad=True)
        values = torch.randn((2, 8, 48), requires_grad=True)
        spatial_shapes = torch.tensor([[2, 2, 2]], dtype=torch.long)
        level_starts = torch.tensor([0], dtype=torch.long)
        reference_points = torch.rand((2, 3, 1, 3))

        output = attention(
            query,
            reference_points,
            values,
            spatial_shapes,
            level_starts,
        )
        self.assertEqual(tuple(output.shape), (2, 3, 48))
        output.square().mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_cuda_path_rejects_cpu_tensors_with_actionable_message(self) -> None:
        attention = MSDeformAttn(
            d_model=48,
            n_levels=1,
            n_heads=6,
            n_points=1,
            use_cuda_extension=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA tensors"):
            attention(
                torch.randn((1, 2, 48)),
                torch.rand((1, 2, 1, 3)),
                torch.randn((1, 8, 48)),
                torch.tensor([[2, 2, 2]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_checkpoint_path_is_converted_for_legacy_pytorch(self) -> None:
        with patch(
            "models.checkpoint.torch.load",
            return_value={"net": {"weight": torch.ones(1)}},
        ) as mocked_load:
            load_checkpoint_file(Path("checkpoint.pt"))

        self.assertIsInstance(mocked_load.call_args.args[0], str)

    def test_wrapper_prefixes_and_removed_domain_keys_are_handled_explicitly(self) -> None:
        model = nn.Sequential(nn.Linear(3, 2))
        model_state = model.state_dict()
        checkpoint = {
            "net": {
                **{f"module.{key}": value.clone() for key, value in model_state.items()},
                "module.backbone_domain_discriminator.net.0.weight": torch.ones(2, 2),
            }
        }
        normalized = extract_model_state(checkpoint)
        self.assertIn("0.weight", normalized)
        report = checkpoint_compatibility(model, checkpoint)
        self.assertTrue(report.compatible)
        self.assertEqual(
            report.ignored_removed,
            ("backbone_domain_discriminator.net.0.weight",),
        )

    def test_active_shape_mismatch_is_not_silently_ignored(self) -> None:
        model = nn.Sequential(nn.Linear(3, 2))
        checkpoint = {
            "net": {
                "0.weight": torch.zeros((3, 3)),
                "0.bias": torch.zeros(2),
            }
        }
        report = checkpoint_compatibility(model, checkpoint)
        self.assertFalse(report.compatible)
        self.assertEqual(len(report.shape_mismatches), 1)


class RelationFormerHeadTests(unittest.TestCase):
    class _Encoder(nn.Module):
        num_features = 16

        def forward(self, samples):
            return torch.nn.functional.avg_pool3d(samples, kernel_size=4).repeat(
                1, self.num_features, 1, 1, 1
            )

    def test_cpu_transformer_and_prediction_contract(self) -> None:
        config = copy.deepcopy(_baseline_config())
        decoder_config = config["model"]["decoder"]
        decoder_config.update(
            hidden_dim=48,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            object_queries=6,
            relation_tokens=2,
            use_cuda_extension=False,
        )
        transformer = DeformableTransformer(
            d_model=48,
            nhead=6,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            num_feature_levels=1,
            dec_n_points=4,
            enc_n_points=4,
            rln_attn=True,
            use_cuda_extension=False,
        )
        model = RelationFormer(self._Encoder(), transformer, config).eval()
        with torch.no_grad():
            tokens, predictions, features = model(torch.randn((1, 1, 16, 16, 16)))

        self.assertEqual(tuple(tokens.shape), (1, 8, 48))
        self.assertEqual(tuple(predictions["pred_logits"].shape), (1, 6, 2))
        self.assertEqual(tuple(predictions["pred_nodes"].shape), (1, 6, 6))
        self.assertEqual(tuple(features.shape), (1, 48, 4, 4, 4))
        self.assertGreaterEqual(float(predictions["pred_nodes"].min()), 0.0)
        self.assertLessEqual(float(predictions["pred_nodes"].max()), 1.0)
        self.assertIn("relation_embed.layers.2.bias", model.state_dict())
        self.assertFalse(
            any("domain_discriminator" in key for key in model.state_dict())
        )


@unittest.skipUnless(MONAI_AVAILABLE, "MONAI is required for the SE-ResNet model")
class RelationFormerForwardTests(unittest.TestCase):
    def test_small_cpu_model_forward_contract(self) -> None:
        from models import build_model

        config = copy.deepcopy(_baseline_config())
        config["model"]["encoder"]["depths"] = [1, 1, 1, 1]
        decoder = config["model"]["decoder"]
        decoder.update(
            hidden_dim=48,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            object_queries=6,
            relation_tokens=2,
            use_cuda_extension=False,
        )
        model = build_model(config).eval()
        with torch.no_grad():
            tokens, predictions, features = model(torch.randn((1, 1, 32, 32, 32)))

        self.assertEqual(tuple(tokens.shape), (1, 8, 48))
        self.assertEqual(tuple(predictions["pred_logits"].shape), (1, 6, 2))
        self.assertEqual(tuple(predictions["pred_nodes"].shape), (1, 6, 6))
        self.assertEqual(features.shape[1], 48)
        self.assertGreaterEqual(float(predictions["pred_nodes"].min()), 0.0)
        self.assertLessEqual(float(predictions["pred_nodes"].max()), 1.0)
        keys = model.state_dict()
        self.assertIn("encoder.layer0.conv1.weight", keys)
        self.assertIn("decoder.encoder.layers.0.self_attn.sampling_offsets.weight", keys)
        self.assertIn("relation_embed.layers.2.bias", keys)
        self.assertFalse(any("domain_discriminator" in key for key in keys))


if __name__ == "__main__":
    unittest.main()
