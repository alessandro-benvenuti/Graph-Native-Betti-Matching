"""Cluster-only CUDA and real-checkpoint compatibility checks."""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import unittest

import torch

from configs import load_config
from models.checkpoint import checkpoint_compatibility, load_checkpoint_file
from models.ops import cuda_extension_available
from models.ops.modules import MSDeformAttn


MONAI_AVAILABLE = importlib.util.find_spec("monai") is not None
CUDA_MODEL_AVAILABLE = (
    MONAI_AVAILABLE and torch.cuda.is_available() and cuda_extension_available()
)


def _config():
    repository = Path(__file__).resolve().parents[1]
    return load_config(
        repository / "configs" / "pretrain_mixed.yaml",
        environment={
            "GNBM_OUTPUT_DIR": os.environ.get("GNBM_OUTPUT_DIR", "/tmp/gnbm"),
            "PLANTS_DATASET": os.environ.get("PLANTS_DATASET", "/plants"),
            "SYNTHETIC_MRI_DATASET": os.environ.get(
                "SYNTHETIC_MRI_DATASET", "/synthetic"
            ),
        },
    )


@unittest.skipUnless(
    CUDA_MODEL_AVAILABLE,
    "CUDA, MONAI, and the compiled deformable-attention extension are required",
)
class CudaModelTests(unittest.TestCase):
    @staticmethod
    def _small_config():
        config = copy.deepcopy(_config())
        config["model"]["encoder"]["depths"] = [1, 1, 1, 1]
        config["model"]["decoder"].update(
            hidden_dim=48,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            object_queries=6,
            relation_tokens=2,
        )
        return config

    def test_cuda_extension_matches_pytorch_fallback(self) -> None:
        torch.manual_seed(11)
        cpu_module = MSDeformAttn(48, 1, 6, 2, False).eval()
        cuda_module = MSDeformAttn(48, 1, 6, 2, True).cuda().eval()
        cuda_module.load_state_dict(cpu_module.state_dict())
        query = torch.randn((2, 3, 48))
        values = torch.randn((2, 8, 48))
        shapes = torch.tensor([[2, 2, 2]], dtype=torch.long)
        starts = torch.tensor([0], dtype=torch.long)
        references = torch.rand((2, 3, 1, 3))
        with torch.no_grad():
            expected = cpu_module(query, references, values, shapes, starts)
            observed = cuda_module(
                query.cuda(),
                references.cuda(),
                values.cuda(),
                shapes.cuda(),
                starts.cuda(),
            ).cpu()
        self.assertTrue(
            torch.allclose(observed, expected, rtol=2.0e-4, atol=2.0e-5),
            msg=f"maximum absolute difference: {(observed - expected).abs().max()}",
        )

    def test_small_relationformer_cuda_forward(self) -> None:
        from models import build_model

        config = self._small_config()
        model = build_model(config).cuda().eval()
        with torch.no_grad():
            tokens, predictions, _ = model(
                torch.randn((1, 1, 32, 32, 32), device="cuda")
            )
        self.assertEqual(tuple(tokens.shape), (1, 8, 48))
        self.assertEqual(tuple(predictions["pred_nodes"].shape), (1, 6, 6))

    def test_small_relationformer_cuda_loss_backward(self) -> None:
        from models import build_model
        from training.losses import build_criterion

        config = self._small_config()
        model = build_model(config).cuda().train()
        criterion = build_criterion(config, model).cuda()
        tokens, predictions, _ = model(
            # At 32^3 the final encoder map is 1^3. BatchNorm therefore needs
            # at least two samples while the model is in genuine train mode.
            torch.randn((2, 1, 32, 32, 32), device="cuda")
        )
        nodes = torch.tensor(
            [[0.25, 0.30, 0.35], [0.70, 0.65, 0.60]], device="cuda"
        )
        edges = torch.tensor([[0, 1]], dtype=torch.long, device="cuda")
        targets = {
            "nodes": [nodes.clone(), nodes.clone()],
            "edges": [edges.clone(), edges.clone()],
        }
        losses = criterion(tokens, predictions, targets)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIsNotNone(model.relation_embed.layers[-1].weight.grad)
        self.assertTrue(
            torch.isfinite(model.relation_embed.layers[-1].weight.grad).all()
        )

    def test_combined_focal_betti_cuda_backward(self) -> None:
        from models import build_model
        from training.losses import build_criterion

        config = self._small_config()
        config["loss"]["node"]["classification"]["name"] = "focal"
        config["loss"]["edge"]["classification"]["name"] = "focal"
        config["loss"]["edge"]["balancing"]["mode"] = "none"
        config["loss"]["edge"]["candidates"].update(
            include_unmatched=True,
            unmatched_object_threshold=0.0,
            max_active_unmatched=3,
            max_unmatched_pairs_per_graph=8,
        )
        for name in ("betti_h0", "betti_h1"):
            config["topology"][name].update(
                enabled=True, log_only=False, weight=0.1
            )
        model = build_model(config).cuda().train()
        criterion = build_criterion(config, model).cuda()
        tokens, predictions, _ = model(
            torch.randn((2, 1, 32, 32, 32), device="cuda")
        )
        nodes = torch.tensor(
            [
                [0.20, 0.25, 0.30],
                [0.50, 0.55, 0.60],
                [0.75, 0.70, 0.65],
            ],
            device="cuda",
        )
        edges = torch.tensor(
            [[0, 1], [1, 2], [0, 2]], dtype=torch.long, device="cuda"
        )
        targets = {
            "nodes": [nodes.clone(), nodes.clone()],
            "edges": [edges.clone(), edges.clone()],
        }
        losses = criterion(tokens, predictions, targets)
        losses["total"].backward()
        for name in ("total", "betti_h0", "betti_h1"):
            self.assertTrue(torch.isfinite(losses[name]))
        self.assertTrue(
            torch.isfinite(model.relation_embed.layers[-1].weight.grad).all()
        )


@unittest.skipUnless(MONAI_AVAILABLE, "MONAI is required to build RelationFormer")
class RealCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GNBM_MRI_CHECKPOINT"),
        "Set GNBM_MRI_CHECKPOINT to run the real checkpoint schema check",
    )
    def test_mri_checkpoint_matches_all_active_model_tensors(self) -> None:
        from models import build_model

        model = build_model(_config())
        checkpoint = load_checkpoint_file(os.environ["GNBM_MRI_CHECKPOINT"])
        report = checkpoint_compatibility(model, checkpoint)
        self.assertEqual(report.missing, ())
        self.assertEqual(report.unexpected, ())
        self.assertEqual(report.shape_mismatches, ())


if __name__ == "__main__":
    unittest.main()
