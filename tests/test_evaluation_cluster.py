"""Optional Magnolia test for checkpoint + real-data evaluation."""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import torch

from configs import load_config
from data.loaders import build_evaluation_loader
from models.checkpoint import load_legacy_model_checkpoint
from models.ops import cuda_extension_available
from training.evaluation import evaluate_model


AVAILABLE = (
    importlib.util.find_spec("monai") is not None
    and importlib.util.find_spec("matplotlib") is not None
    and torch.cuda.is_available()
    and cuda_extension_available()
    and bool(os.environ.get("SYNTHETIC_MRI_DATASET"))
    and bool(os.environ.get("GNBM_MRI_CHECKPOINT"))
)


@unittest.skipUnless(
    AVAILABLE,
    "Requires MONAI, Matplotlib, CUDA, the dataset, and GNBM_MRI_CHECKPOINT",
)
class RealEvaluationPipelineTests(unittest.TestCase):
    def test_two_real_mri_samples_run_inference_metrics_and_export(self):
        from models import build_model

        repository = Path(__file__).resolve().parents[1]
        config = load_config(
            repository / "configs" / "finetune_synthetic_mri.yaml",
            environment={
                "GNBM_OUTPUT_DIR": os.environ.get("GNBM_OUTPUT_DIR", "/tmp/gnbm"),
                "SYNTHETIC_MRI_DATASET": os.environ["SYNTHETIC_MRI_DATASET"],
            },
        )
        config = copy.deepcopy(config)
        config["data"]["batch_size"] = 1
        config["runtime"]["workers"] = 0
        loader = build_evaluation_loader(
            config,
            dataset_name="synthetic_mri",
            split="val",
            max_samples=2,
        )
        model = build_model(config).cuda()
        load_legacy_model_checkpoint(
            model, os.environ["GNBM_MRI_CHECKPOINT"], map_location="cpu"
        )
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = evaluate_model(
                model,
                loader,
                config,
                torch.device("cuda"),
                output_dir=Path(directory),
                max_visualizations=1,
            )
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(len(rows), 2)
            self.assertTrue((Path(directory) / "summary.json").is_file())
            self.assertTrue(
                (Path(directory) / "plots" / "sample_000000.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
