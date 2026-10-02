"""Unit and integration coverage for staged control-relative Betti activation."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_node_edge_betti import create_storage
from scripts.staged_betti_activation import (
    ABSOLUTE_OBJECTIVES,
    DELTA_OBJECTIVES,
    _deltas,
    _merge_epoch_records,
    _trial_config,
    load_staged,
)


SMOKE = ROOT / "configs/experiments/staged_betti_activation/smoke.yaml"
ENVIRONMENT = {
    "GNBM_OUTPUT_DIR": "/unused",
    "PLANTS_DATASET": "/plants",
    "SYNTHETIC_MRI_DATASET": "/synthetic",
}


class StagedBettiUnitTests(unittest.TestCase):
    def test_two_gpu_execution_preserves_global_batch(self):
        environment = {
            **ENVIRONMENT,
            "GNBM_STAGED_GPUS": "2",
            "GNBM_STAGED_GLOBAL_BATCH_SIZE": "32",
        }
        with patch.dict(os.environ, environment, clear=False):
            base, _ = load_staged(SMOKE, environment)
        self.assertTrue(base["runtime"]["distributed"])
        self.assertEqual(base["data"]["batch_size"], 16)

    def test_metric_histories_are_epoch_sorted_and_later_phases_win(self):
        merged = _merge_epoch_records(
            [{"epoch": 10, "node_mAP": 0.1}, {"epoch": 5, "node_mAP": 0.05}],
            [{"epoch": 10, "node_mAP": 0.2}, {"epoch": 15, "node_mAP": 0.3}],
        )
        self.assertEqual([row["epoch"] for row in merged], [5, 10, 15])
        self.assertEqual(merged[1]["node_mAP"], 0.2)

    def test_protocol_and_delta_signs_are_fixed(self):
        base, staged = load_staged(SMOKE, ENVIRONMENT)
        self.assertEqual(
            DELTA_OBJECTIVES,
            (
                ("delta_node_mAP", "maximize"),
                ("delta_edge_mAP", "maximize"),
                ("delta_beta0_error", "minimize"),
                ("delta_beta1_error", "minimize"),
            ),
        )
        observed = _deltas(
            {
                "node_mAP": 0.8, "edge_mAP": 0.7,
                "beta0_absolute_error": 1.0, "beta1_absolute_error": 2.0,
            },
            {
                "node_mAP": 0.7, "edge_mAP": 0.75,
                "beta0_absolute_error": 1.5, "beta1_absolute_error": 1.0,
            },
        )
        self.assertAlmostEqual(observed["delta_node_mAP"], 0.1)
        self.assertAlmostEqual(observed["delta_edge_mAP"], -0.05)
        self.assertAlmostEqual(observed["delta_beta0_error"], -0.5)
        self.assertAlmostEqual(observed["delta_beta1_error"], 1.0)
        self.assertEqual(staged["branch_epochs"], [2, 4, 6])
        self.assertEqual(base["loss"]["node"]["classification"]["name"], "focal")
        self.assertEqual(base["loss"]["edge"]["classification"]["name"], "cross_entropy")

    def test_trial_configuration_is_branch_relative_and_narrow(self):
        base, staged = load_staged(SMOKE, ENVIRONMENT)
        reference = {
            "branch_checkpoint": "/prefix/checkpoints/epoch_0004.pt",
            "metrics": {name: 1.0 for name, _ in ABSOLUTE_OBJECTIVES},
        }
        params = {
            "topology.betti_h0.weight": 0.001,
            "topology.betti_h1.weight": 0.01,
            "topology.betti_h1.false_positive_weight": 0.1,
            "betti_ramp_epochs": 2,
        }
        config = _trial_config(base, staged, 4, 7, params, reference)
        self.assertEqual(config["training"]["stop_after_epoch"], 6)
        for name in ("betti_h0", "betti_h1"):
            self.assertEqual(config["topology"][name]["activation_epoch"], 4)
            self.assertEqual(config["topology"][name]["warmup_epochs"], 0)
            self.assertEqual(config["topology"][name]["ramp_epochs"], 2)
        self.assertNotIn("betti_warmup_epochs", staged["search_space"])
        self.assertNotIn("edge_loss", staged["search_space"])
        self.assertNotIn("model.matcher.structure_weight", staged["search_space"])


class StagedBettiIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            import optuna  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            self.skipTest(str(error))
        self.environment = dict(os.environ)
        self.environment.update(ENVIRONMENT)
        self.environment["WANDB_MODE"] = "offline"

    def command(self, mode, output, initial, *extra):
        return [
            sys.executable,
            str(ROOT / "scripts/staged_betti_activation.py"),
            mode,
            "--config", str(SMOKE),
            "--output", str(output),
            "--initial-weights", str(initial),
            *map(str, extra),
            "--poll-interval", "0.02",
            "--train-command", sys.executable,
            str(ROOT / "tests/fixtures/fake_optuna_train.py"),
            "--config", "{config}",
            "--output-dir", "{output}",
            "--run-name", "{run_name}",
        ]

    def run_checked(self, command, **kwargs):
        return subprocess.run(
            command, cwd=ROOT, env=self.environment, check=True, **kwargs
        )

    def test_tiny_three_branch_smoke_interrupt_resume_and_final(self):
        import optuna

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "staged"
            initial = root / "initial.pt"
            initial.write_bytes(b"model-only-initialization")

            self.run_checked(self.command("prefix", output, initial))
            self.run_checked(self.command("prepare", output, initial))
            for epoch in (2, 4, 6, 8):
                self.assertTrue(
                    (output / f"shared-prefix/checkpoints/epoch_{epoch:04d}.pt").is_file()
                )

            slow_environment = dict(self.environment)
            slow_environment["FAKE_TRAIN_SLEEP_PER_EPOCH"] = "5"
            interrupted = subprocess.Popen(
                self.command(
                    "screen", output, initial,
                    "--branch", "2", "--worker-trials", "1",
                ),
                cwd=ROOT,
                env=slow_environment,
            )
            latest = output / "screening/branch_0002/runs/trial_0000/models/latest_checkpoint.pt"
            deadline = time.monotonic() + 15
            while not latest.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(latest.is_file())
            interrupted.send_signal(signal.SIGTERM)
            self.assertNotEqual(interrupted.wait(timeout=20), 0)

            branch_dir = output / "screening/branch_0002"
            study = optuna.load_study(
                study_name="staged-betti-activation-e0002",
                storage=create_storage(optuna, branch_dir, "journal"),
            )
            self.assertEqual(study.trials[0].state.name, "RUNNING")
            params_before = dict(study.trials[0].params)

            self.run_checked(self.command(
                "screen", output, initial,
                "--branch", "2", "--worker-trials", "1", "--resume-running",
            ))
            self.run_checked(self.command("screen", output, initial, "--branch", "2"))
            for branch in (4, 6):
                self.run_checked(self.command("screen", output, initial, "--branch", str(branch)))

            resumed = optuna.load_study(
                study_name="staged-betti-activation-e0002",
                storage=create_storage(optuna, branch_dir, "journal"),
            )
            self.assertEqual([trial.number for trial in resumed.trials], [0, 1])
            self.assertEqual(resumed.trials[0].state.name, "COMPLETE")
            self.assertEqual(resumed.trials[0].user_attrs["resume_count"], 1)
            self.assertEqual(resumed.trials[0].params, params_before)
            self.assertEqual(len(set(resumed.trials[0].params)), 4)
            config = yaml.safe_load(Path(resumed.trials[0].user_attrs["config_path"]).read_text())
            self.assertEqual(config["topology"]["betti_h0"]["activation_epoch"], 2)
            provenance = json.loads(
                (Path(resumed.trials[0].user_attrs["run_dir"]) / "resume-provenance.json").read_text()
            )
            self.assertTrue(provenance["optimizer_restored"])
            self.assertTrue(provenance["scheduler_restored"])

            self.run_checked(self.command(
                "final", output, initial, "--branch", "2", "--trial", "0"
            ))
            self.run_checked(self.command("summarize", output, initial))
            for name in (
                "screening-pareto.csv",
                "control-relative-comparison.csv",
                "final-epoch-500-comparison.csv",
                "report.md",
            ):
                self.assertTrue((output / "summaries" / name).is_file(), name)
            comparison = (output / "summaries/control-relative-comparison.csv").read_text()
            self.assertIn("delta_node_mAP", comparison)
            self.assertIn("delta_beta1_error", comparison)
            final = (output / "summaries/final-epoch-500-comparison.csv").read_text()
            self.assertIn("betti_supervised_epochs", final)
            smoke_validation = json.loads(
                (output / "summaries/smoke-validation.json").read_text()
            )
            self.assertTrue(smoke_validation["passed"])
            self.assertEqual(smoke_validation["wandb_mode"], "offline")
            self.assertTrue(
                (output / "final-continuations/branch-e2-final-0000/.complete.json").is_file()
            )
            final_run = output / "final-continuations/branch-e2-final-0000"
            bootstrap_path = (
                output
                / "final-continuations/metric-history/branch-e2-final-0000.jsonl"
            )
            bootstrap = [
                json.loads(line)
                for line in bootstrap_path.read_text().splitlines()
            ]
            stitched = [
                json.loads(line)
                for line in (final_run / "stitched-validation-metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["epoch"] for row in bootstrap], [1, 2, 3, 4])
            self.assertEqual([row["epoch"] for row in stitched], list(range(1, 9)))
            self.assertEqual([row["betti_active"] for row in stitched[:2]], [0.0, 0.0])
            self.assertTrue(all(row["betti_active"] == 1.0 for row in stitched[2:]))


if __name__ == "__main__":
    unittest.main()
