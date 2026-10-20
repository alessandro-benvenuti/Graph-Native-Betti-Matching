"""CPU study lifecycle tests; fake training never substitutes for the GPU smoke."""
import argparse
import copy
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

import torch
import yaml

from configs import load_config
from scripts.optimize_focal_h0 import (
    CampaignError, ablation_prepare, atomic_json, concrete_config, dominates,
    evaluate_trial, freeze, initialize_study, load_settings, prepare,
    promote, scientific_reference, stage_reference, state, study_for,
    summary, training_command, trial_lock, worker,
)

ROOT = Path(__file__).resolve().parents[1]


class FocalH0Tests(unittest.TestCase):
    def setUp(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("Install requirements/optuna.txt to run this study suite")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "study"
        self.source = self.root / "source"
        self.reference = self.root / "reference"
        self.source.mkdir()
        self.reference.mkdir()
        environment = {"GNBM_OUTPUT_DIR": str(self.root), "SYNTHETIC_MRI_DATASET": "/dataset", "PLANTS_DATASET": "/plants"}
        with patch.dict(os.environ, environment):
            config = load_config(ROOT / "configs/experiments/boundary_gamma_sweep_500/finetune_node_edge_focal_g05.yaml")
        self.settings = load_settings(ROOT / "configs/experiments/focal_h0_study.yaml")
        self.settings.update(branch_epoch=2, endpoints=[5, 8, 11], continuation_epochs=14,
                             schedule_duration_epochs=12, validation_interval=1, trials=4,
                             promotion_counts=[2, 1], h0_anchors=[.001, .003], tpe_startup_trials=2)
        self.settings_path = self.root / "settings.yaml"
        self.settings_path.write_text(yaml.safe_dump(self.settings))
        config["data"].update(batch_size=8)
        config["training"].update(epochs=14, warmup_epochs=0)
        config["training"]["scheduler"]["duration_epochs"] = 12
        config["training"]["optimizer"]["base_lr"] = 1e-5
        config["evaluation"]["interval_epochs"] = 1
        self.config = config
        for run in (self.source, self.reference):
            (run / "resolved-config.yaml").write_text(yaml.safe_dump(config))
        initial = self.source / "models/latest_checkpoint.pt"
        initial.parent.mkdir()
        torch.save({"epoch": 2, "net": {}, "optimizer": {"state": "retained"},
                    "scheduler": {}, "trainer_state": {"world_size": 2}}, initial)
        atomic_json(self.reference / "extension-source.json", {
            "source_epoch": 2, "target_epoch": 14, "scheduler_restart_base_lr": 1e-5,
            "scheduler_duration_epochs": 12, "early_stopping_reset": True,
            "source_world_size": 2, "continuation_world_size": 4, "global_batch_size": 32,
            "resume_checkpoint": str(initial),
        })
        manifest = {"schema_version": 1, "experiment_seed": config["experiment"]["seed"],
                    "train": ["sample-a", "sample-b"], "validation": ["sample-v"]}
        for run in (self.source, self.reference):
            atomic_json(run / "dataset-manifest.json", manifest)
            (run / "validation-metrics.jsonl").write_text("".join(json.dumps({
                "epoch": epoch, "iteration": epoch, "node_mAP": .8, "edge_mAP": .7,
                "beta0_absolute_error": 2., "beta1_absolute_error": 4.,
            }) + "\n" for epoch in range(1, 15 if run == self.reference else 3)))
        self.args = argparse.Namespace(
            output=self.output, source_run=self.source, reference_run=self.reference,
            settings=self.settings_path, family="h0", endpoint=None, resume=True,
            trials=None, parent=None, poll_interval=.01,
            train_command=[sys.executable, str(ROOT / "tests/fixtures/fake_optuna_train.py"),
                           "--config", "{config}", "--output-dir", "{output}", "--run-name", "{run_name}"],
        )
        self.environment = patch.dict(os.environ, {"FAKE_TRAIN_WORLD_SIZE": "4"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def prepared(self):
        prepare(self.args)
        stage_reference(self.output, 5)
        initialize_study(self.output)

    def test_reference_rejects_changed_recipe(self):
        bad = copy.deepcopy(self.config)
        bad["loss"]["edge"]["classification"]["focal_gamma"] = 1
        with self.assertRaisesRegex(CampaignError, "edge focal"):
            scientific_reference(bad, self.settings)

    def test_wrong_source_epoch_is_rejected(self):
        path = self.source / "models/latest_checkpoint.pt"
        torch.save({"epoch": 1}, path)
        with self.assertRaisesRegex(CampaignError, "branch epoch"):
            prepare(self.args)

    def test_prepare_freezes_checkpoint_control_and_protocol(self):
        self.prepared()
        prepare(self.args)
        copied = Path(state(self.output)["checkpoint"])
        self.assertEqual(copied.read_bytes(), (self.source / "models/latest_checkpoint.pt").read_bytes())
        self.assertFalse((self.output / "control").exists())
        with self.assertRaisesRegex(CampaignError, "frozen"):
            freeze(self.output / "campaign.json", {"changed": True})
        with self.assertRaisesRegex(CampaignError, "configured fidelity"):
            stage_reference(self.output, 7)

    def test_h1_disabled_and_h0_uses_every_local_graph(self):
        self.prepared()
        config = concrete_config(self.output, "h0", 0, {"h0_weight": .01}, 5)
        self.assertFalse(config["topology"]["betti_h1"]["enabled"])
        self.assertFalse(config["topology"]["betti_h1"]["log_only"])
        self.assertEqual(config["topology"]["sampling"]["betti_h0_max_graphs_per_rank"], 8)
        self.assertEqual(config["loss"], self.config["loss"])
        self.assertEqual(config["training"]["scheduler"], self.config["training"]["scheduler"])

    def test_real_h0_backward_skips_h1_in_training_and_validation(self):
        from tests.test_graph_losses import _config, _batch, CountingRelationHead
        from models.matcher import HungarianMatcher
        from training.losses import GraphCriterion
        self.prepared()
        config = _config()
        config["topology"] = concrete_config(self.output, "h0", 0, {"h0_weight": .01}, 5)["topology"]
        config["topology"]["betti_h0"].update(activation_epoch=0, ramp_epochs=0)
        for validation in (False, True):
            criterion = GraphCriterion(config, HungarianMatcher(2., 5.), CountingRelationHead(), validation=validation)
            criterion.set_training_progress(3, 0.)
            tokens, predictions, targets = _batch()
            with patch("training.losses.criterion.cycle_space_matching_loss", side_effect=AssertionError("H1 must not execute")):
                losses = criterion(tokens, predictions, targets)
                self.assertEqual(float(losses["topology_betti_h1_selected_graphs"]), 0.)
                self.assertTrue(torch.isfinite(losses["betti_h0_weighted"]))
                losses["total"].backward()
                self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_scheduler_reset_only_at_initial_branch(self):
        self.prepared()
        first = training_command(self.output, Path("config"), "run", Path("source"), True)
        strict = training_command(self.output, Path("config"), "run", Path("own"), False)
        self.assertIn("--reset-scheduler-on-resume", first)
        self.assertIn("--allow-world-size-change-on-resume", first)
        self.assertNotIn("--reset-scheduler-on-resume", strict)
        self.assertNotIn("--reset-early-stopping-on-resume", strict)
        self.assertNotIn("--allow-world-size-change-on-resume", strict)
        self.assertIn("--nproc_per_node=4", first)

    def test_pareto_tradeoffs_have_no_scalar_threshold(self):
        left = {"node_mAP": .8, "edge_mAP": .7, "beta0_absolute_error": 2., "beta1_absolute_error": 4.}
        right = {**left, "beta0_absolute_error": 1., "edge_mAP": .69}
        self.assertFalse(dominates(left, right))
        self.assertFalse(dominates(right, left))
        self.assertTrue(dominates({**left, "beta0_absolute_error": 1.}, left))

    def test_missing_reference_epochs_fail_instead_of_comparing_different_budgets(self):
        self.prepared()
        (self.reference / "validation-metrics.jsonl").write_text("{}\n")
        with self.assertRaises((CampaignError, KeyError)):
            stage_reference(self.output, 8)

    def test_kernel_trial_lock_prevents_duplicate_resume(self):
        with trial_lock(self.output, "h0", 0) as first:
            with trial_lock(self.output, "h0", 0) as second:
                self.assertTrue(first)
                self.assertFalse(second)

    def test_existing_reference_manifest_must_match_trial(self):
        self.prepared()
        atomic_json(self.output / "dataset-manifest.json", {"different": True})
        worker(self.args)
        self.assertTrue(all(t.state.name == "FAIL" for t in study_for(self.output, "h0").trials))
        self.assertFalse(list((self.output / "results").glob("**/trial_*.json")))

    def test_two_workers_never_allocate_extra_trials(self):
        self.prepared()
        command = [sys.executable, str(ROOT / "scripts/optimize_focal_h0.py"), "worker", "--output", str(self.output), "--resume", "--poll-interval", ".02", "--train-command", *self.args.train_command]
        with (self.root / "parallel.log").open("w") as log:
            processes = [subprocess.Popen(command, env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index), "FAKE_TRAIN_SLEEP_PER_EPOCH": ".1"}, stdout=log, stderr=log) for index in range(2)]
            try:
                for process in processes:
                    self.assertEqual(process.wait(timeout=60), 0, (self.root / "parallel.log").read_text())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        trials = study_for(self.output, "h0").trials
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(t.state.name == "COMPLETE" for t in trials))

    def test_launcher_requires_smoke_and_preserves_chain_dependencies(self):
        self.prepared()
        venv = self.root / "venv/bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(sys.executable)
        executable = self.root / "sbatch"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "root=pathlib.Path(os.environ['WORK'])\n"
            "path=root/'submissions.jsonl'\n"
            "old=path.read_text().splitlines() if path.exists() else []\n"
            "with path.open('a') as h: h.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "print('Submitted batch job',9001+len(old))\n"
        )
        executable.chmod(0o755)
        env = {**os.environ, "WORK": str(self.root), "GNBM_OUTPUT_DIR": str(self.output),
               "GNBM_A100_VENV": str(venv.parent), "GNBM_H0_SEGMENTS": "2", "GNBM_H0_AFTER_JOB_ID": "77",
               "GNBM_H0_FAMILY": "h0", "PATH": str(self.root) + os.pathsep + os.environ["PATH"]}
        command = ["bash", str(ROOT / "cluster/jean_zay/submit_focal_h0_study.sh"), "screen", "2"]
        failed = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("GPU smoke", failed.stderr)
        atomic_json(self.output / "gpu-smoke/passed.json", {"passed": True})
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        submitted = [json.loads(line) for line in (self.root / "submissions.jsonl").read_text().splitlines()]
        self.assertEqual(len(submitted), 2)
        self.assertIn("--dependency=afterany:77", submitted[0])
        self.assertIn("--dependency=afterany:9001", submitted[1])
        self.assertIn("--array=0-1", submitted[0])
        self.assertIn("--gres=gpu:4", submitted[0])

    def test_full_lifecycle_promotions_and_optional_h1_are_separate(self):
        self.prepared()
        worker(self.args)
        study = study_for(self.output, "h0")
        self.assertEqual([t.state.name for t in study.trials], ["COMPLETE"] * 4)
        self.assertEqual([t.params["h0_weight"] for t in study.trials[:2]], [.001, .003])
        self.args.endpoint, self.args.trials = 8, "0,1"
        promote(self.args)
        worker(self.args)
        # Promotion must not overwrite the original Optuna objective values.
        self.assertEqual(study_for(self.output, "h0").trials[0].user_attrs["endpoint"], 5)
        self.args.endpoint, self.args.trials = 11, "0"
        promote(self.args)
        worker(self.args)
        provenance = json.loads((self.output / "runs/h0/trial_0000/resume-provenance.json").read_text())
        self.assertFalse(provenance["scheduler_reset"])
        self.assertFalse(provenance["patience_reset"])
        summary(self.args)
        self.assertTrue((self.output / "summaries/h0/epoch_0011.csv").exists())
        self.assertFalse((self.output / "results/h0/epoch_0011/trial_0001.json").exists())
        ablation_prepare(self.output, 0)
        self.args.family, self.args.endpoint = "h1", None
        worker(self.args)
        h1 = study_for(self.output, "h1")
        self.assertEqual(len(h1.trials), 3)
        self.assertEqual({t.params["h1_weight"] for t in h1.trials}, {0., .001, .003})
        self.assertEqual(len(study_for(self.output, "h0").trials), 4)

    def test_real_signal_interruption_resumes_same_trial(self):
        self.settings.update(trials=1, h0_anchors=[.003], promotion_counts=[1, 1])
        self.settings_path.write_text(yaml.safe_dump(self.settings))
        self.prepared()
        command = [sys.executable, str(ROOT / "scripts/optimize_focal_h0.py"), "worker",
                   "--output", str(self.output), "--resume", "--poll-interval", ".02", "--train-command", *self.args.train_command]
        # Wait for a real intermediate checkpoint before delivering SIGTERM.
        with (self.root / "worker.log").open("w") as log:
            process = subprocess.Popen(command, env={**os.environ, "FAKE_TRAIN_SLEEP_PER_EPOCH": "1"}, stdout=log, stderr=log)
            try:
                own = self.output / "runs/h0/trial_0000/models/latest_checkpoint.pt"
                deadline = time.monotonic() + 45
                while not own.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail((self.root / "worker.log").read_text())
                    time.sleep(.05)
                self.assertTrue(own.exists())
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=15)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        study = study_for(self.output, "h0")
        self.assertEqual(study.trials[0].state.name, "RUNNING")
        worker(self.args)
        study = study_for(self.output, "h0")
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state.name, "COMPLETE")
        self.assertEqual(study.trials[0].user_attrs["resume_count"], 1)
        provenance = json.loads((self.output / "runs/h0/trial_0000/resume-provenance.json").read_text())
        self.assertFalse(provenance["scheduler_reset"])


if __name__ == "__main__":
    unittest.main()
