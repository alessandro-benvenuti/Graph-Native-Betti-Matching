"""CPU study lifecycle tests; fake training never substitutes for the GPU smoke."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import signal
import socket
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
    CampaignError, H1_ABLATION_WEIGHTS, ablation_prepare, ablation_reference, atomic_json, concrete_config, dominates,
    evaluate_trial, freeze, initialize_study, load_settings, prepare,
    manual_stop_path, promote, prune_trials, result_path, scientific_reference, stage_reference, state, study_for,
    single_node_launcher, smoke_config, summary, training_command, trial_lock, worker,
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

    def test_historical_source_without_sampling_schema_is_accepted(self):
        path = self.source / "resolved-config.yaml"
        historical = yaml.safe_load(path.read_text())
        historical["topology"].pop("sampling")
        historical["topology"]["betti_h0"].pop("activation_epoch")
        content = yaml.safe_dump(historical)
        path.write_text(content)
        self.prepared()
        self.assertEqual(path.read_text(), content)
        base = json.loads((self.output / "base-config.json").read_text())
        self.assertIn("sampling", base["topology"])

    def test_historical_source_still_requires_matching_scientific_recipe(self):
        path = self.source / "resolved-config.yaml"
        historical = yaml.safe_load(path.read_text())
        historical["topology"].pop("sampling")
        historical["loss"]["edge"]["classification"]["focal_gamma"] = 1.
        path.write_text(yaml.safe_dump(historical))
        with self.assertRaisesRegex(CampaignError, "differ in loss"):
            prepare(self.args)

    def test_h1_disabled_and_h0_uses_every_local_graph(self):
        self.prepared()
        config = concrete_config(self.output, "h0", 0, {"h0_weight": .01}, 5)
        self.assertFalse(config["topology"]["betti_h1"]["enabled"])
        self.assertFalse(config["topology"]["betti_h1"]["log_only"])
        self.assertEqual(config["topology"]["sampling"]["betti_h0_max_graphs_per_rank"], 8)
        self.assertEqual(config["loss"], self.config["loss"])
        self.assertEqual(config["training"]["scheduler"], self.config["training"]["scheduler"])

    def test_gpu_smoke_caps_the_independent_metric_loader(self):
        self.prepared()
        config = smoke_config(self.output, 0, "control", {"h0_weight": 0., "h1_weight": 0.})
        self.assertEqual(config["data"]["datasets"]["synthetic_mri"]["validation_samples"], 16)
        self.assertEqual(config["evaluation"]["training_metrics"]["max_samples"], 16)
        self.assertEqual(config["experiment"]["name"], "gpu-smoke/v2/control")
        scientific = concrete_config(self.output, "h0", 0, {"h0_weight": .003}, 5)
        self.assertIsNone(scientific["evaluation"]["training_metrics"]["max_samples"])

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

    def test_all_four_h1_weights_have_real_finite_sampled_backward(self):
        from tests.test_graph_losses import _config, _batch, CountingRelationHead
        from models.matcher import HungarianMatcher
        from training.losses import GraphCriterion
        self.prepared()
        for weight in H1_ABLATION_WEIGHTS:
            for validation in (False, True):
                with self.subTest(h1_weight=weight, validation=validation):
                    config = _config()
                    config["topology"] = concrete_config(
                        self.output, "h1", 0, {"h0_weight": .03, "h1_weight": weight}, 5,
                    )["topology"]
                    for name in ("betti_h0", "betti_h1"):
                        config["topology"][name].update(activation_epoch=0, ramp_epochs=0)
                    relation = CountingRelationHead()
                    criterion = GraphCriterion(config, HungarianMatcher(2., 5.), relation, validation=validation)
                    criterion.set_training_progress(3, 0.)
                    tokens, predictions, targets = _batch()
                    tokens = tokens.detach().repeat(2, 1, 1).requires_grad_()
                    predictions = {key: value.detach().repeat(2, 1, 1).requires_grad_()
                                   for key, value in predictions.items()}
                    targets = {key: value * 2 for key, value in targets.items()}
                    losses = criterion(tokens, predictions, targets)
                    self.assertEqual(float(losses["topology_betti_h0_selected_graphs"]), 2.)
                    self.assertEqual(float(losses["topology_betti_h1_selected_graphs"]), 2. if validation else 1.)
                    self.assertGreater(float(losses["betti_h1"].detach()), 0.)
                    (losses["betti_h0_weighted"] + losses["betti_h1_weighted"]).backward()
                    self.assertTrue(torch.isfinite(tokens.grad).all())
                    self.assertIsNotNone(relation.linear.weight.grad)
                    self.assertTrue(torch.isfinite(relation.linear.weight.grad).all())

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
        self.assertIn("--rdzv-backend=static", first)
        self.assertNotIn("--standalone", first)

    def test_launcher_avoids_an_occupied_port_and_uses_unique_rendezvous(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            commands = [single_node_launcher(2), single_node_launcher(2)]
            for command in commands:
                port = int(next(part.split("=", 1)[1] for part in command if part.startswith("--master-port=")))
                self.assertGreater(port, 0)
                self.assertNotEqual(port, occupied.getsockname()[1])
        ids = [next(part for part in command if part.startswith("--rdzv-id=")) for command in commands]
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(single_node_launcher(1), [sys.executable, "-u"])

    def test_real_cpu_rendezvous_and_collective_across_repeated_launches(self):
        fixture = ROOT / "tests/fixtures/ddp_rendezvous_probe.py"
        environment = dict(os.environ)
        # This probe is strictly single-node. Do not let Gloo choose a login
        # node's InfiniBand/hostname interface for local CPU communication.
        environment.update(
            GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LOGLEVEL="INFO",
        )
        for index in range(2):
            report = self.root / f"collective-{index}.json"
            process = subprocess.Popen(
                single_node_launcher(2) + [str(fixture), str(report)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=environment, start_new_session=True,
            )
            try:
                # Includes launcher startup and torch imports from shared
                # storage, not just the fixture's 30-second collective timeout.
                log, _ = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                # subprocess.run(timeout=...) only kills the launcher, leaving
                # its distributed workers alive and hiding their useful logs.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    log, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    log, _ = process.communicate()
                self.fail(f"CPU rendezvous launch {index} timed out after 180 seconds.\n{log}")
            self.assertEqual(process.returncode, 0, log)
            result = json.loads(report.read_text())
            self.assertEqual(result["world_size"], 2)
            self.assertEqual(result["sum"], 3.)
            self.assertTrue(result["agent_store"])

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
        self.assertIn("--job-name=gnbm-focal-h0", submitted[0])
        h1_command = ["bash", command[1], "h1-screen", "2"]
        failed = subprocess.run(h1_command, env=env, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("h1-prepare", failed.stderr)
        worker(self.args)
        prepared = subprocess.run(
            ["bash", command[1], "h1-prepare", "0", "0.001,0.003,0.01,0.03"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
        subprocess.run(h1_command, env=env, check=True, capture_output=True, text=True)
        submitted = [json.loads(line) for line in (self.root / "submissions.jsonl").read_text().splitlines()]
        self.assertEqual(len(submitted), 4)
        self.assertIn("--job-name=gnbm-focal-h1", submitted[2])
        self.assertIn("--array=0-1", submitted[2])
        self.assertIn("--gres=gpu:4", submitted[2])
        self.assertIn("--dependency=afterany:9003", submitted[3])
        stopped = subprocess.run(
            ["bash", command[1], "prune", "0", "Manual stop with an explicit reason"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
        self.assertIn("already COMPLETE", stopped.stdout)
        self.assertEqual(len((self.root / "submissions.jsonl").read_text().splitlines()), 4)

    def _allocate_h0_trials(self, count):
        study = study_for(self.output, "h0")
        trials = []
        for _ in range(count):
            trial = study.ask()
            trial.suggest_float("h0_weight", *self.settings["h0_range"], log=True)
            trials.append(trial)
        return study, trials

    def _complete_h0_trial(self, study, trial, tell=True):
        reference = stage_reference(self.output, 5)
        row = {"family": "h0", "trial": trial.number, "endpoint": 5,
               "parameters": trial.params, "aggregation": reference,
               "control": reference["metrics"], "metrics": reference["metrics"],
               "deltas": {key: 0. for key in reference["metrics"]}}
        path = result_path(self.output, "h0", 5, trial.number)
        atomic_json(path, row)
        if tell:
            study.tell(trial, values=[reference["metrics"][key] for key in (
                "node_mAP", "edge_mAP", "beta0_absolute_error", "beta1_absolute_error",
            )])
        return path

    def test_manual_pruning_is_terminal_idempotent_and_preserves_files(self):
        self.prepared()
        study, trials = self._allocate_h0_trials(4)
        for trial in trials[:2]:
            self._complete_h0_trial(study, trial)
        h1 = study_for(self.output, "h1")
        h1.enqueue_trial({"h0_weight": .03, "h1_weight": .001})
        retained = [self.output / name for name in (
            "campaign.json", "base-config.json", "initialization/epoch_0500.pt", "studies/h1/study.journal",
        )]
        for trial in trials[2:]:
            run = self.output / f"runs/h0/trial_{trial.number:04d}"
            checkpoint = run / "models/latest_checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"retained checkpoint")
            history = run / "validation-metrics.jsonl"
            history.write_text('{"epoch":3}\n{"epoch":4}\n')
            retained += [checkpoint, history]
        before = {path: path.read_bytes() for path in retained}
        self.args.trials, self.args.reason = "2,3", "H0-only trade-off; switching to the H1 ablation"
        prune_trials(self.args)
        self.assertEqual([trial.state.name for trial in study.trials], ["COMPLETE", "COMPLETE", "PRUNED", "PRUNED"])
        for trial in study.trials[2:]:
            audit = json.loads(manual_stop_path(self.output, "h0", trial.number).read_text())
            self.assertEqual(audit["latest_validation_epoch"], 4)
            self.assertFalse(audit["endpoint_comparison_eligible"])
            self.assertTrue(audit["checkpoint_retained"])
            self.assertEqual(audit["reason"], self.args.reason)
            self.assertEqual(trial.user_attrs["manual_stop"], audit)
        prune_trials(self.args)
        with patch("scripts.optimize_focal_h0.run_command", side_effect=AssertionError("Pruned trials must never resume")):
            worker(self.args)
        self.assertEqual(len(study.trials), 4)
        for path in retained:
            self.assertEqual(path.read_bytes(), before[path], str(path))
        summary(self.args)
        with (self.output / "summaries/h0/epoch_0005.csv").open() as handle:
            self.assertEqual({row["trial"] for row in csv.DictReader(handle)}, {"0", "1"})
        self.assertFalse((self.output / "runs/h0/trial_0002/training-complete").exists())

    def test_manual_pruning_refuses_any_live_worker_before_mutating_any_trial(self):
        self.prepared()
        study, _ = self._allocate_h0_trials(2)
        self.args.trials, self.args.reason = "0,1", "Explicit manual stop"
        with trial_lock(self.output, "h0", 1) as acquired:
            self.assertTrue(acquired)
            with self.assertRaisesRegex(CampaignError, "still locked"):
                prune_trials(self.args)
        self.assertTrue(all(trial.state.name == "RUNNING" for trial in study.trials))
        self.assertFalse(manual_stop_path(self.output, "h0", 0).exists())
        self.assertFalse(any(trial.user_attrs for trial in study.trials))
        for numbers in ("0,999", "0,0", "-1", "bad"):
            self.args.trials = numbers
            with self.subTest(numbers=numbers), self.assertRaises(CampaignError):
                prune_trials(self.args)
        self.args.trials, self.args.reason = "0", ""
        with self.assertRaisesRegex(CampaignError, "reason"):
            prune_trials(self.args)

    def test_manual_pruning_recovers_an_already_written_endpoint_as_complete(self):
        self.prepared()
        study, trials = self._allocate_h0_trials(1)
        path = self._complete_h0_trial(study, trials[0], tell=False)
        retained = path.read_bytes()
        self.args.trials, self.args.reason = "0", "Manual cancellation raced endpoint completion"
        prune_trials(self.args)
        self.assertEqual(study.trials[0].state.name, "COMPLETE")
        self.assertEqual(path.read_bytes(), retained)
        self.assertFalse(manual_stop_path(self.output, "h0", 0).exists())
        prune_trials(self.args)
        self.assertEqual(study.trials[0].state.name, "COMPLETE")

    def test_manual_stop_record_blocks_resume_if_finalization_is_interrupted(self):
        import optuna
        self.settings.update(trials=1, h0_anchors=[.003], promotion_counts=[1, 1])
        self.settings_path.write_text(yaml.safe_dump(self.settings))
        self.prepared()
        study, _ = self._allocate_h0_trials(1)
        self.args.trials, self.args.reason = "0", "Explicit manual stop"
        with patch.object(optuna.study.Study, "tell", side_effect=RuntimeError("interrupted finalization")):
            with self.assertRaisesRegex(RuntimeError, "interrupted finalization"):
                prune_trials(self.args)
        self.assertEqual(study.trials[0].state.name, "RUNNING")
        with patch("scripts.optimize_focal_h0.run_command", side_effect=AssertionError("Stop marker must block resume")):
            worker(self.args)
        prune_trials(self.args)
        self.assertEqual(study.trials[0].state.name, "PRUNED")
        self.assertEqual(len(study.trials), 1)

    def test_h1_prepare_from_initial_budget_is_frozen_and_preserves_h0(self):
        self.settings.update(h0_anchors=[.03])
        self.settings_path.write_text(yaml.safe_dump(self.settings))
        self.prepared()
        worker(self.args)
        untouched = [self.output / name for name in (
            "campaign.json", "base-config.json", "dataset-manifest.json", "initialization/epoch_0500.pt",
            "studies/h0/study.journal", "runs/h0/trial_0000/models/latest_checkpoint.pt",
        )]
        before = {path: path.read_bytes() for path in untouched}
        self.assertFalse((self.output / "results/h0/epoch_0008/trial_0000.json").exists())
        ablation_prepare(self.output, 0)
        plan_path = self.output / "h1-plan.json"
        frozen = plan_path.read_bytes()
        plan = json.loads(frozen)
        self.assertEqual(plan["h0_weight"], .03)
        self.assertEqual(plan["h1_weights"], list(H1_ABLATION_WEIGHTS))
        self.assertEqual(plan["h0_reference"]["endpoint"], 5)
        self.assertEqual(plan["initial_checkpoint"], state(self.output)["checkpoint"])
        ablation_prepare(self.output, 0)
        self.assertEqual(plan_path.read_bytes(), frozen)
        self.assertEqual(len(study_for(self.output, "h1").trials), 4)
        with self.assertRaisesRegex(CampaignError, "frozen H1"):
            ablation_prepare(self.output, 0, [.001, .01])
        self.args.family = "h1"
        worker(self.args)
        trials = study_for(self.output, "h1").trials
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(t.state.name == "COMPLETE" for t in trials))
        for trial in trials:
            self.assertEqual(trial.params["h0_weight"], .03)
            provenance = json.loads((self.output / f"runs/h1/trial_{trial.number:04d}/resume-provenance.json").read_text())
            self.assertEqual(Path(provenance["checkpoint"]).resolve(), Path(state(self.output)["checkpoint"]).resolve())
            self.assertEqual(provenance["checkpoint_epoch"], 2)
            self.assertTrue(provenance["optimizer_restored"])
            self.assertTrue(provenance["scheduler_reset"])
        for path in untouched:
            self.assertEqual(path.read_bytes(), before[path], str(path))
        summary(self.args)
        with (self.output / "summaries/h1/epoch_0005.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["h0_reference_trial"], "0")
            self.assertAlmostEqual(float(row["delta_vs_h0_beta0_absolute_error"]), 0.)
            self.assertAlmostEqual(float(row["delta_vs_h0_beta1_absolute_error"]), -2 * float(row["h1_weight"]))
        # Never compare later H1 results against an earlier H0 endpoint.
        self.assertIsNone(ablation_reference(self.output, 8))

    def test_h1_rejects_unfinished_parent_and_invalid_weights(self):
        self.prepared()
        with self.assertRaisesRegex(CampaignError, "completed epoch 5"):
            ablation_prepare(self.output, 0)
        self.assertFalse((self.output / "h1-plan.json").exists())
        worker(self.args)
        for weights in ([], [.001, .001], [-.1], [float("nan")], [float("inf")], ["bad"]):
            with self.subTest(weights=weights), self.assertRaises(CampaignError):
                ablation_prepare(self.output, 0, weights)
        self.assertFalse((self.output / "h1-plan.json").exists())

    def test_h1_prepare_repairs_partially_enqueued_grid(self):
        import optuna
        self.prepared()
        worker(self.args)
        enqueue = optuna.study.Study.enqueue_trial
        calls = 0
        def interrupted(study, parameters):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted enqueue")
            return enqueue(study, parameters)
        with patch.object(optuna.study.Study, "enqueue_trial", interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted enqueue"):
                ablation_prepare(self.output, 0)
        self.assertEqual(len(study_for(self.output, "h1").trials), 1)
        ablation_prepare(self.output, 0)
        self.assertEqual(len(study_for(self.output, "h1").trials), 4)

    def test_two_h1_workers_allocate_only_the_four_frozen_arms(self):
        self.prepared()
        worker(self.args)
        ablation_prepare(self.output, 0)
        h0_journal = self.output / "studies/h0/study.journal"
        original = h0_journal.read_bytes()
        command = [sys.executable, str(ROOT / "scripts/optimize_focal_h0.py"), "worker",
                   "--output", str(self.output), "--family", "h1", "--resume", "--poll-interval", ".02",
                   "--train-command", *self.args.train_command]
        with (self.root / "parallel-h1.log").open("w") as log:
            processes = [subprocess.Popen(
                command, env={**os.environ, "SLURM_ARRAY_TASK_ID": str(index), "FAKE_TRAIN_SLEEP_PER_EPOCH": ".1"},
                stdout=log, stderr=log,
            ) for index in range(2)]
            try:
                for process in processes:
                    self.assertEqual(process.wait(timeout=60), 0, (self.root / "parallel-h1.log").read_text())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        trials = study_for(self.output, "h1").trials
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(trial.state.name == "COMPLETE" for trial in trials))
        self.assertEqual({trial.params["h1_weight"] for trial in trials}, set(H1_ABLATION_WEIGHTS))
        self.assertEqual(h0_journal.read_bytes(), original)

    def test_legacy_h1_plan_keeps_its_own_grid(self):
        self.prepared()
        worker(self.args)
        atomic_json(self.output / "h1-plan.json", {
            "parent_trial": 0, "h0_weight": .001,
            "h1_weights": [0., .001, .003], "endpoint": 5,
        })
        ablation_prepare(self.output, 0)
        self.args.family = "h1"
        worker(self.args)
        self.assertEqual(len(study_for(self.output, "h1").trials), 3)

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
        self.assertEqual(len(h1.trials), 4)
        self.assertEqual({t.params["h1_weight"] for t in h1.trials}, set(H1_ABLATION_WEIGHTS))
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
