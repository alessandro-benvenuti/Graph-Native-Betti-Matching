"""CPU-only infrastructure tests for the node/edge Betti Optuna controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from configs import validate_config
from scripts.optimize_node_edge_betti import (
    CampaignError,
    ControllerLock,
    JsonlMonitor,
    apply_parameters,
    build_sampler_pruner,
    compute_objective,
    load_campaign,
    run_command,
)
from scripts.summarize_node_edge_betti_optuna import select_best


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/experiments/node_edge_betti_optuna/smoke.yaml"
ENVIRONMENT = {
    "SYNTHETIC_MRI_DATASET": "/tmp/synthetic-mri",
    "GNBM_OUTPUT_DIR": "/tmp/gnbm-output",
}


def metrics(**overrides):
    value = {
        "node_mAP": 0.80, "edge_mAP": 0.70,
        "node_f1": 0.75, "edge_f1": 0.65,
        "beta0_absolute_error": 2.0, "beta1_absolute_error": 4.0,
        "smd": 1.25,
    }
    value.update(overrides)
    return value


class ConfigurationTests(unittest.TestCase):
    def test_nested_overrides_apply_common_schedule(self):
        base, _ = load_campaign(CAMPAIGN, ENVIRONMENT)
        result = apply_parameters(base, {
            "topology.betti_h0.weight": 0.003,
            "topology.betti_h1.weight": 0.01,
            "topology.betti_h1.false_positive_weight": 0.25,
            "betti_warmup_epochs": 2,
            "betti_ramp_epochs": 3,
        })
        self.assertEqual(result["topology"]["betti_h0"]["weight"], 0.003)
        self.assertEqual(result["topology"]["betti_h1"]["warmup_epochs"], 2)
        self.assertEqual(result["topology"]["betti_h0"]["ramp_epochs"], 3)

    def test_ordinary_validation_succeeds_after_metadata_removed(self):
        config, metadata = load_campaign(CAMPAIGN, ENVIRONMENT)
        self.assertNotIn("optuna", config)
        self.assertEqual(metadata["n_trials"], 2)
        validate_config(config)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "epsilon": 1e-8, "penalty_scale": 10,
            "tolerances": {"node_f1": .003, "edge_f1": .003,
                           "node_mAP": .005, "edge_mAP": .005},
        }

    def test_objective_components(self):
        result = compute_objective(
            metrics(beta0_absolute_error=1, beta1_absolute_error=2),
            metrics(), self.config,
        )
        self.assertAlmostEqual(result["topology_score"], 0.5)
        self.assertEqual(result["objective"], 0.5)
        self.assertTrue(result["feasible"])

    def test_zero_beta_denominator_uses_epsilon(self):
        result = compute_objective(
            metrics(beta0_absolute_error=0, beta1_absolute_error=0),
            metrics(beta0_absolute_error=0, beta1_absolute_error=0), self.config,
        )
        self.assertEqual(result["topology_score"], 0)

    def test_feasibility_boundary_above_and_below(self):
        control = metrics()
        boundary = metrics(node_f1=control["node_f1"] - .003,
                           edge_f1=control["edge_f1"] - .003,
                           node_mAP=control["node_mAP"] - .005,
                           edge_mAP=control["edge_mAP"] - .005)
        self.assertTrue(compute_objective(boundary, control, self.config)["feasible"])
        below = dict(boundary, edge_f1=boundary["edge_f1"] - 1e-6)
        self.assertFalse(compute_objective(below, control, self.config)["feasible"])

    def test_missing_and_malformed_metric_rejected(self):
        missing = metrics(); missing.pop("smd")
        with self.assertRaises(CampaignError):
            compute_objective(missing, metrics(), self.config)
        with self.assertRaises(CampaignError):
            compute_objective(metrics(edge_f1="bad"), metrics(), self.config)


class SelectionTests(unittest.TestCase):
    def row(self, number, feasible, topology, edge=.6, node=.7, violation=0):
        return {"number": number, "state": "COMPLETE", "feasible": feasible,
                "topology_score": topology, "edge_f1": edge, "node_f1": node,
                "total_violation": violation}

    def test_best_feasible_and_tie_breaks(self):
        chosen, feasible = select_best([
            self.row(0, True, .8, .7), self.row(1, True, .8, .8),
            self.row(2, False, .1, violation=.01),
        ])
        self.assertTrue(feasible)
        self.assertEqual(chosen["number"], 1)

    def test_no_feasible_ranks_by_violation(self):
        chosen, feasible = select_best([
            self.row(0, False, .2, violation=.02),
            self.row(1, False, .9, violation=.01),
        ])
        self.assertFalse(feasible)
        self.assertEqual(chosen["number"], 1)


class MonitorAndProcessTests(unittest.TestCase):
    def test_incremental_jsonl_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"epoch": 1}\n{"epoch":')
            monitor = JsonlMonitor(path)
            self.assertEqual(monitor.read_new(), [{"epoch": 1}])
            with path.open("a") as handle: handle.write(" 2}\n")
            self.assertEqual(monitor.read_new(), [{"epoch": 2}])

    def test_malformed_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("not-json\n")
            with self.assertRaises(CampaignError):
                JsonlMonitor(path).read_new()

    def test_pruned_process_is_terminated_and_artifact_preserved(self):
        class Pruned(Exception): pass
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            code = (
                "import json,time,pathlib; p=pathlib.Path(r'%s'); "
                "p.mkdir(parents=True,exist_ok=True); (p/'keep.txt').write_text('yes'); "
                "f=(p/'validation-metrics.jsonl').open('w'); "
                "f.write(json.dumps({'epoch':1})+'\\n'); f.flush(); time.sleep(20)"
            ) % run
            def callback(_record, process):
                process.terminate()
                raise Pruned()
            with self.assertRaises(Pruned):
                run_command([sys.executable, "-c", code], run, callback, poll=.01)
            self.assertEqual((run / "keep.txt").read_text(), "yes")

    def test_completed_trial_directory_refusal_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs/trial_0000"
            path.mkdir(parents=True)
            self.assertTrue(path.exists())
            with self.assertRaises(CampaignError):
                if path.exists():
                    raise CampaignError(f"refusing to reuse existing trial directory: {path}")

    def test_controller_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".lock"
            with ControllerLock(path):
                with self.assertRaises(CampaignError):
                    with ControllerLock(path): pass
            self.assertFalse(path.exists())


@unittest.skipUnless(__import__("importlib").util.find_spec("optuna"), "Optuna not installed")
class OptunaTests(unittest.TestCase):
    def test_deterministic_sampling(self):
        import optuna
        def sequence():
            sampler, pruner = build_sampler_pruner(optuna, {
                "seed": 364505, "pruner": {"name": "none"}
            })
            study = optuna.create_study(sampler=sampler, pruner=pruner)
            study.optimize(lambda trial: trial.suggest_categorical("x", [0, 1, 2]), n_trials=8)
            return [trial.params["x"] for trial in study.trials]
        self.assertEqual(sequence(), sequence())

    def test_resumed_study_keeps_completed_trial(self):
        import optuna
        with tempfile.TemporaryDirectory() as directory:
            storage = "sqlite:///" + str(Path(directory) / "study.sqlite3")
            first = optuna.create_study(study_name="resume", storage=storage)
            first.optimize(lambda _trial: 1.0, n_trials=1)
            resumed = optuna.create_study(
                study_name="resume", storage=storage, load_if_exists=True
            )
            self.assertEqual(len(resumed.trials), 1)
            self.assertEqual(resumed.trials[0].state.name, "COMPLETE")

    def test_fake_training_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "study"
            checkpoint = root / "initial.pt"
            checkpoint.write_bytes(b"fixed-checkpoint")
            environment = dict(os.environ)
            environment.update(ENVIRONMENT)
            common = [
                "--config", str(CAMPAIGN), "--output", str(output),
                "--initial-weights", str(checkpoint), "--poll-interval", ".01",
                "--train-command", sys.executable,
                str(ROOT / "tests/fixtures/fake_optuna_train.py"),
                "--config", "{config}", "--output-dir", "{output}",
                "--run-name", "{run_name}", "--initial-weights", "{initial_weights}",
            ]
            script = str(ROOT / "scripts/optimize_node_edge_betti.py")
            subprocess.run([sys.executable, script, "control", *common],
                           cwd=ROOT, env=environment, check=True)
            subprocess.run([sys.executable, script, "optimize", *common],
                           cwd=ROOT, env=environment, check=True)
            before = sorted((output / "runs").iterdir())
            subprocess.run([sys.executable, script, "optimize", *common],
                           cwd=ROOT, env=environment, check=True)
            self.assertEqual(before, sorted((output / "runs").iterdir()))
            summary = ROOT / "scripts/summarize_node_edge_betti_optuna.py"
            subprocess.run([
                sys.executable, str(summary), "--output", str(output),
                "--study-name", "node-edge-betti-optuna-smoke",
            ], cwd=ROOT, env=environment, check=True)
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "trials.csv").is_file())
            self.assertTrue((output / "best-feasible.yaml").is_file())
            self.assertEqual(len(before), 2)


if __name__ == "__main__":
    unittest.main()
