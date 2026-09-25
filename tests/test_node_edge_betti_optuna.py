"""CPU-only tests for fixed-tail multi-objective Betti orchestration."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from configs import validate_config
from scripts.optimize_node_edge_betti import (
    CampaignError, ControllerLock, JsonlMonitor, OBJECTIVES, aggregate_tail,
    apply_parameters, build_sampler, create_storage, load_campaign, run_command,
    recover_stale_trials, validate_optuna_config,
)
from scripts.summarize_node_edge_betti_optuna import (
    balanced_representative, dominates, pareto_front, representatives, write_summary,
)
from scripts.prepare_node_edge_betti_final_pair import prepare as prepare_final_pair
from scripts.propose_node_edge_betti_study_b import proposed_space

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/node_edge_betti_optuna"
SMOKE = CONFIG_ROOT / "smoke.yaml"
STUDY_A = CONFIG_ROOT / "study_a.yaml"
STUDY_B = CONFIG_ROOT / "study_b_template.yaml"
ENVIRONMENT = {"SYNTHETIC_MRI_DATASET": "/tmp/synthetic-mri",
               "GNBM_OUTPUT_DIR": "/tmp/gnbm-output"}


def metric_record(epoch, **overrides):
    value = {"epoch": epoch, "iteration": epoch * 10, "node_mAP": .8,
             "edge_mAP": .7, "beta0_absolute_error": 2.0,
             "beta1_absolute_error": 4.0, "node_f1": .75,
             "edge_f1": .65, "smd": 1.25, "predicted_nodes": 19.0,
             "predicted_edges": 20.0}
    value.update(overrides); return value


def row(number, node, edge, beta0, beta1, state="COMPLETE"):
    return {"number": number, "state": state, "node_mAP": node,
            "edge_mAP": edge, "beta0_absolute_error": beta0,
            "beta1_absolute_error": beta1}


class ConfigurationTests(unittest.TestCase):
    def test_valid_objective_order_and_directions(self):
        _, config = load_campaign(STUDY_A, ENVIRONMENT)
        self.assertEqual(tuple((x["metric"], x["direction"]) for x in config["objectives"]), OBJECTIVES)
        self.assertEqual(config["n_trials"], 48)

    def test_legacy_scalar_fields_rejected(self):
        _, config = load_campaign(STUDY_A, ENVIRONMENT)
        for key, value in (("direction", "minimize"), ("objective", {}), ("pruner", {})):
            invalid = dict(config); invalid[key] = value
            with self.assertRaisesRegex(CampaignError, "legacy scalar"):
                validate_optuna_config(invalid)

    def test_study_a_space_and_schedule_application(self):
        base, _ = load_campaign(STUDY_A, ENVIRONMENT)
        result = apply_parameters(base, {"topology.betti_h0.weight": .003,
            "topology.betti_h1.weight": .01,
            "topology.betti_h1.false_positive_weight": .25,
            "betti_warmup_epochs": 20, "betti_ramp_epochs": 30})
        self.assertEqual(result["topology"]["betti_h0"]["warmup_epochs"], 20)
        self.assertEqual(result["topology"]["betti_h1"]["ramp_epochs"], 30)
        self.assertEqual(result["topology"]["betti_h1"]["false_positive_weight"], .25)

    def test_study_b_edge_loss_mapping_and_alpha(self):
        base, _ = load_campaign(STUDY_B, ENVIRONMENT)
        focal = apply_parameters(base, {"edge_loss": "focal_gamma_1.0",
                                        "topology.complex.alpha": .75})
        self.assertEqual(focal["loss"]["edge"]["classification"]["name"], "focal")
        self.assertEqual(focal["loss"]["edge"]["classification"]["focal_gamma"], 1.0)
        self.assertEqual(focal["topology"]["complex"]["alpha"], .75)
        ce = apply_parameters(base, {"edge_loss": "cross_entropy"})
        self.assertEqual(ce["loss"]["edge"]["classification"]["name"], "cross_entropy")

    def test_hungarian_and_ordinary_validation(self):
        for path in (SMOKE, STUDY_A, STUDY_B):
            base, _ = load_campaign(path, ENVIRONMENT)
            self.assertEqual(base["model"]["matcher"]["type"], "hungarian")
            validate_config(base)

    def test_study_b_proposal_uses_selected_unique_values(self):
        trials = []
        for number, h0 in ((2, .001), (5, .003)):
            trials.append({"number": number, "topology.betti_h0.weight": h0,
                "topology.betti_h1.weight": .01,
                "topology.betti_h1.false_positive_weight": .25,
                "betti_warmup_epochs": 20, "betti_ramp_epochs": 30})
        summary = {"trials": trials, "representatives": {
            "best_node_mAP": {"trial": 2}, "balanced": {"trial": 5}}}
        space, selected = proposed_space(summary)
        self.assertEqual(selected, [2, 5])
        self.assertEqual(space["topology.betti_h0.weight"], [.001, .003])
        self.assertNotIn("model.matcher.structure_weight", space)

    def test_final_pair_is_full_data_paired_and_fixed_epoch(self):
        base, _ = load_campaign(SMOKE, ENVIRONMENT)
        with tempfile.TemporaryDirectory() as directory:
            plan = prepare_final_pair(base, Path(directory), [364505])
            self.assertEqual(plan["checkpoint_rule"], "fixed epoch 500 for both arms")
            configs = [json.loads(json.dumps(__import__("yaml").safe_load(Path(run["config"]).read_text())))
                       for run in plan["runs"]]
            self.assertTrue(all(c["data"]["datasets"]["synthetic_mri"]["train_samples"] is None for c in configs))
            self.assertEqual(configs[0]["loss"], configs[1]["loss"])
            self.assertEqual(configs[0]["training"]["epochs"], 500)


class AggregationTests(unittest.TestCase):
    def test_exact_tail_mean_three_and_epochs(self):
        records = [metric_record(i, node_mAP=float(i)) for i in range(1, 6)]
        result = aggregate_tail(records, 3)
        self.assertEqual(result["method"], "tail_mean_3")
        self.assertEqual(result["contributing_epochs"], [3, 4, 5])
        self.assertEqual(result["metrics"]["node_mAP"], 4.0)
        self.assertEqual(result["final_epoch"], 5)

    def test_fewer_than_tail_fails_but_smoke_tail_one_works(self):
        with self.assertRaisesRegex(CampaignError, "requires 3"):
            aggregate_tail([metric_record(1), metric_record(2)], 3)
        self.assertEqual(aggregate_tail([metric_record(1)], 1)["objectives"], (.8, .7, 2., 4.))

    def test_missing_and_nonfinite_metrics_rejected(self):
        missing = metric_record(1); missing.pop("node_mAP")
        with self.assertRaises(CampaignError): aggregate_tail([missing], 1)
        with self.assertRaises(CampaignError): aggregate_tail([metric_record(1, edge_mAP=float("nan"))], 1)

    def test_no_best_epoch_selection(self):
        records = [metric_record(1, node_mAP=.99), metric_record(2, node_mAP=.4),
                   metric_record(3, node_mAP=.5)]
        result = aggregate_tail(records, 2)
        self.assertAlmostEqual(result["metrics"]["node_mAP"], .45)
        self.assertNotEqual(result["metrics"]["node_mAP"], .99)


class ParetoTests(unittest.TestCase):
    def test_dominance_equality_and_incomparability(self):
        a, b = row(0, .8, .7, 2, 3), row(1, .7, .7, 2, 4)
        self.assertTrue(dominates(a, b)); self.assertFalse(dominates(a, dict(a)))
        c = row(2, .9, .6, 3, 2)
        self.assertFalse(dominates(a, c)); self.assertFalse(dominates(c, a))

    def test_front_excludes_failed_and_keeps_incomparable(self):
        rows = [row(0, .8, .7, 2, 3), row(1, .7, .7, 2, 4),
                row(2, .9, .6, 3, 2), row(3, 1, 1, 0, 0, "FAIL")]
        self.assertEqual([x["number"] for x in pareto_front(rows)], [0, 2])

    def test_anchors_balanced_zero_range_and_ties(self):
        front = [row(2, .8, .7, 2, 4), row(1, .9, .6, 3, 2)]
        reps = representatives(front)
        self.assertEqual(reps["best_node_mAP"]["trial"], 1)
        self.assertEqual(reps["best_edge_mAP"]["trial"], 2)
        equal = [row(4, .8, .7, 2, 4), row(3, .8, .7, 2, 4)]
        self.assertEqual(balanced_representative(equal)["trial"]["number"], 3)
        self.assertTrue(all(v == 1 for v in balanced_representative(equal)["normalized_objectives"].values()))


class ProcessTests(unittest.TestCase):
    def test_incremental_monitor_and_malformed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation-metrics.jsonl"
            path.write_text('{"epoch":1}\n{"epoch":')
            monitor = JsonlMonitor(path); self.assertEqual(monitor.read_new(), [{"epoch": 1}])
            with path.open("a") as handle: handle.write("2}\n")
            self.assertEqual(monitor.read_new(), [{"epoch": 2}])
            path2 = Path(directory) / "bad"; path2.write_text("bad\n")
            with self.assertRaises(CampaignError): JsonlMonitor(path2).read_new()

    def test_interrupted_process_cleanup_preserves_artifact(self):
        class Stop(Exception): pass
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            code = ("import json,time,pathlib;p=pathlib.Path(r'%s');"
                    "(p/'keep').write_text('yes');f=(p/'validation-metrics.jsonl').open('w');"
                    "f.write(json.dumps({'epoch':1})+'\\n');f.flush();time.sleep(20)") % run
            def callback(_record, process): process.terminate(); raise Stop()
            with self.assertRaises(Stop): run_command([sys.executable, "-c", code], run, callback, .01)
            self.assertEqual((run / "keep").read_text(), "yes")

    def test_sqlite_controller_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".lock"
            with ControllerLock(path):
                with self.assertRaises(CampaignError):
                    with ControllerLock(path): pass


OPTUNA = importlib.util.find_spec("optuna") is not None


@unittest.skipUnless(OPTUNA, "Optuna not installed")
class OptunaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.environment = dict(os.environ); self.environment.update(ENVIRONMENT)

    def command(self, mode, output, checkpoint, storage="sqlite", worker_trials=None):
        command = [sys.executable, str(ROOT / "scripts/optimize_node_edge_betti.py"), mode,
            "--config", str(SMOKE), "--output", str(output), "--initial-weights", str(checkpoint),
            "--storage", storage, "--poll-interval", ".01"]
        if worker_trials is not None: command += ["--worker-trials", str(worker_trials)]
        if mode in {"control", "worker"}:
            command += ["--train-command", sys.executable,
                str(ROOT / "tests/fixtures/fake_optuna_train.py"), "--config", "{config}",
                "--output-dir", "{output}", "--run-name", "{run_name}",
                "--initial-weights", "{initial_weights}"]
        return command

    def test_deterministic_nsga2(self):
        import optuna
        def sequence():
            sampler = build_sampler(optuna, {"seed": 364505, "sampler": {"population_size": 4}})
            study = optuna.create_study(directions=["maximize"] * 2, sampler=sampler)
            study.optimize(lambda t: (t.suggest_categorical("x", [0, 1, 2]), t.number), n_trials=8)
            return [t.params["x"] for t in study.trials]
        self.assertEqual(sequence(), sequence())

    def test_journal_storage_creation(self):
        import optuna
        with tempfile.TemporaryDirectory() as directory:
            storage = create_storage(optuna, Path(directory), "journal")
            study = optuna.create_study(study_name="journal", storage=storage,
                                         directions=[x[1] for x in OBJECTIVES])
            self.assertEqual(study.study_name, "journal")

    def test_explicit_stale_running_recovery(self):
        import optuna
        study = optuna.create_study(directions=[x[1] for x in OBJECTIVES])
        running = study.ask(); running.set_user_attr("run_dir", "/retained")
        self.assertEqual(recover_stale_trials(optuna, study), 1)
        self.assertEqual(study.trials[0].state.name, "FAIL")
        self.assertEqual(study.trials[0].user_attrs["run_dir"], "/retained")

    def test_end_to_end_tail_resume_summary_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "study"
            checkpoint = root / "initial.pt"; checkpoint.write_bytes(b"fixed")
            subprocess.run(self.command("control", output, checkpoint), cwd=ROOT, env=self.environment, check=True)
            subprocess.run(self.command("worker", output, checkpoint), cwd=ROOT, env=self.environment, check=True)
            before = sorted((output / "runs").iterdir())
            subprocess.run(self.command("worker", output, checkpoint), cwd=ROOT, env=self.environment, check=True)
            self.assertEqual(before, sorted((output / "runs").iterdir()))
            report = write_summary(output, "node-edge-betti-pareto-smoke")
            self.assertEqual(report["trial_count"], 4)
            self.assertTrue(all(len(t["aggregation" if False else "contributing_epochs"]) == 2 for t in report["trials"]))
            for name in ("summary.json", "trials.csv", "pareto-front.csv", "pareto-front.json",
                         "representative-configurations.json", "representative-configurations.md",
                         "control-reference.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue(all("delta_node_mAP" in trial for trial in report["trials"]))
            self.assertTrue(all((Path(trial["run_dir"]) / "resolved-config.yaml").is_file() for trial in report["trials"]))

    def test_failed_subprocess_records_failed_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "study"
            checkpoint = root / "initial.pt"; checkpoint.write_bytes(b"fixed")
            subprocess.run(self.command("control", output, checkpoint), cwd=ROOT, env=self.environment, check=True)
            failed_env = dict(self.environment); failed_env["FAKE_TRAIN_FAIL"] = "1"
            subprocess.run(self.command("worker", output, checkpoint, worker_trials=1), cwd=ROOT, env=failed_env, check=True)
            import optuna
            study = optuna.load_study(study_name="node-edge-betti-pareto-smoke",
                                      storage=create_storage(optuna, output, "sqlite"))
            self.assertEqual(study.trials[0].state.name, "FAIL")
            self.assertIn("status 3", study.trials[0].user_attrs["failure_reason"])

    def test_two_journal_workers_get_unique_trials_and_respect_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory) / "study"
            checkpoint = root / "initial.pt"; checkpoint.write_bytes(b"fixed")
            subprocess.run(self.command("control", output, checkpoint, "journal"), cwd=ROOT, env=self.environment, check=True)
            commands = [self.command("worker", output, checkpoint, "journal", 1) for _ in range(2)]
            processes = [subprocess.Popen(command, cwd=ROOT, env=self.environment) for command in commands]
            self.assertTrue(all(process.wait() == 0 for process in processes))
            import optuna
            study = optuna.load_study(study_name="node-edge-betti-pareto-smoke",
                                      storage=create_storage(optuna, output, "journal"))
            self.assertEqual([trial.number for trial in study.trials], [0, 1])
            self.assertEqual(len({trial.user_attrs["run_dir"] for trial in study.trials}), 2)
            report1 = write_summary(output, study.study_name, "journal")
            report2 = write_summary(output, study.study_name, "journal")
            self.assertEqual(report1["pareto_trial_numbers"], report2["pareto_trial_numbers"])

    def test_independent_front_agrees_with_optuna(self):
        import optuna
        study = optuna.create_study(directions=[x[1] for x in OBJECTIVES])
        study.enqueue_trial({})
        study.optimize(lambda _t: (.8, .7, 2., 3.), n_trials=1)
        study.optimize(lambda _t: (.7, .7, 2., 4.), n_trials=1)
        rows = [row(t.number, *t.values) for t in study.trials]
        self.assertEqual({r["number"] for r in pareto_front(rows)}, {t.number for t in study.best_trials})


if __name__ == "__main__": unittest.main()
