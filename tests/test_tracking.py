"""Unit tests for W&B tracking without network access."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from training.tracking import build_tracker


class _FakeRun:
    def __init__(self, run_id="run-123"):
        self.id = run_id
        self.name = "test-run"
        self.project = "test-project"
        self.entity = "test-entity"
        self.url = "https://wandb.example/run-123"
        self.defined_metrics = []
        self.logged = []
        self.exit_codes = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


class _FakeWandb:
    def __init__(self, run):
        self.run = run
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def _config():
    return {
        "experiment": {"name": "test-run"},
        "tracking": {
            "enabled": True,
            "project": "configured-project",
            "entity": None,
            "group": None,
            "tags": ["gnbm", "test"],
            "mode": None,
            "save_code": True,
        },
    }


class WandbTrackingTests(unittest.TestCase):
    def test_initializes_logs_complete_records_and_finishes(self):
        fake_run = _FakeRun()
        fake_wandb = _FakeWandb(fake_run)
        environment = {
            "WANDB_PROJECT": "environment-project",
            "WANDB_ENTITY": "research-team",
            "WANDB_RUN_GROUP": "ablation",
            "WANDB_MODE": "offline",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"wandb": fake_wandb}
        ), patch.dict(os.environ, environment, clear=False):
            tracker = build_tracker(
                _config(),
                Path(directory),
                resume=False,
                launch_metadata={"slurm_job_id": "42"},
            )
            tracker.log_training(
                {"total": 1.5, "edges": 0.25},
                iteration=7,
                epoch=2,
                learning_rate=1.0e-4,
            )
            tracker.log_validation({"total": 1.25}, iteration=8, epoch=2)
            tracker.finish(exit_code=0)
            metadata = json.loads(
                (Path(directory) / "wandb-run.json").read_text(encoding="utf-8")
            )

        init = fake_wandb.init_calls[0]
        self.assertEqual(init["project"], "environment-project")
        self.assertEqual(init["entity"], "research-team")
        self.assertEqual(init["group"], "ablation")
        self.assertEqual(init["mode"], "offline")
        self.assertEqual(init["config"]["launch"]["slurm_job_id"], "42")
        self.assertIsNone(init["id"])
        self.assertIsNone(init["resume"])
        self.assertEqual(len(fake_run.logged), 2)
        self.assertEqual(fake_run.logged[0]["train/iteration"], 7)
        self.assertEqual(fake_run.logged[0]["train/total"], 1.5)
        self.assertEqual(fake_run.logged[1]["validation/epoch"], 2)
        self.assertEqual(fake_run.exit_codes, [0])
        self.assertEqual(metadata["id"], "run-123")

    def test_resume_reuses_persisted_run_id(self):
        fake_run = _FakeRun("existing-id")
        fake_wandb = _FakeWandb(fake_run)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "wandb-run.json").write_text(
                json.dumps({"id": "existing-id"}), encoding="utf-8"
            )
            with patch.dict(sys.modules, {"wandb": fake_wandb}), patch.dict(
                os.environ,
                {
                    "WANDB_PROJECT": "",
                    "WANDB_ENTITY": "",
                    "WANDB_RUN_GROUP": "",
                    "WANDB_MODE": "disabled",
                },
                clear=False,
            ):
                build_tracker(_config(), run_dir, resume=True)

        init = fake_wandb.init_calls[0]
        self.assertEqual(init["id"], "existing-id")
        self.assertEqual(init["resume"], "allow")

    def test_disabled_configuration_does_not_import_wandb(self):
        config = _config()
        config["tracking"]["enabled"] = False
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(build_tracker(config, Path(directory), resume=False))


if __name__ == "__main__":
    unittest.main()
