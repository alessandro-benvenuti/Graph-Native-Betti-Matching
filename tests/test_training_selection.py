"""Checkpoint selection, patience, and two-rank stopping without a GPU."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_training import _config, _batch, TinyGraphModel, RecordingTracker
from training.engine import Trainer
from training.optim import build_optimizer, build_scheduler
from training.checkpoint import load_runtime_state, load_training_checkpoint


def setup(directory):
    config = _config()
    config["experiment"].update(output_dir=directory, name="selection")
    config["training"].update(epochs=10, early_stopping={
        "enabled": True, "monitor": "edge_mAP", "mode": "max",
        "patience_epochs": 2, "min_delta": 0.0,
    })
    config["training"]["checkpoint"].update(policy="best_only", latest_interval_epochs=7)
    config["evaluation"]["interval_epochs"] = 1
    config["evaluation"]["training_metrics"].update(
        enabled=True, save_best_checkpoint=True, save_f1_checkpoints=True
    )
    model = TinyGraphModel()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, 1)
    tracker = RecordingTracker()
    trainer = Trainer(model, None, None, optimizer, scheduler, [_batch()], [_batch()],
                      config, "cpu", tracker, [_batch()])
    return trainer


def metrics(ap=0.5, node=0.6, edge=0.7):
    return {"edge_mAP": ap, "node_mAP": 0.4, "node_f1": node, "edge_f1": edge}, []


def fit_mocked(trainer, series, **kwargs):
    with patch("training.engine.train_step", return_value={"total": torch.tensor(1.0)}) as training_step, \
         patch("training.engine.evaluate_loss", return_value={"total": 1.0}), \
         patch("training.engine.evaluate_model", side_effect=series):
        trainer.fit(**kwargs)
        return training_step.call_count


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + directory + "/rendezvous",
                            world_size=2, rank=rank)
    try:
        trainer = setup(directory)
        trainer.rank, trainer.world_size = rank, 2
        if rank:
            trainer.metric_loader = None
        steps = fit_mocked(trainer, [metrics()] * 3)
        # Each rank must leave the loop after the SAME three iterations.
        Path(directory, "rank-{}.json".format(rank)).write_text(
            json.dumps({"steps": steps})
        )
    finally:
        dist.destroy_process_group()


class TrainingSelectionTests(unittest.TestCase):
    def test_separate_winners_and_forced_final_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = setup(directory)
            fit_mocked(trainer, [metrics(), metrics(node=0.8, edge=0.6), metrics(node=0.7, edge=0.9)])
            run = Path(directory) / "selection"
            self.assertEqual(len(trainer.tracker.training), 3)
            for name, epoch in (("best_metric", 1), ("best_node_f1", 2),
                                ("best_edge_f1", 3), ("latest", 3)):
                checkpoint = load_runtime_state(run / "models" / (name + "_checkpoint.pt"))
                self.assertEqual(checkpoint["epoch"], epoch)
            for name, epoch in (("best-metric", 1), ("best-node-f1", 2), ("best-edge-f1", 3)):
                record = json.loads((run / (name + ".json")).read_text())
                self.assertEqual(record["epoch"], epoch)
                self.assertTrue((run / record["checkpoint"]).is_file())
            last = load_runtime_state(run / "models/latest_checkpoint.pt")
            self.assertTrue(last["trainer_state"]["early_stopping"]["stopped"])
            status = json.loads((run / "training-status.json").read_text())
            self.assertEqual(status["reason"], "early_stopping")
            self.assertEqual(len(list((run / "models").glob("*.pt"))), 5)
            # Coincident edge-F1/latest winners occupy one payload on this FS.
            self.assertEqual((run / "models/best_edge_f1_checkpoint.pt").stat().st_ino,
                             (run / "models/latest_checkpoint.pt").stat().st_ino)

    def test_interrupted_resume_preserves_winners_and_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            first = setup(directory)
            first.config["training"]["checkpoint"]["latest_interval_epochs"] = 1
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                fit_mocked(first, [metrics(), metrics(node=0.8), RuntimeError("interrupted")])
            run = Path(directory) / "selection"
            resumed = setup(directory)
            epoch, iteration, state = load_training_checkpoint(
                run / "models/latest_checkpoint.pt", resumed.model,
                resumed.optimizer, resumed.scheduler, return_trainer_state=True,
            )
            fit_mocked(resumed, [metrics(node=0.7)], start_epoch=epoch,
                       start_iteration=iteration, trainer_state=state)
            self.assertEqual(len(resumed.tracker.training), 1)
            self.assertEqual(json.loads((run / "best-node-f1.json").read_text())["epoch"], 2)
            latest = load_runtime_state(run / "models/latest_checkpoint.pt")
            self.assertEqual(latest["epoch"], 3)
            self.assertTrue(latest["trainer_state"]["early_stopping"]["stopped"])

    def test_early_stopped_checkpoint_does_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = setup(directory)
            fit_mocked(trainer, [metrics()] * 3)
            checkpoint = load_runtime_state(Path(directory) / "selection/models/latest_checkpoint.pt")
            resumed = setup(directory)
            fit_mocked(resumed, [], start_epoch=3, start_iteration=3,
                       trainer_state=checkpoint["trainer_state"])
            self.assertEqual(len(resumed.tracker.training), 0)

    def test_f1_monitor_can_be_selected_independently_of_ap_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = setup(directory)
            trainer.config["training"]["early_stopping"]["monitor"] = "edge_f1"
            fit_mocked(trainer, [metrics(ap=0.1), metrics(ap=0.2), metrics(ap=0.3)])
            record = load_runtime_state(Path(directory) / "selection/models/latest_checkpoint.pt")
            self.assertEqual(record["epoch"], 3)
            self.assertEqual(record["trainer_state"]["early_stopping"]["best_epoch"], 1)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_ranks_stop_together_and_persist_both_rng_states(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)
            for rank in (0, 1):
                record = json.loads(Path(directory, "rank-{}.json".format(rank)).read_text())
                self.assertEqual(record["steps"], 3)
            checkpoint = load_runtime_state(Path(directory) / "selection/models/latest_checkpoint.pt")
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(len(checkpoint["runtime_states"]), 2)
            self.assertFalse((Path(directory) / "selection/models/.runtime_states").exists())


if __name__ == "__main__":
    unittest.main()
