"""Regression coverage for epoch matching and genuine/false loop attribution."""

import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from data.loaders.common import SamplePaths
from scripts.analyze_staged_betti_loops import (
    build_report, checkpoint_epoch, freeze, infer_model, prepare_inputs, recovery_counts,
)
from scripts.compare_loop_patch_predictions import compare_predictions


class StagedLoopAuditTests(unittest.TestCase):
    def test_cycle_suppression_is_distinct_from_genuine_recovery(self):
        rows = [
            {"control_shared_cycle_rank": 0, "betti_shared_cycle_rank": 1,
             "control_false_cycle_rank": 3, "betti_false_cycle_rank": 1},
            {"control_shared_cycle_rank": 2, "betti_shared_cycle_rank": 1,
             "control_false_cycle_rank": 0, "betti_false_cycle_rank": 1},
        ]
        self.assertEqual(recovery_counts(rows), {
            "genuine_rank_gained": 1, "genuine_rank_lost": 1,
            "false_rank_removed": 2, "false_rank_added": 1,
        })

    def test_nonloop_audit_is_explicit_opt_in(self):
        nodes = np.asarray([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], dtype=np.float32)
        path = [[0, 1], [1, 2]]
        triangle = [*path, [0, 2]]
        prediction = {"source_sample_id": "tree", "nodes_dhw": nodes.tolist(), "edges": triangle}
        corrected = {**prediction, "edges": path}
        dataset = [SamplePaths(Path("raw"), Path("seg"), Path("graph"), "tree")]
        def reader(_):
            return nodes, np.asarray(path)
        with self.assertRaisesRegex(ValueError, "non-loop"):
            compare_predictions([prediction], [corrected], dataset, graph_reader=reader)
        rows = compare_predictions([prediction], [corrected], dataset, graph_reader=reader, require_gt_loops=False)
        self.assertEqual(recovery_counts(rows)["false_rank_removed"], 1)
        self.assertEqual(recovery_counts(rows)["genuine_rank_gained"], 0)

    def test_wrong_epoch_refused(self):
        with patch("scripts.analyze_staged_betti_loops.checkpoint_epoch", return_value=350):
            with self.assertRaisesRegex(ValueError, "not at epoch 400"):
                prepare_inputs(Path("/unused"), 400)

    def test_real_checkpoint_epoch(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"epoch": 400}, path)
            self.assertEqual(checkpoint_epoch(path), 400)

    def test_changed_inputs_cannot_reuse_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            freeze(path, {"epoch": 400})
            freeze(path, {"epoch": 400})
            with self.assertRaisesRegex(ValueError, "Inputs changed"):
                freeze(path, {"epoch": 350})
            self.assertEqual(json.loads(path.read_text()), {"epoch": 400})

    def test_failed_inference_is_retried_and_complete_export_is_reused(self):
        args = SimpleNamespace(epoch=400, batch_size=32, workers=8)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = {"config": "config.yaml", "checkpoint": "checkpoint.pt"}
            with patch("scripts.analyze_staged_betti_loops.subprocess.run", side_effect=subprocess.CalledProcessError(1, "evaluate")):
                with self.assertRaises(subprocess.CalledProcessError):
                    infer_model(directory, source, args, [])
            self.assertFalse((directory / "inference-complete.json").exists())
            (directory / "predictions.json").write_text("[]")
            with patch("scripts.analyze_staged_betti_loops.subprocess.run") as run:
                infer_model(directory, source, args, [])
                infer_model(directory, source, args, [])
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertIn("--export-all-edge-scores", command)
                self.assertEqual(command[command.index("--bn-calibration-batches") + 1], "0")

    def test_mismatched_thresholds_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            for relative in ("shared-prefix", "promotion-0400/O07", "promotion-0400/S07"):
                directory = campaign / relative
                directory.mkdir(parents=True)
                (directory / "resolved-config.yaml").write_text("config")
                (directory / "training-complete").touch()
                checkpoint = directory / ("checkpoints/epoch_0400.pt" if relative == "shared-prefix" else "models/latest_checkpoint.pt")
                checkpoint.parent.mkdir()
                checkpoint.touch()
            configs = [
                {"evaluation": {"node_threshold": .5, "edge_threshold": .5},
                 "data": {"datasets": {"synthetic_mri": {"root": temporary}}}},
                {"evaluation": {"node_threshold": .5, "edge_threshold": .4}},
            ]
            with patch("configs.load_config", side_effect=configs), patch("scripts.analyze_staged_betti_loops.checkpoint_epoch", return_value=400):
                with self.assertRaisesRegex(ValueError, "different inference thresholds"):
                    prepare_inputs(campaign, 400)

    def test_launcher_submits_one_gpu_job_for_requested_scope(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("promotion-0400/O07", "promotion-0400/S07"):
                run = root / relative
                (run / "models").mkdir(parents=True)
                for artifact in ("training-complete", "resolved-config.yaml", "models/latest_checkpoint.pt"):
                    (run / artifact).touch()
            (root / "shared-prefix/checkpoints").mkdir(parents=True)
            (root / "shared-prefix/checkpoints/epoch_0400.pt").touch()
            (root / "shared-prefix/resolved-config.yaml").touch()
            executable = root / "sbatch"
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nprintf "scope=%s\\n" "$GNBM_LOOP_AUDIT_SCOPE"\necho "Submitted batch job 123"\n')
            executable.chmod(0o755)
            environment = {**os.environ, "WORK": temporary, "SYNTHETIC_MRI_DATASET": temporary,
                           "GNBM_OUTPUT_DIR": temporary, "PATH": temporary + os.pathsep + os.environ["PATH"]}
            result = subprocess.run(["bash", str(repository / "cluster/jean_zay/submit_staged_betti_loop_audit.sh"), "all-validation"], env=environment, text=True, capture_output=True, check=True)
            self.assertIn("scope=all-validation", result.stdout)
            self.assertIn("--time=04:00:00", result.stdout)
            self.assertIn("Submitted batch job 123", result.stdout)

    def test_report_reuses_spatial_matcher_and_reports_cohorts(self):
        nodes = np.asarray([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], dtype=np.float32)
        path = [[0, 1], [1, 2]]
        triangle = [*path, [0, 2]]
        dataset = [SamplePaths(Path("raw"), Path("seg"), Path("graph"), "loop"),
                   SamplePaths(Path("raw"), Path("seg"), Path("tree"), "tree")]
        def prediction(edges):
            return [{"source_sample_id": "loop", "nodes_dhw": nodes.tolist(), "edges": edges}]
        predictions = {"control": prediction(path), "O07": prediction(triangle), "S07": prediction(path)}
        for label in predictions:
            predictions[label].append({"source_sample_id": "tree", "nodes_dhw": nodes.tolist(),
                                       "edges": triangle if label == "control" else path})
        # Keep the real matcher and summaries; stub only GT disk I/O.
        original_compare = compare_predictions
        def compare(*args, **kwargs):
            return original_compare(*args, graph_reader=lambda graph: (nodes, np.asarray(path if str(graph) == "tree" else triangle)), **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.compare_loop_patch_predictions.compare_predictions", side_effect=compare):
                build_report(output, predictions, dataset, .1, "all-validation")
            report = json.loads((output / "summary.json").read_text())
            result = report["candidates"]["O07"]["target_loop"]
            self.assertEqual(result["rank_changes"]["genuine_rank_gained"], 1)
            self.assertEqual(result["methods"]["betti"]["cycle_recall"], 1.)
            self.assertEqual(report["candidates"]["O07"]["target_nonloop"]["rank_changes"]["false_rank_removed"], 1)
            self.assertTrue((output / "comparison/O07/per-patch.json").is_file())


if __name__ == "__main__":
    unittest.main()
