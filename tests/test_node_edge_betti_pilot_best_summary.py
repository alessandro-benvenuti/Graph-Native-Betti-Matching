"""Tests for the best-validation paired Betti pilot report."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.summarize_node_edge_betti_pilot_4000 import FINAL_EPOCH, RUNS
from scripts.summarize_node_edge_betti_pilot_4000_best import (
    DIAGNOSTIC_METRICS,
    PRIMARY_METRICS,
    load_best_results,
    render_best,
)


class NodeEdgeBettiPilotBestSummaryTests(unittest.TestCase):
    def test_script_can_be_executed_directly(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "summarize_node_edge_betti_pilot_4000_best.py"),
                "--help",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-root", completed.stdout)

    def _write_run(
        self, root: Path, run_name: str, *, best_epoch: int, offset: float
    ) -> None:
        run = root / run_name
        run.mkdir()
        (run / "training-status.json").write_text(
            json.dumps({"epoch": FINAL_EPOCH, "reason": "max_epochs"}),
            encoding="utf-8",
        )
        records = []
        for epoch, edge_map in ((5, 0.2 + offset), (best_epoch, 0.8 + offset)):
            record = {
                "epoch": epoch,
                **{name: index + offset for index, name in enumerate(PRIMARY_METRICS)},
                **{
                    name: index + 20.0 + offset
                    for index, name in enumerate(DIAGNOSTIC_METRICS)
                },
            }
            record["edge_mAP"] = edge_map
            records.append(record)
        (run / "validation-metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (run / "best-metric.json").write_text(
            json.dumps(
                {
                    "epoch": best_epoch,
                    "metric": "edge_mAP",
                    "mode": "max",
                    "value": 0.8 + offset,
                    "checkpoint": "models/best_metric_checkpoint.pt",
                }
            ),
            encoding="utf-8",
        )

    def test_uses_recorded_best_checkpoint_epoch_and_renders_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, RUNS["control"], best_epoch=80, offset=0.0)
            self._write_run(
                root, RUNS["node-aware Betti"], best_epoch=95, offset=0.1
            )
            results = load_best_results(root)
            report = render_best(results)

        self.assertEqual(results["control"]["epoch"], 80)
        self.assertEqual(results["node-aware Betti"]["epoch"], 95)
        self.assertIn("maximum edge_mAP", report)
        self.assertIn("selected_epoch", report)
        self.assertIn("predicted_nodes", report)
        self.assertIn("+0.100000", report)

    def test_rejects_selection_history_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, RUNS["control"], best_epoch=80, offset=0.0)
            self._write_run(
                root, RUNS["node-aware Betti"], best_epoch=95, offset=0.1
            )
            selection = root / RUNS["control"] / "best-metric.json"
            payload = json.loads(selection.read_text(encoding="utf-8"))
            payload["value"] = -1.0
            selection.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_best_results(root)


if __name__ == "__main__":
    unittest.main()
