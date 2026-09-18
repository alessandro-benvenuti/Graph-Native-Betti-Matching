"""Tests for the paired 4,000-patch Betti pilot report."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_node_edge_betti_pilot_4000 import (
    FINAL_EPOCH,
    METRICS,
    RUNS,
    load_results,
    render,
)


class NodeEdgeBettiPilotSummaryTests(unittest.TestCase):
    def _write_run(self, root: Path, run_name: str, offset: float):
        run = root / run_name
        run.mkdir()
        (run / "training-status.json").write_text(
            json.dumps({"epoch": FINAL_EPOCH, "reason": "max_epochs"}),
            encoding="utf-8",
        )
        record = {
            "epoch": FINAL_EPOCH,
            **{name: index + offset for index, name in enumerate(METRICS)},
        }
        (run / "validation-metrics.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    def test_renders_paired_final_epoch_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, RUNS["control"], 0.0)
            self._write_run(root, RUNS["node-aware Betti"], 0.25)
            report = render(load_results(root))

        self.assertIn("Final epoch-100 validation comparison", report)
        self.assertIn("+0.250000", report)
        self.assertIn("not statistical significance", report)

    def test_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, RUNS["control"], 0.0)
            with self.assertRaisesRegex(ValueError, "Incomplete run artifacts"):
                load_results(root)


if __name__ == "__main__":
    unittest.main()
