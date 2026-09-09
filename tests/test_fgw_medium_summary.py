"""Tests for the medium-scale FGW comparison report."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_fgw_medium_confirmation import (
    METRICS,
    load_results,
    render_summary,
    run_name,
)


class FgwMediumSummaryTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int, offset: float) -> None:
        run = root / run_name(method, seed)
        run.mkdir()
        (run / "training-status.json").write_text(
            json.dumps(
                {
                    "reason": "execution_stop",
                    "epoch": 100,
                    "max_epochs": 100,
                    "stop_after_epoch": 100,
                }
            ),
            encoding="utf-8",
        )
        earlier = {"epoch": 28, **{name: 0.0 for name in METRICS}}
        final = {
            "epoch": 100,
            **{
                name: index + seed % 10 + offset
                for index, name in enumerate(METRICS)
            },
        }
        (run / "validation-metrics.jsonl").write_text(
            json.dumps(earlier) + "\n" + json.dumps(final) + "\n",
            encoding="utf-8",
        )

    def test_report_contains_final_rows_aggregate_and_paired_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (364505, 364506, 364507):
                self._write_run(root, "hungarian", seed, 0.0)
                self._write_run(root, "fgw_a04", seed, 0.25)
            report = render_summary(load_results(root))

        self.assertIn("Final epoch-100 results", report)
        self.assertIn("364505", report)
        self.assertIn("FGW alpha=.4", report)
        self.assertIn("Three-seed mean +/- sample standard deviation", report)
        self.assertIn("delta node_mAP             +0.250000 +/- 0.000000", report)

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (364505, 364506, 364507):
                self._write_run(root, "hungarian", seed, 0.0)
                self._write_run(root, "fgw_a04", seed, 0.25)
            missing = (
                root
                / run_name("fgw_a04", 364507)
                / "training-status.json"
            )
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "Missing completion status"):
                load_results(root)


if __name__ == "__main__":
    unittest.main()
