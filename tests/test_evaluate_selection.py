"""Tests for explicit source-patch selection in evaluate.py."""

from pathlib import Path
import tempfile
import unittest

from evaluate import _parser, _selected_sample_ids


class EvaluationSelectionTests(unittest.TestCase):
    @staticmethod
    def _required_arguments():
        return [
            "--config", "config.yaml",
            "--checkpoint", "checkpoint.pt",
            "--output-dir", "output",
        ]

    def test_repeated_sample_ids_are_collected(self):
        args = _parser().parse_args(
            self._required_arguments()
            + ["--sample-id", "sample_b", "--sample-id", "sample_a"]
        )
        self.assertEqual(_selected_sample_ids(args), ["sample_b", "sample_a"])

    def test_sample_list_ignores_blanks_and_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.txt"
            path.write_text(
                "# shared random manifest\nsample_b\n\n sample_a \n",
                encoding="utf-8",
            )
            args = _parser().parse_args(
                self._required_arguments() + ["--sample-list", str(path)]
            )
            self.assertEqual(_selected_sample_ids(args), ["sample_b", "sample_a"])

    def test_selection_modes_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                self._required_arguments()
                + ["--max-samples", "2", "--sample-id", "sample_a"]
            )


if __name__ == "__main__":
    unittest.main()
