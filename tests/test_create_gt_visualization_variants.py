import tempfile
import unittest
from pathlib import Path

from scripts.create_gt_visualization_variants import MODE_META, create_variants


class GroundTruthVisualizationVariantTests(unittest.TestCase):
    def test_creates_gt_sibling_from_each_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "sample_a" / "baseline"
            baseline.mkdir(parents=True)
            baseline.joinpath("sample.html").write_text(
                '<html><head></head><body><div class="plotly-graph-div"></div></body></html>',
                encoding="utf-8",
            )
            outputs = create_variants(root)
            self.assertEqual(outputs, [root / "sample_a" / "GT" / "sample.html"])
            result = outputs[0].read_text(encoding="utf-8")
            self.assertIn(MODE_META, result)
            self.assertIn('id="gnbm-paper-controls"', result)


if __name__ == "__main__":
    unittest.main()
