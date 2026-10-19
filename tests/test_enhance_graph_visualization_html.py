import tempfile
import unittest
from pathlib import Path

from scripts.enhance_graph_visualization_html import enhance_file, enhance_html


class GraphVisualizationEnhancementTests(unittest.TestCase):
    def test_injects_controls_once(self):
        original = '<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
        enhanced = enhance_html(original)
        self.assertIn('id="gnbm-paper-controls"', enhanced)
        self.assertIn("MRI slices", enhanced)
        self.assertIn("Plotly.downloadImage", enhanced)
        self.assertIn("clean['showlegend'] = false", enhanced)
        self.assertEqual(enhance_html(enhanced), enhanced)

    def test_file_update_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.html"
            path.write_text("<html><head></head><body></body></html>")
            self.assertTrue(enhance_file(path))
            self.assertFalse(enhance_file(path))


if __name__ == "__main__":
    unittest.main()
