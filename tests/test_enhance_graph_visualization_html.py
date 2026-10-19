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
        self.assertIn("'marker.showscale': false", enhanced)
        self.assertIn("clean['scene.domain.x'] = [0, 1]", enhanced)
        self.assertIn("clean[`margin.${side}`] = 0", enhanced)
        self.assertIn("name: 'Predicted loop edges'", enhanced)
        self.assertIn("line: {color: '#ff3300', width: group.width}", enhanced)
        self.assertIn("width: Number(trace.line?.width ?? 3)", enhanced)
        self.assertIn("linecolor`] = '#000000'", enhanced)
        self.assertIn("backgroundcolor`] = 'rgba(229,236,246,0.5)'", enhanced)
        self.assertIn("paper_bgcolor: 'rgba(255,255,255,0.5)'", enhanced)
        self.assertIn("trace.name?.startsWith('Ground truth ')", enhanced)
        self.assertEqual(enhance_html(enhanced), enhanced)

    def test_file_update_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.html"
            path.write_text("<html><head></head><body></body></html>")
            self.assertTrue(enhance_file(path))
            self.assertFalse(enhance_file(path))

    def test_refreshes_existing_controls_script(self):
        original = '<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
        enhanced = enhance_html(original)
        stale = enhanced.replace("clean['scene.domain.x'] = [0, 1];", "// stale script")
        refreshed = enhance_html(stale)
        self.assertIn("clean['scene.domain.x'] = [0, 1]", refreshed)
        self.assertNotIn("// stale script", refreshed)


if __name__ == "__main__":
    unittest.main()
