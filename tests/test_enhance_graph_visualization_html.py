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
        self.assertIn("legendgroup: 'prediction-loops'", enhanced)
        self.assertIn("visible: 'legendonly'", enhanced)
        self.assertIn("x: -1.7506160977336473", enhanced)
        self.assertIn('id="gnbm-node-scale"', enhanced)
        self.assertIn('id="gnbm-edge-scale"', enhanced)
        self.assertIn('value="1"', enhanced)
        self.assertIn("initializeSizeControls()", enhanced)
        self.assertIn("await applyNodeScale()", enhanced)
        self.assertIn("await applyEdgeScale()", enhanced)
        self.assertIn("{'marker.size': [scaled]}", enhanced)
        self.assertIn("{'marker.size': original * scale}", enhanced)
        self.assertIn("{'line.width': original * scale}", enhanced)
        self.assertNotIn("quadruplePredictionEdgeWidths", enhanced)
        self.assertNotIn("quadruplePredictionNodeSizes", enhanced)
        self.assertIn("content === 'ground-truth-only'", enhanced)
        self.assertIn("await configureGroundTruthOnly()", enhanced)
        self.assertIn("'line.color': '#33cc33'", enhanced)
        self.assertIn("'marker.color': '#33cc33'", enhanced)
        self.assertNotIn("predictionEdgeWidth", enhanced)
        self.assertIn("const groundTruthTraces = gtOnly ? []", enhanced)
        self.assertIn("await placeOpaqueNodesAboveEdges()", enhanced)
        self.assertIn("{'marker.opacity': 1}", enhanced)
        self.assertIn("Plotly.moveTraces(gd, indices)", enhanced)
        self.assertIn("{'scene.camera': defaultCamera}", enhanced)
        self.assertIn("trace.name?.startsWith('Ground truth ')", enhanced)
        self.assertEqual(enhance_html(enhanced), enhanced)

    def test_file_update_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.html"
            path.write_text("<html><head></head><body></body></html>")
            self.assertTrue(enhance_file(path))
            self.assertFalse(enhance_file(path))

    def test_refreshes_existing_controls(self):
        original = '<html><head></head><body><div class="plotly-graph-div"></div></body></html>'
        enhanced = enhance_html(original)
        stale = enhanced.replace("clean['scene.domain.x'] = [0, 1];", "// stale script")
        stale = stale.replace("Node size", "Stale node control", 1)
        stale = stale.replace(".gnbm-range-control{", ".stale-range-control{")
        refreshed = enhance_html(stale)
        self.assertIn("clean['scene.domain.x'] = [0, 1]", refreshed)
        self.assertIn("Node size", refreshed)
        self.assertIn(".gnbm-range-control{", refreshed)
        self.assertNotIn("// stale script", refreshed)
        self.assertNotIn("Stale node control", refreshed)
        self.assertNotIn(".stale-range-control{", refreshed)


if __name__ == "__main__":
    unittest.main()
