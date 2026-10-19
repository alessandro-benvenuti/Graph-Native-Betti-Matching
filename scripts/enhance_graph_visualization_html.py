#!/usr/bin/env python3
"""Add paper-figure controls to portable Plotly graph visualizations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "gnbm-paper-controls"

STYLE = r"""
<style id="gnbm-paper-controls-style">
#gnbm-paper-controls{position:sticky;top:0;z-index:1000;display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;padding:.65rem;margin:0 0 1rem;background:rgba(255,255,255,.96);border:1px solid #cfcfcf;border-radius:.4rem;box-shadow:0 1px 5px rgba(0,0,0,.12)}
#gnbm-paper-controls button{font:inherit;padding:.4rem .7rem;border:1px solid #888;border-radius:.3rem;background:#fff;color:#222}
#gnbm-paper-controls button:hover{background:#f0f0f0}
#gnbm-paper-controls button:disabled{opacity:.55}
#gnbm-camera-json{flex:1 1 28rem;min-width:16rem;height:2.25rem;padding:.35rem .5rem;border:1px solid #999;border-radius:.3rem;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
#gnbm-control-status{flex-basis:100%;min-height:1.2em;color:#444;font-size:.9rem}
@media (prefers-color-scheme:dark){#gnbm-paper-controls{background:rgba(30,30,30,.96);border-color:#666;color:#eee}#gnbm-paper-controls button{background:#292929;color:#eee;border-color:#888}#gnbm-paper-controls button:hover{background:#3a3a3a}#gnbm-camera-json{background:#202020;color:#eee;border-color:#777}#gnbm-control-status{color:#ccc}}
</style>
"""

CONTROLS = r"""
<div id="gnbm-paper-controls" role="toolbar" aria-label="Paper figure controls">
  <button id="gnbm-toggle-mri" type="button">Hide MRI image</button>
  <button id="gnbm-copy-camera" type="button">Copy camera</button>
  <button id="gnbm-paste-camera" type="button">Paste camera</button>
  <button id="gnbm-apply-camera" type="button">Apply camera JSON</button>
  <button id="gnbm-screenshot" type="button">Save clean PNG</button>
  <textarea id="gnbm-camera-json" aria-label="Camera JSON" placeholder="Camera coordinates appear here; paste the same JSON into another report."></textarea>
  <span id="gnbm-control-status" aria-live="polite"></span>
</div>
"""

SCRIPT = r"""
<script id="gnbm-paper-controls-script">
(async () => {
  const gd = document.querySelector('.plotly-graph-div');
  if (!gd || !window.Plotly) return;
  const status = document.getElementById('gnbm-control-status');
  const cameraField = document.getElementById('gnbm-camera-json');
  const toggleMri = document.getElementById('gnbm-toggle-mri');
  const copyCamera = document.getElementById('gnbm-copy-camera');
  const pasteCamera = document.getElementById('gnbm-paste-camera');
  const applyCamera = document.getElementById('gnbm-apply-camera');
  const screenshot = document.getElementById('gnbm-screenshot');
  const mriIndices = () => gd.data
    .map((trace, index) => trace.name === 'MRI slices' ? index : -1)
    .filter(index => index >= 0);
  const setStatus = message => { status.textContent = message; };
  const clone = value => JSON.parse(JSON.stringify(value || {}));
  const currentCamera = () => clone(
    gd._fullLayout?.scene?.camera || gd.layout?.scene?.camera || {}
  );
  const defaultCamera = {
    up: {x: 0, y: 0, z: 1},
    center: {x: 0, y: 0, z: 0},
    eye: {
      x: -1.7506160977336473,
      y: -0.7684048551754746,
      z: 1.0061854073816852
    },
    projection: {type: 'perspective'}
  };

  const predictionGraph = () => {
    const vertexIds = new Map();
    const vertices = [];
    const edges = [];
    const vertex = point => {
      const key = point.map(value => Number(value).toPrecision(12)).join('|');
      if (!vertexIds.has(key)) {
        vertexIds.set(key, vertices.length);
        vertices.push(point);
      }
      return vertexIds.get(key);
    };
    gd.data
      .filter(trace => trace.name === 'Prediction edges' && trace.mode === 'lines')
      .forEach(trace => {
        for (let index = 0; index + 1 < trace.x.length; index += 3) {
          const left = [trace.x[index], trace.y[index], trace.z[index]];
          const right = [trace.x[index + 1], trace.y[index + 1], trace.z[index + 1]];
          edges.push({
            left: vertex(left), right: vertex(right),
            leftPoint: left, rightPoint: right,
            width: Number(trace.line?.width ?? 3)
          });
        }
      });
    return {vertexCount: vertices.length, edges};
  };

  const cycleEdgeIndices = ({vertexCount, edges}) => {
    const adjacency = Array.from({length: vertexCount}, () => []);
    edges.forEach((edge, index) => {
      adjacency[edge.left].push([edge.right, index]);
      adjacency[edge.right].push([edge.left, index]);
    });
    const discovery = Array(vertexCount).fill(-1);
    const low = Array(vertexCount).fill(-1);
    const bridges = new Set();
    let time = 0;
    const visit = (node, parentEdge) => {
      discovery[node] = low[node] = time++;
      for (const [neighbor, edgeIndex] of adjacency[node]) {
        if (edgeIndex === parentEdge) continue;
        if (discovery[neighbor] < 0) {
          visit(neighbor, edgeIndex);
          low[node] = Math.min(low[node], low[neighbor]);
          if (low[neighbor] > discovery[node]) bridges.add(edgeIndex);
        } else {
          low[node] = Math.min(low[node], discovery[neighbor]);
        }
      }
    };
    for (let node = 0; node < vertexCount; node += 1) {
      if (discovery[node] < 0) visit(node, -1);
    }
    return edges.map((_, index) => index).filter(index => !bridges.has(index));
  };

  const addPredictionLoopOverlay = async () => {
    const graph = predictionGraph();
    const loopIndices = cycleEdgeIndices(graph);
    if (!loopIndices.length) return;
    const groups = new Map();
    loopIndices.forEach(index => {
      const edge = graph.edges[index];
      if (!groups.has(edge.width)) {
        groups.set(edge.width, {width: edge.width, x: [], y: [], z: []});
      }
      const group = groups.get(edge.width);
      group.x.push(edge.leftPoint[0], edge.rightPoint[0], null);
      group.y.push(edge.leftPoint[1], edge.rightPoint[1], null);
      group.z.push(edge.leftPoint[2], edge.rightPoint[2], null);
    });
    const traces = Array.from(groups.values()).map((group, index) => ({
      type: 'scatter3d', mode: 'lines',
      x: group.x, y: group.y, z: group.z,
      line: {color: '#ff3300', width: group.width},
      name: 'Predicted loop edges', legendgroup: 'Prediction',
      showlegend: index === 0,
      hovertemplate: 'Predicted edge belonging to a cycle<extra></extra>',
      visible: true
    }));
    await Plotly.addTraces(gd, traces);
  };

  const doublePredictionEdgeWidths = async () => {
    const indices = gd.data
      .map((trace, index) => trace.name === 'Prediction edges' ? index : -1)
      .filter(index => index >= 0);
    for (const index of indices) {
      const width = Number(gd.data[index].line?.width ?? 3);
      await Plotly.restyle(gd, {'line.width': width * 2}, [index]);
    }
  };

  await doublePredictionEdgeWidths();
  await addPredictionLoopOverlay();
  await Plotly.relayout(gd, {'scene.camera': defaultCamera});
  cameraField.value = JSON.stringify(defaultCamera);

  const updateMriLabel = () => {
    const shown = mriIndices().some(index => {
      const visible = gd.data[index].visible;
      return visible !== false && visible !== 'legendonly';
    });
    toggleMri.textContent = shown ? 'Hide MRI image' : 'Show MRI image';
    toggleMri.setAttribute('aria-pressed', String(shown));
  };
  updateMriLabel();

  toggleMri.addEventListener('click', async () => {
    const indices = mriIndices();
    const currentlyShown = indices.some(index => {
      const visible = gd.data[index].visible;
      return visible !== false && visible !== 'legendonly';
    });
    await Plotly.restyle(gd, {visible: !currentlyShown}, indices);
    updateMriLabel();
    setStatus(currentlyShown ? 'MRI image hidden.' : 'MRI image shown.');
  });

  const copyText = async text => {
    if (navigator.clipboard?.writeText) {
      try { await navigator.clipboard.writeText(text); return true; } catch (_) {}
    }
    cameraField.focus();
    cameraField.select();
    return document.execCommand('copy');
  };

  copyCamera.addEventListener('click', async () => {
    const text = JSON.stringify(currentCamera());
    cameraField.value = text;
    const copied = await copyText(text);
    setStatus(copied ? 'Camera JSON copied.' : 'Camera JSON selected; copy it manually.');
  });

  const applyCameraText = async text => {
    let camera;
    try { camera = JSON.parse(text); }
    catch (error) { setStatus(`Invalid camera JSON: ${error.message}`); return false; }
    await Plotly.relayout(gd, {'scene.camera': camera});
    cameraField.value = JSON.stringify(camera);
    setStatus('Camera applied.');
    return true;
  };

  applyCamera.addEventListener('click', () => applyCameraText(cameraField.value));
  pasteCamera.addEventListener('click', async () => {
    let text = '';
    if (navigator.clipboard?.readText) {
      try { text = await navigator.clipboard.readText(); } catch (_) {}
    }
    if (!text) text = window.prompt('Paste camera JSON:') || '';
    if (text) await applyCameraText(text);
  });

  gd.on('plotly_relayout', event => {
    if (event['scene.camera']) cameraField.value = JSON.stringify(event['scene.camera']);
  });

  screenshot.addEventListener('click', async () => {
    screenshot.disabled = true;
    setStatus('Preparing clean high-resolution PNG…');
    const scene = gd.layout.scene || {};
    const axes = ['xaxis', 'yaxis', 'zaxis'];
    const restore = {};
    const clean = {};
    restore['showlegend'] = gd.layout.showlegend ?? null;
    clean['showlegend'] = false;
    for (const side of ['l', 'r', 'b', 't', 'pad']) {
      restore[`margin.${side}`] = gd.layout.margin?.[side] ?? null;
      clean[`margin.${side}`] = 0;
    }
    restore['scene.domain.x'] = scene.domain?.x ?? null;
    restore['scene.domain.y'] = scene.domain?.y ?? null;
    clean['scene.domain.x'] = [0, 1];
    clean['scene.domain.y'] = [0, 1];
    axes.forEach(axis => {
      const settings = scene[axis] || {};
      restore[`scene.${axis}.showticklabels`] = settings.showticklabels ?? null;
      restore[`scene.${axis}.title.text`] = settings.title?.text ?? null;
      clean[`scene.${axis}.showticklabels`] = false;
      clean[`scene.${axis}.title.text`] = '';
    });
    const confidenceTraces = gd.data
      .map((trace, index) => trace.marker?.colorbar?.title ? index : -1)
      .filter(index => index >= 0);
    const confidenceShowscale = confidenceTraces.map(
      index => gd.data[index].marker.showscale ?? null
    );
    const groundTruthTraces = gd.data
      .map((trace, index) => trace.name?.startsWith('Ground truth ') ? index : -1)
      .filter(index => index >= 0);
    const groundTruthVisibility = groundTruthTraces.map(
      index => gd.data[index].visible ?? null
    );
    try {
      if (confidenceTraces.length) {
        await Plotly.restyle(gd, {'marker.showscale': false}, confidenceTraces);
      }
      if (groundTruthTraces.length) {
        await Plotly.restyle(gd, {visible: false}, groundTruthTraces);
      }
      await Plotly.relayout(gd, clean);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const parts = location.pathname.split('/').filter(Boolean);
      const model = parts.length > 1 ? parts[parts.length - 2] : 'model';
      const source = document.querySelector('table td')?.textContent?.trim() || 'graph';
      await Plotly.downloadImage(gd, {
        format: 'png', width: 1800, height: 1400, scale: 2,
        filename: `${source}-${model}-clean`
      });
      setStatus('Clean PNG saved with predicted loops highlighted in red.');
    } catch (error) {
      setStatus(`Screenshot failed: ${error.message}`);
    } finally {
      await Plotly.relayout(gd, restore);
      if (confidenceTraces.length) {
        await Plotly.restyle(
          gd, {'marker.showscale': confidenceShowscale}, confidenceTraces
        );
      }
      if (groundTruthTraces.length) {
        await Plotly.restyle(
          gd, {visible: groundTruthVisibility}, groundTruthTraces
        );
      }
      screenshot.disabled = false;
    }
  });
})();
</script>
"""


def enhance_html(text: str) -> str:
    if f'id="{MARKER}"' in text:
        updated, replacements = re.subn(
            r'<script id="gnbm-paper-controls-script">.*?</script>',
            SCRIPT.strip(),
            text,
            count=1,
            flags=re.DOTALL,
        )
        if replacements != 1:
            raise ValueError("Existing paper controls have no replaceable script")
        return updated
    if "</head>" not in text or "<body>" not in text or "</body>" not in text:
        raise ValueError("Expected a complete HTML document")
    text = text.replace("</head>", STYLE + "</head>", 1)
    text = text.replace("<body>", "<body>" + CONTROLS, 1)
    return text.replace("</body>", SCRIPT + "</body>", 1)


def enhance_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = enhance_html(original)
    if updated == original:
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    files = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.html")))
        elif path.suffix.lower() == ".html":
            files.append(path)
        else:
            raise ValueError(f"Not an HTML file or directory: {path}")
    changed = sum(enhance_file(path) for path in files)
    print(f"Enhanced {changed} of {len(files)} HTML files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
