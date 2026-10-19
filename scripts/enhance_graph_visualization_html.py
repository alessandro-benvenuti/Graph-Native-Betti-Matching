#!/usr/bin/env python3
"""Add paper-figure controls to portable Plotly graph visualizations."""

from __future__ import annotations

import argparse
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
(() => {
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
    axes.forEach(axis => {
      const settings = scene[axis] || {};
      restore[`scene.${axis}.showticklabels`] = settings.showticklabels ?? null;
      restore[`scene.${axis}.title.text`] = settings.title?.text ?? null;
      clean[`scene.${axis}.showticklabels`] = false;
      clean[`scene.${axis}.title.text`] = '';
    });
    try {
      await Plotly.relayout(gd, clean);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const parts = location.pathname.split('/').filter(Boolean);
      const model = parts.length > 1 ? parts[parts.length - 2] : 'model';
      const source = document.querySelector('table td')?.textContent?.trim() || 'graph';
      await Plotly.downloadImage(gd, {
        format: 'png', width: 1800, height: 1400, scale: 2,
        filename: `${source}-${model}-clean`
      });
      setStatus('Clean PNG saved without axis numbers.');
    } catch (error) {
      setStatus(`Screenshot failed: ${error.message}`);
    } finally {
      await Plotly.relayout(gd, restore);
      screenshot.disabled = false;
    }
  });
})();
</script>
"""


def enhance_html(text: str) -> str:
    if f'id="{MARKER}"' in text:
        return text
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
