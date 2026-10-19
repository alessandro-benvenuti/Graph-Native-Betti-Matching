#!/usr/bin/env python3
"""Create ground-truth-only variants of focal-loss paper visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.enhance_graph_visualization_html import enhance_html


MODE_META = '<meta name="gnbm-visualization-mode" content="ground-truth-only">'


def ground_truth_variant(text: str) -> str:
    if MODE_META not in text:
        if "</head>" not in text:
            raise ValueError("Expected a complete HTML document")
        text = text.replace("</head>", MODE_META + "</head>", 1)
    return enhance_html(text)


def create_variants(root: Path) -> list[Path]:
    outputs = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        baseline_dir = sample_dir / "baseline"
        sources = sorted(baseline_dir.glob("*.html"))
        if len(sources) != 1:
            raise ValueError(
                f"Expected exactly one baseline HTML in {baseline_dir}, found {len(sources)}"
            )
        destination_dir = sample_dir / "GT"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / sources[0].name
        updated = ground_truth_variant(sources[0].read_text(encoding="utf-8"))
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(destination)
        outputs.append(destination)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing one directory per patch")
    args = parser.parse_args()
    outputs = create_variants(args.root)
    print(f"Created {len(outputs)} ground-truth-only HTML files")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
