#!/usr/bin/env python3
"""Record venv-local packages without importing them or freezing cluster modules."""

import argparse
from importlib import metadata
from pathlib import Path
import re


def requirements_for(venv: Path) -> list[str]:
    locations = sorted(venv.glob("lib/python*/site-packages"))
    if not locations:
        raise RuntimeError(f"No site-packages found in {venv}")
    packages = {}
    for distribution in metadata.distributions(path=[str(path) for path in locations]):
        name = distribution.metadata.get("Name")
        if not name:
            raise RuntimeError(f"Distribution without a name in {venv}")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        # Keep Jean Zay's module-provided CUDA stack. The repository rebuilds
        # its custom deformable-attention operator for the selected architecture.
        if normalized in {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"} \
                or normalized.startswith("nvidia-") \
                or "multiscaledeformableattention" in normalized.replace("-", ""):
            continue
        packages[normalized] = f"{name}=={distribution.version}"
    return [packages[key] for key in sorted(packages)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = requirements_for(args.venv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Venv-local package versions; excludes the cluster CUDA/PyTorch stack.\n"
        + "\n".join(rows) + "\n", encoding="utf-8",
    )
    print(f"Recorded {len(rows)} venv-local package versions: {args.output}")


if __name__ == "__main__":
    main()
