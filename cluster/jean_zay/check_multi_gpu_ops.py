#!/usr/bin/env python3
"""Exercise the custom CUDA operator on every visible non-default GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from models.ops.modules import MSDeformAttn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.devices:
        raise RuntimeError(
            f"requested {args.devices} visible GPUs, found {torch.cuda.device_count()}"
        )
    original_device = torch.cuda.current_device()
    try:
        # Deliberately leave zero current. Correct extension code must guard
        # each input tensor's device before selecting a stream or launching.
        torch.cuda.set_device(0)
        for index in range(args.devices):
            device = torch.device(f"cuda:{index}")
            module = MSDeformAttn(48, 1, 6, 2, True).to(device).train()
            query = torch.randn((2, 3, 48), device=device, requires_grad=True)
            values = torch.randn((2, 8, 48), device=device, requires_grad=True)
            shapes = torch.tensor([[2, 2, 2]], dtype=torch.long, device=device)
            starts = torch.tensor([0], dtype=torch.long, device=device)
            references = torch.rand((2, 3, 1, 3), device=device)
            output = module(query, references, values, shapes, starts)
            output.square().mean().backward()
            torch.cuda.synchronize(device)
            if not torch.isfinite(query.grad).all() or not torch.isfinite(values.grad).all():
                raise FloatingPointError(f"non-finite operator gradient on {device}")
            print(f"deformable-attention forward/backward passed on {device}")
            del module, query, values, shapes, starts, references, output
    finally:
        torch.cuda.set_device(original_device)


if __name__ == "__main__":
    main()
