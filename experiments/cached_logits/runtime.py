"""Explicit CPU budgets for the disposable diagnostic; no loss/model changes."""
import argparse
import json
import math
from pathlib import Path

SHORT_SECONDS = 3300
LONG_SECONDS = 27000
MAX_SECONDS = 28800


def check_benchmark(benchmark, budget):
    if budget not in (SHORT_SECONDS, LONG_SECONDS):
        raise ValueError('CPU job budget must be 3300 or 27000 seconds')
    estimate = benchmark.get('projected_32_patch_200_step_seconds')
    if not benchmark.get('complete') or not isinstance(estimate, (int,float)) or isinstance(estimate,bool):
        raise ValueError('CPU benchmark is incomplete or lacks a numeric projection')
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError('CPU projection must be finite and positive')
    if estimate > budget:
        raise ValueError(f'CPU projection {estimate:.1f}s exceeds the explicit {budget}s job budget')
    return dict(projected_seconds=estimate, budget_seconds=budget,
                threads=1, gpus=0, steps=200)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--budget',type=int,choices=(SHORT_SECONDS,LONG_SECONDS),required=True)
    args=parser.parse_args()
    print(json.dumps(check_benchmark(json.loads(args.benchmark.read_text()),args.budget),sort_keys=True))
