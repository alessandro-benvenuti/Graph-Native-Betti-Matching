"""CLI: select, cache, optimize, report. No model imports in optimization."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from .cache import (DATASET, SOURCE_RUN, select_manifest, cache_predictions,
    load_torch, torch_write, json_write, new_directory, code_provenance, OUTPUT_ROOT)
from .core import ARMS, STEPS, fingerprint, optimize, sha256
from .report import report


def load_cache(path):
    marker = Path(path).parent/'CACHE_COMPLETE.json'
    if not marker.is_file(): raise ValueError('cache is not complete')
    expected = json.loads(marker.read_text())['sha256']
    if sha256(path) != expected: raise ValueError('cache file changed')
    cache = load_torch(path)
    digest = cache.pop('content_sha256')
    if fingerprint(cache) != digest: raise ValueError('cache content changed')
    cache['content_sha256'] = digest
    if cache['provenance']['split'] != 'train': raise ValueError('not a training cache')
    return cache


def run(args):
    cache = load_cache(args.cache)
    if not (1 <= args.threads <= 8 and 0 < args.steps <= 200 and 0 < args.max_seconds <= 14400 and args.lr > 0):
        raise ValueError('bounds: threads 1..8, steps 1..200, runtime (0,14400], lr>0')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    if (output/'COMPLETE.json').exists(): raise FileExistsError('completed outputs cannot be overwritten')
    settings = dict(cache_sha256=sha256(args.cache),steps=args.steps,lr=args.lr,threads=args.threads,
                    code=code_provenance(),optimizer='Adam',arms=ARMS,
                    h1_false_positive_weight=.25,h1_false_negative_weight=1.,
                    observation_steps=list(STEPS),constant_topology_weights=True)
    settings=json.loads(json.dumps(settings))
    if args.resume:
        old = json.loads((output/'run.json').read_text())
        if old != settings: raise ValueError('resume code/settings/cache mismatch')
    else:
        new_directory(output); json_write(output/'run.json',settings)
    lock = output/'RUNNING.lock'
    # A killed job can leave a lock. Operator must verify it stopped before removing.
    with lock.open('x') as handle: handle.write(str(os.getpid()))
    begun = time.perf_counter()
    entries = []
    try:
        for record in cache['records']:
            for mode in ('A','B'):
                for arm in ARMS:
                    path = output/f"{record['source_sample_id']}-{mode}-{arm}.pt"
                    state = load_torch(path) if path.exists() else None
                    if not state or not state['complete']:
                        remaining = args.max_seconds-(time.perf_counter()-begun)
                        if remaining <= 0:
                            json_write(output/'INCOMPLETE.json',dict(reason='runtime bound reached'))
                            return 2
                        state = optimize(record,cache['config'],mode,arm,steps=args.steps,lr=args.lr,state=state,
                            max_seconds=remaining,save=lambda value,p=path:torch_write(p,value))
                        if not state['complete']:
                            json_write(output/'INCOMPLETE.json',dict(reason='runtime bound reached',state=str(path)))
                            return 2
                    entries.append(dict(record=record,state=state))
        report(output,cache,entries)
        json_write(output/'COMPLETE.json',dict(cache_sha256=settings['cache_sha256'],
            samples=len(cache['records']),steps=args.steps,seconds=time.perf_counter()-begun))
        (output/'INCOMPLETE.json').unlink(missing_ok=True)
        print(json.dumps(dict(output=str(output),seconds=time.perf_counter()-begun,complete=True)))
        return 0
    finally:
        lock.unlink(missing_ok=True)


def main():
    os.environ['WANDB_MODE']='disabled'
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    synthetic=sub.add_parser('synthetic-cache')
    synthetic.add_argument('--output',type=Path,required=True)
    select=sub.add_parser('select')
    select.add_argument('--dataset',type=Path,default=DATASET)
    select.add_argument('--output',type=Path,required=True)
    select.add_argument('--count',type=int,default=64)
    select.add_argument('--seed',type=int,default=364505)
    select.add_argument('--pool',type=Path)
    cache=sub.add_parser('cache')
    cache.add_argument('--config',type=Path,default=SOURCE_RUN/'resolved-config.yaml')
    cache.add_argument('--checkpoint',type=Path,default=SOURCE_RUN/'models/latest_checkpoint.pt')
    cache.add_argument('--dataset',type=Path,default=DATASET)
    cache.add_argument('--dataset-name',default='synthetic_mri')
    cache.add_argument('--manifest',type=Path,required=True)
    cache.add_argument('--output',type=Path,required=True)
    cache.add_argument('--select-count',type=int)
    cache.add_argument('--workers',type=int,default=4)
    cache.add_argument('--device',default='cuda')
    cache.add_argument('--spatial-max-node-distance',type=float,default=.1)
    opt=sub.add_parser('optimize')
    opt.add_argument('--cache',type=Path,required=True)
    opt.add_argument('--output',type=Path,required=True)
    opt.add_argument('--steps',type=int,default=200)
    opt.add_argument('--lr',type=float,default=.03)
    opt.add_argument('--threads',type=int,default=1)
    opt.add_argument('--max-seconds',type=float,default=3600)
    opt.add_argument('--resume',action='store_true')
    next_parser=sub.add_parser('next-commands')
    next_parser.add_argument('--root',type=Path,default=OUTPUT_ROOT)
    verify=sub.add_parser('verify-smoke')
    verify.add_argument('--cache',type=Path,required=True)
    verify.add_argument('--run',type=Path,required=True)
    bench=sub.add_parser('benchmark')
    bench.add_argument('--cache',type=Path,required=True)
    bench.add_argument('--output',type=Path,required=True)
    bench.add_argument('--steps',type=int,default=5)
    bench.add_argument('--threads',type=int,default=1)
    bench.add_argument('--max-seconds',type=float,default=300.)
    rep=sub.add_parser('report')
    rep.add_argument('--cache',type=Path,required=True)
    rep.add_argument('--run',type=Path,required=True)
    rep.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='synthetic-cache':
        from .synthetic import create_cache
        print(create_cache(args.output))
    elif args.command=='select':
        if not 1 <= args.count <= 128: raise ValueError('count must be 1..128')
        select_manifest(args)
    elif args.command=='cache': cache_predictions(args)
    elif args.command=='optimize': return run(args)
    elif args.command=='next-commands':
        from .verify import next_commands
        print(next_commands(args.root))
    elif args.command=='verify-smoke':
        from .verify import verify
        print(json.dumps(verify(load_cache(args.cache),args.run),indent=2))
    elif args.command=='benchmark':
        from .verify import benchmark
        if not 1 <= args.steps <= 20 or not 1 <= args.threads <= 8 or not 0 < args.max_seconds <= 600:
            raise ValueError('benchmark bounds: 1..20 steps, 1..8 threads, <=600 seconds')
        print(json.dumps(benchmark(load_cache(args.cache),args.output,steps=args.steps,threads=args.threads,max_seconds=args.max_seconds),indent=2))
    else:
        cache=load_cache(args.cache)
        if not (args.run/'COMPLETE.json').is_file(): raise ValueError('run is incomplete')
        output=new_directory(args.output)
        entries=[]
        for record in cache['records']:
            for mode in ('A','B'):
                for arm in ARMS:
                    state=load_torch(args.run/f"{record['source_sample_id']}-{mode}-{arm}.pt")
                    if not state['complete']: raise ValueError('incomplete arm')
                    entries.append(dict(record=record,state=state))
        report(output,cache,entries)
    return 0

if __name__=='__main__': sys.exit(main())
