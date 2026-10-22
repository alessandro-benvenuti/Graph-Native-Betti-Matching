"""Independent smoke inspection and measured CPU budget projection."""
import json
from pathlib import Path
import time
import statistics

import torch
from .core import ARMS, metrics, mode_view, optimize
from .cache import load_torch, json_write, new_directory


def verify(cache, run_dir):
    run_dir=Path(run_dir)
    marker=run_dir/('SYNTHETIC_VERIFIED.json' if cache['provenance'].get('synthetic') else 'SMOKE_VERIFIED.json')
    if marker.exists(): raise FileExistsError(marker)
    if not (run_dir/'COMPLETE.json').is_file(): raise ValueError('run did not complete')
    manifest=json.loads((run_dir/'COMPLETE.json').read_text())
    if manifest['samples'] != 2 or manifest['steps'] < 20: raise ValueError('smoke must use two patches and >=20 steps')
    results=[]
    for r in cache['records']:
        for mode in ('A','B'):
            expected=cache['step_zero'][r['source_sample_id']][mode]
            actual=metrics(mode_view(r,mode),r['node_logits'],mode_view(r,mode)['relation_logits'],mode,cache['config'])
            if expected != actual: raise ValueError('cache step-zero metrics do not reproduce')
            for arm in ARMS:
                state=load_torch(run_dir/f"{r['source_sample_id']}-{mode}-{arm}.pt")
                observed=dict(state['history'][0]['metrics']); observed.pop('spatial_cycle_audit',None)
                if observed != expected: raise ValueError('arm step-zero metrics differ from cache')
                if not state['complete']: raise ValueError('incomplete arm')
                results.append(dict(sample_id=r['source_sample_id'],mode=mode,arm=arm,
                    initial=expected,final=state['history'][-1]['metrics'],seconds=state['elapsed']))
    for filename in ('summary.json','per-patch.csv','gradients.json','trajectories-A.png','trajectories-B.png','report.md'):
        if not (run_dir/filename).is_file(): raise ValueError('missing artifact: '+filename)
    result=dict(cache_content=cache['content_sha256'],synthetic=cache['provenance'].get('synthetic',False),
        code=json.loads((run_dir/'run.json').read_text())['code'],checks='all arm starts match cache; all endpoints finite and complete; artifacts present',
        results=results,note='This checks execution fidelity. Read report.md before main submission.')
    json_write(marker,result)
    return result


def benchmark(cache, output, *, steps=5, max_seconds=300., threads=1):
    torch.set_num_threads(threads)
    output=new_directory(output)
    rows=[]; begun=time.perf_counter()
    # First, median-size and largest cached graphs, never an unbounded scan.
    ordered=sorted(cache['records'],key=lambda r:len(r['pairs']))
    indices=sorted({0,len(ordered)//2,len(ordered)-1})
    for index in indices:
        r=ordered[index]
        state=optimize(r,cache['config'],'B','h0_h1',steps=steps,spatial=False,
            max_seconds=max_seconds-(time.perf_counter()-begun))
        rows.append(dict(sample_id=r['source_sample_id'],nodes=len(r['selected']),pairs=len(r['pairs']),
            steps=state['step'],seconds=state['elapsed'],complete=state['complete']))
        if not state['complete']: break
    finished=all(r['complete'] for r in rows)
    # Six arms/modes, 32 patches, 200 updates. Conservative max observed / update,
    # plus 50% overhead. Graph cost varies; this is a projection, not a guarantee.
    seconds_per_step=max((r['seconds']/max(1,r['steps']) for r in rows),default=0.)
    result=dict(rows=rows,complete=finished,threads=threads,
        projected_32_patch_200_step_seconds=seconds_per_step*32*6*200*1.5 if finished else None,
        caveat='CPU projection from at most three cached graphs. GF2 graph topology cost varies with support; includes 50% headroom.',
        execution_seconds=time.perf_counter()-begun)
    json_write(output/'benchmark.json',result)
    return result



def next_commands(root):
    from .cache import code_provenance
    root=Path(root)
    marker=root/'smoke-opt/SMOKE_VERIFIED.json'
    if not marker.is_file(): raise ValueError('Real two-patch cluster smoke is not verified yet')
    smoke=json.loads(marker.read_text())
    if smoke.get('synthetic'): raise ValueError('Synthetic smoke cannot authorize a real diagnostic')
    current=code_provenance()
    if current['dirty'] or smoke['code']['commit'] != current['commit']:
        raise ValueError('Code changed since cluster smoke; repeat smoke in new output directories')
    benchmark_path=root/'cpu-benchmark/benchmark.json'
    if not (root/'main-cache/CACHE_COMPLETE.json').is_file():
        return ('Inspect smoke-opt/report.md, then submit the bounded 64-patch pool / 32-patch cache:\n'
            'CACHED_LOGIT_STAGE=main sbatch --export=ALL --output="'+str(root)+'/logs/%x-%j.out" experiments/cached_logits/cluster/cache.slurm\n'
            'After that job completes, run next-commands again.')
    if not benchmark_path.is_file():
        return ('Main cache is ready. Inspect main-cache/selection.json for category availability and shortfall.\n'
            'CACHED_LOGIT_CPU_STAGE=benchmark sbatch --export=ALL --output="'+str(root)+'/logs/%x-%j.out" experiments/cached_logits/cluster/optimize.slurm\n'
            'After the CPU benchmark completes, run next-commands again.')
    benchmark=json.loads(benchmark_path.read_text())
    estimate=benchmark.get('projected_32_patch_200_step_seconds')
    if not benchmark['complete'] or estimate is None or estimate>3300:
        return 'CPU benchmark exceeds the bounded main request or is incomplete. Inspect benchmark.json; main is not authorized automatically. No resource/sample/step increases were made.'
    return (f'Measured CPU projection: {estimate:.1f} seconds including 50% headroom.\n'
        'CACHED_LOGIT_CPU_STAGE=main sbatch --export=ALL --output="'+str(root)+'/logs/%x-%j.out" experiments/cached_logits/cluster/optimize.slurm\n'
        'For an interrupted, stopped job use CACHED_LOGIT_CPU_STAGE=resume instead; inspect any stale RUNNING.lock first.')
