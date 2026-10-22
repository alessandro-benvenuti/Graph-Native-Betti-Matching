"""Only an explicit submission option may request the longer CPU budget."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.cached_logits.runtime import check_benchmark, LONG_SECONDS, SHORT_SECONDS
from experiments.cached_logits.synthetic import create_cache
from experiments.cached_logits.__main__ import run

ROOT=Path(__file__).resolve().parents[1]


class CachedLogitRuntimeTests(unittest.TestCase):
    def test_reported_projection_requires_explicit_long_budget(self):
        benchmark=dict(complete=True,projected_32_patch_200_step_seconds=24955.1393097639)
        with self.assertRaisesRegex(ValueError,'exceeds.*3300'):
            check_benchmark(benchmark,SHORT_SECONDS)
        result=check_benchmark(benchmark,LONG_SECONDS)
        self.assertEqual(result['budget_seconds'],27000)
        self.assertEqual(result['gpus'],0)
        self.assertEqual(result['steps'],200)

    def test_incomplete_invalid_and_over_budget_benchmarks_are_rejected(self):
        for value in (None,True,float('nan'),float('inf'),-1,0,27001):
            with self.assertRaises(ValueError):
                check_benchmark(dict(complete=True,projected_32_patch_200_step_seconds=value),LONG_SECONDS)
        with self.assertRaises(ValueError):
            check_benchmark(dict(complete=False,projected_32_patch_200_step_seconds=100),LONG_SECONDS)
        with self.assertRaises(ValueError):
            check_benchmark(dict(complete=True,projected_32_patch_200_step_seconds=100),999999)

    def test_long_budget_is_recorded_without_changing_objective_or_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cache=create_cache(root/'cache')/'predictions.pt'
            arguments=Namespace(cache=cache,output=root/'short',steps=1,lr=.03,threads=1,
                max_seconds=SHORT_SECONDS,resume=False)
            with patch('experiments.cached_logits.__main__.code_provenance',return_value=dict(commit='test',branch='test',dirty='')):
                self.assertEqual(run(arguments),0)
                arguments.output=root/'long';arguments.max_seconds=LONG_SECONDS
                self.assertEqual(run(arguments),0)
            left=json.loads((root/'short/per-patch.json').read_text())
            right=json.loads((root/'long/per-patch.json').read_text())
            for a,b in zip(left,right):
                for key in ('edge_f1','node_f1','beta0_error','beta1_error','shared_cycle_rank','false_cycle_rank','missed_cycle_rank'):
                    self.assertEqual(a[key],b[key])
            entry=json.loads((root/'long/invocations.jsonl').read_text())
            self.assertEqual(entry['max_seconds'],27000)
            self.assertFalse(entry['resume'])

    def test_only_named_long_submission_selects_eight_hour_allocation(self):
        submit=(ROOT/'experiments/cached_logits/cluster/submit.sh').read_text()
        self.assertIn('export CACHED_LOGIT_CPU_BUDGET_SECONDS=3300',submit)
        self.assertIn('main-long|resume-long)',submit)
        self.assertIn('export CACHED_LOGIT_CPU_BUDGET_SECONDS=27000',submit)
        self.assertIn('slurm_options+=(--time=08:00:00)',submit)
        batch=(ROOT/'experiments/cached_logits/cluster/optimize.slurm').read_text()
        self.assertIn('#SBATCH --time=01:00:00',batch)
        self.assertIn('#SBATCH --cpus-per-task=4',batch)
        self.assertNotIn('--gres',batch)
        self.assertIn('--budget "$cpu_budget"',batch)
        self.assertIn('--max-seconds "$cpu_budget"',batch)


if __name__=='__main__': unittest.main()
