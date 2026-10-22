"""Login provenance and real job-script startup without compute-node Git."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.cached_logits import provenance

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / 'experiments/cached_logits/cluster'


class CachedLogitClusterEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'work/projects/Graph-Native-Betti-Matching-cached-logit-test'
        self.repo.mkdir(parents=True)
        self.git('init')
        self.git('config', 'user.email', 'synthetic@example.invalid')
        self.git('config', 'user.name', 'Synthetic fixture')
        self.git('checkout', '-b', provenance.BRANCH)
        self.git('remote', 'add', 'origin', provenance.REMOTE)
        (self.repo / '.git/cached-logit-disposable').touch()
        (self.repo / '.gitignore').write_text('artifacts/\n__pycache__/\n')
        self.write('experiments/sample.py', 'value = 1\n')
        self.commit()

    def git(self, *arguments):
        return subprocess.run(['git', '-C', str(self.repo), *arguments],
                              check=True, capture_output=True, text=True).stdout.strip()

    def write(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-m', 'Synthetic fixture')

    def test_snapshot_reuses_identical_capture_without_overwriting(self):
        first = provenance.capture(self.repo)
        contents = first.read_bytes()
        self.assertEqual(first, provenance.capture(self.repo))
        self.assertEqual(first.read_bytes(), contents)
        result = provenance.verify(first, self.repo)
        self.assertEqual(result['commit'], self.git('rev-parse', 'HEAD'))
        self.assertEqual(result['dirty'], '')

    def test_batch_code_provenance_never_invokes_git(self):
        snapshot = provenance.capture(self.repo)
        with patch.dict(os.environ, {'GNBM_CODE_PROVENANCE_FILE':str(snapshot), 'SLURM_JOB_ID':'123', 'PATH':''}), \
             patch.object(provenance, 'ROOT', self.repo), \
             patch.object(provenance.subprocess, 'run', side_effect=AssertionError('Batch launched a subprocess')):
            result = provenance.code_provenance()
        self.assertEqual(result['verification'], 'login_snapshot_sha256')
        self.assertEqual(result['branch'], provenance.BRANCH)

    def test_batch_without_snapshot_does_not_fall_back_to_git(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID':'123'}), \
             patch.object(provenance.subprocess, 'run', side_effect=AssertionError('Batch invoked Git')):
            with patch.dict(os.environ):
                os.environ.pop('GNBM_CODE_PROVENANCE_FILE', None)
                with self.assertRaisesRegex(ValueError, 'Missing login-node provenance'):
                    provenance.code_provenance()

    def test_changed_tracked_file_is_rejected(self):
        snapshot = provenance.capture(self.repo)
        self.write('experiments/sample.py', 'value = 2\n')
        with self.assertRaisesRegex(ValueError, 'Code changed.*experiments/sample.py'):
            provenance.verify(snapshot, self.repo)

    def test_new_untracked_python_source_is_rejected(self):
        snapshot = provenance.capture(self.repo)
        self.write('experiments/untracked.py', 'value = 2\n')
        with self.assertRaisesRegex(ValueError, 'Python source files changed'):
            provenance.verify(snapshot, self.repo)

    def test_modified_snapshot_and_other_checkout_are_rejected(self):
        snapshot = provenance.capture(self.repo)
        with self.assertRaisesRegex(ValueError, 'another checkout'):
            provenance.verify(snapshot, self.root)
        value = json.loads(snapshot.read_text())
        value['commit'] = '0'*40
        snapshot.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'content hash changed'):
            provenance.verify(snapshot, self.repo)

    def test_login_refuses_dirty_or_wrong_branch(self):
        self.write('experiments/sample.py', 'value = 2\n')
        with self.assertRaisesRegex(ValueError, 'Commit changes'):
            provenance.capture(self.repo)
        self.commit()
        self.git('checkout', '-b', 'unrelated')
        with self.assertRaisesRegex(ValueError, 'Wrong disposable branch'):
            provenance.capture(self.repo)

    def test_git_status_failure_includes_original_error_text(self):
        result = subprocess.CompletedProcess(['git'], 128, '', 'fixture status failure')
        with patch.object(provenance.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'exit 128.*fixture status failure'):
                provenance._git(self.repo, 'status', '--porcelain')

    def test_real_gpu_and_cpu_job_startup_without_git(self):
        # The real shell jobs and real snapshot verification run. Expensive
        # model/data Python commands are stubs; this tests startup, not inference.
        for name in ('guard.sh', 'cache.slurm', 'optimize.slurm'):
            text = (CLUSTER / name).read_text().replace(
                '/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100',
                str(self.repo/'artifacts'))
            self.write('experiments/cached_logits/cluster/'+name, text)
        for helper in ('provenance.py','retry.py','runtime.py'):
            self.write('experiments/cached_logits/'+helper,
                       (ROOT/'experiments/cached_logits'/helper).read_text())
        self.write('cluster/jean_zay/env_a100.sh',
                   'export PATH="$TEST_BIN"\ncd "$GNBM_REPO_DIR"\n')
        self.commit()
        snapshot = provenance.capture(self.repo)
        bin_dir = self.root/'bin'
        bin_dir.mkdir()
        python = bin_dir/'python'
        python.write_text('''#!/bin/bash
if [[ "$1" == */provenance.py || "$1" == */retry.py || "$1" == */runtime.py ]]; then exec "$TEST_PYTHON" "$@"; fi
printf '%s\\n' "$*" >> "$TEST_PYTHON_LOG"
''')
        python.chmod(0o755)
        venv = self.root/'venv'
        (venv/'bin').mkdir(parents=True)
        (venv/'bin/activate').write_text('export PATH="$TEST_BIN"\n')
        marker = self.repo/'artifacts/smoke-opt/SMOKE_VERIFIED.json'
        env = dict(os.environ, WORK=str(self.root/'work'), PATH=str(bin_dir),
            GNBM_CODE_PROVENANCE_FILE=str(snapshot), GNBM_A100_VENV=str(venv),
            TEST_BIN=str(bin_dir), TEST_PYTHON=sys.executable,
            TEST_PYTHON_LOG=str(self.root/'python.log'), SLURM_JOB_ID='123')
        cases=(('cache.slurm',None,3300),('optimize.slurm','benchmark',3300),
               ('optimize.slurm','main',3300),('optimize.slurm','main',27000),
               ('optimize.slurm','resume',27000))
        for name,stage,budget in cases:
            if name=='optimize.slurm':
                marker.parent.mkdir(parents=True,exist_ok=True)
                marker.write_text('{}')
                benchmark=self.repo/'artifacts/cpu-benchmark/benchmark.json'
                benchmark.parent.mkdir(parents=True,exist_ok=True)
                benchmark.write_text(json.dumps(dict(complete=True,
                    projected_32_patch_200_step_seconds=24955 if budget==27000 else 100)))
                env['CACHED_LOGIT_CPU_STAGE']=stage
                env['CACHED_LOGIT_CPU_BUDGET_SECONDS']=str(budget)
            env['TEST_JOB'] = str(self.repo/'experiments/cached_logits/cluster'/name)
            result = subprocess.run(['/bin/bash', '-c', 'module() { :; }; source "$TEST_JOB"'],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            self.assertIn('login_snapshot_sha256', result.stdout)
            self.assertNotIn('command not found', result.stderr)
        log = (self.root/'python.log').read_text()
        self.assertIn('experiments.cached_logits cache', log)
        self.assertIn('experiments.cached_logits verify-smoke', log)
        self.assertIn('experiments.cached_logits benchmark', log)
        self.assertIn('--steps 200 --threads 1 --max-seconds 27000 --resume',log)

    def test_actual_submit_requests_long_walltime_only_for_named_long_stage(self):
        for name in ('submit.sh','git_env.sh'):
            text=(CLUSTER/name).read_text().replace(
                '/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100',
                str(self.repo/'artifacts'))
            self.write('experiments/cached_logits/cluster/'+name,text)
        self.write('experiments/cached_logits/provenance.py',
                   (ROOT/'experiments/cached_logits/provenance.py').read_text())
        self.commit()
        bin_dir=self.root/'bin';bin_dir.mkdir()
        (bin_dir/'python3').symlink_to(sys.executable)
        mock=bin_dir/'sbatch'
        mock.write_text("""#!/bin/bash
exec "$TEST_PYTHON" -c 'import json,os,sys;print(json.dumps(dict(args=sys.argv[1:],stage=os.environ.get("CACHED_LOGIT_CPU_STAGE"),budget=os.environ.get("CACHED_LOGIT_CPU_BUDGET_SECONDS"),snapshot=os.environ.get("GNBM_CODE_PROVENANCE_FILE"))))' "$@"
""")
        mock.chmod(0o755)
        environment=dict(os.environ,WORK=str(self.root/'work'),TEST_PYTHON=sys.executable,
                         PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
        for stage in ('main','main-long','resume-long'):
            result=subprocess.run(['/bin/bash',str(self.repo/'experiments/cached_logits/cluster/submit.sh'),stage],
                env=environment,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            record=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(record['stage'],'resume' if stage=='resume-long' else 'main')
            self.assertEqual(record['budget'],'3300' if stage=='main' else '27000')
            self.assertEqual('--time=08:00:00' in record['args'],stage!='main')
            self.assertTrue(Path(record['snapshot']).is_file())
            self.assertFalse(any(arg.startswith('--gres') for arg in record['args']))
            if stage!='main':
                preflight=json.loads(result.stdout.splitlines()[-2])
                self.assertIn('--test-only',preflight['args'])

    def test_submit_captures_snapshot_before_sbatch_and_jobs_do_not_use_git(self):
        submit = (CLUSTER/'submit.sh').read_text()
        self.assertLess(submit.index('provenance.py" capture'), submit.index('\nsbatch \"${slurm_options[@]}\"'))
        for name in ('guard.sh', 'cache.slurm', 'optimize.slurm'):
            text = (CLUSTER/name).read_text()
            self.assertNotIn('cached_logit_require_git', text)
            self.assertNotIn('git -C', text)
        self.assertIn('GNBM_CODE_PROVENANCE_FILE', submit)


if __name__ == '__main__':
    unittest.main()
