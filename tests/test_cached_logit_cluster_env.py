"""Batch environment regressions using an initially Git-free PATH."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / 'experiments/cached_logits/cluster/guard.sh'


class CachedLogitClusterEnvironmentTests(unittest.TestCase):
    def run_guard(self, *, fail_load=False, fail_status=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / 'work/projects/Graph-Native-Betti-Matching-cached-logit-test'
            (repo / '.git').mkdir(parents=True)
            (repo / '.git/cached-logit-disposable').touch()
            empty = root / 'empty'
            empty.mkdir()
            git_dir = root / 'git/bin'
            git_dir.mkdir(parents=True)
            executable = git_dir / 'git'
            executable.write_text('''#!/bin/bash
if [[ "$1" == --version ]]; then echo 'git test fixture'; exit 0; fi
if [[ "$1" == -C ]]; then shift 2; fi
case "$1" in
  branch) echo codex/cached-logit-betti-test ;;
  status) exit "${TEST_STATUS_RC:-0}" ;;
  *) exit 1 ;;
esac
''')
            executable.chmod(0o755)
            environment = dict(os.environ, WORK=str(root/'work'), PATH=str(empty),
                TEST_GIT_DIR=str(git_dir), TEST_EMPTY_PATH=str(empty),
                TEST_STATUS_RC='128' if fail_status else '0',
                TEST_LOAD_RC='1' if fail_load else '0', TEST_GUARD=str(GUARD))
            script = '''set -euo pipefail
module() {
  [[ "$1" == load && "$2" == git ]] || return 1
  [[ "$TEST_LOAD_RC" == 0 ]] || return 1
  export PATH="$TEST_GIT_DIR:$PATH"
  module_loads=$((module_loads+1))
}
module_loads=0
source "$TEST_GUARD"
[[ "$module_loads" == 1 ]]
# Simulate the purge in env_a100.sh or the CPU batch environment.
export PATH="$TEST_EMPTY_PATH"
cached_logit_require_git
[[ "$module_loads" == 2 ]]
git --version
'''
            return subprocess.run(['/bin/bash', '-c', script], env=environment,
                                  text=True, capture_output=True)

    def test_loads_git_before_guard_and_restores_it_after_purge(self):
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('git test fixture', result.stdout)

    def test_failed_module_load_stops_with_clear_error(self):
        result = self.run_guard(fail_load=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Cannot load Git', result.stderr)

    def test_failed_git_status_cannot_look_like_clean_checkout(self):
        result = self.run_guard(fail_status=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Cannot inspect disposable checkout', result.stderr)

    def test_jobs_recheck_git_after_environment_setup(self):
        for name, setup in (('cache.slurm', 'env_a100.sh'),
                            ('optimize.slurm', '/bin/activate')):
            script = (GUARD.parent / name).read_text()
            self.assertGreater(script.index('\ncached_logit_require_git\n'),
                               script.index(setup))
            self.assertIn('[[ "$PWD" -ef "$GNBM_REPO_DIR" ]]', script)


if __name__ == '__main__':
    unittest.main()
