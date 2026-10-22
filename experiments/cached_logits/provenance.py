"""Capture provenance on a login node; verify code bytes without Git in jobs.

This module deliberately uses only the Python standard library, so it runs
before importing the model or loading any images.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BRANCH = 'codex/cached-logit-betti-test'
REMOTE = 'https://github.com/alessandro-benvenuti/Graph-Native-Betti-Matching.git'
SNAPSHOT_ENV = 'GNBM_CODE_PROVENANCE_FILE'


def _git(root, *arguments):
    result = subprocess.run(['git', '-C', str(root), *arguments],
                            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or '(Git emitted no error text)'
        raise RuntimeError(f"git {' '.join(arguments)} failed (exit {result.returncode}): {detail}")
    return result.stdout.strip()


def _inside(path, root):
    # Keep this login-node helper usable with older system Python versions.
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _source_files(root, tracked):
    """Reject newly added Python sources in import roots, excluding runtime trees."""
    sources = set()
    directories = {Path(name).parts[0] for name in tracked if len(Path(name).parts) > 1}
    ignored = {'.git', '__pycache__', '.venv', 'venv', 'build', 'dist',
               'artifacts', 'output', 'outputs', 'results', 'logs', 'wandb',
               'raw', 'processed', 'generated', 'cache'}
    sources.update(path.name for path in root.glob('*.py'))
    for directory in sorted(directories - ignored):
        for current, children, files in os.walk(root / directory):
            children[:] = sorted(child for child in children if child not in ignored)
            for name in files:
                if name.endswith('.py'):
                    sources.add((Path(current) / name).relative_to(root).as_posix())
    return sorted(sources)


def capture(root=None):
    """Login-only: require the authorized clean independent clone."""
    root = Path(ROOT if root is None else root).resolve()
    if not (root / '.git').is_dir() or not (root / '.git/cached-logit-disposable').is_file():
        raise ValueError('Not the marked independent disposable clone')
    if _git(root, 'remote', 'get-url', 'origin') != REMOTE:
        raise ValueError('Unexpected disposable origin')
    branch = _git(root, 'branch', '--show-current')
    if branch != BRANCH:
        raise ValueError('Wrong disposable branch')
    if _git(root, 'status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Commit changes before submitting the diagnostic')
    if (root / _git(root, 'rev-parse', '--git-common-dir')).resolve() != root / '.git':
        raise ValueError('Expected an independent clone, not a worktree')
    commit = _git(root, 'rev-parse', 'HEAD')
    # NUL paths are decoded without stripping names; normal repo paths are UTF-8.
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    names = sorted(name.decode() for name in result.stdout.split(b'\0') if name)
    files = {}
    for name in names:
        path = root / name
        if not path.is_file() or not _inside(path.resolve(), root):
            raise ValueError('Tracked file missing or outside the clone: ' + name)
        files[name] = _digest_file(path)
    value = dict(schema=1, repository=str(root), commit=commit, branch=branch,
                 dirty='', files=files, python_sources=_source_files(root, names))
    value['content_sha256'] = _digest_json(value)
    destination = root / '.git/cached-logit-provenance'
    destination.mkdir(exist_ok=True)
    path = destination / (commit + '-' + value['content_sha256'][:16] + '.json')
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing immutable provenance snapshot differs')
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
    return path


def verify(path, root=None):
    """Compute-node safe: hashes only; never launches Git or another process."""
    root = Path(ROOT if root is None else root).resolve()
    value = json.loads(Path(path).read_text())
    digest = value.pop('content_sha256')
    if _digest_json(value) != digest:
        raise ValueError('Provenance snapshot content hash changed')
    if value['schema'] != 1 or Path(value['repository']).resolve() != root:
        raise ValueError('Provenance snapshot is for another checkout')
    if value['branch'] != BRANCH or value['dirty']:
        raise ValueError('Snapshot was not captured from the clean disposable branch')
    if not (root / '.git').is_dir() or not (root / '.git/cached-logit-disposable').is_file():
        raise ValueError('Independent clone marker is missing')
    for name, expected in value['files'].items():
        file_path = root / name
        if not _inside(file_path.resolve(), root):
            raise ValueError('Snapshot file path escapes checkout: ' + name)
        if not file_path.is_file() or _digest_file(file_path) != expected:
            raise ValueError('Code changed after login-node provenance capture: ' + name)
    if _source_files(root, value['files']) != value['python_sources']:
        raise ValueError('Python source files changed after login-node provenance capture')
    return dict(commit=value['commit'], branch=value['branch'], dirty='',
                verification='login_snapshot_sha256', source_sha256=digest)


def code_provenance():
    snapshot = os.environ.get(SNAPSHOT_ENV)
    if snapshot:
        return verify(snapshot)
    # Batch jobs must not silently fall back to Git if submission omitted snapshot.
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Missing login-node provenance snapshot; use cluster/submit.sh')
    return dict(commit=_git(ROOT, 'rev-parse', 'HEAD'),
                branch=_git(ROOT, 'branch', '--show-current'),
                dirty=_git(ROOT, 'status', '--porcelain', '--untracked-files=normal'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('capture', 'verify'))
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        print(capture(args.repo))
    else:
        if args.manifest is None:
            parser.error('verify requires --manifest')
        print(json.dumps(verify(args.manifest, args.repo), sort_keys=True))


if __name__ == '__main__':
    main()
