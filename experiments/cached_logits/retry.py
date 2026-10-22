"""Choose a fresh smoke-cache path while preserving failed artifacts."""
import argparse
from pathlib import Path


def smoke_cache_output(root, job_id):
    root = Path(root)
    if not str(job_id).isdigit():
        raise ValueError('Expected a numeric Slurm job ID')
    if (root / 'smoke-opt').exists():
        raise FileExistsError('smoke-opt already exists; inspect it before choosing a fresh smoke run')
    output = root / 'smoke-cache'
    if output.exists():
        if (output / 'CACHE_COMPLETE.json').exists():
            raise FileExistsError('A complete smoke cache exists; inspect/reuse it rather than rerunning the network')
        output = root / ('smoke-cache-retry-' + str(job_id))
        if output.exists():
            raise FileExistsError('Retry cache output already exists: ' + str(output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    print(smoke_cache_output(args.root, args.job_id))
