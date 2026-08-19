#!/usr/bin/env bash
# Source this file before using the project on Jean Zay.

if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH must be defined by the Jean Zay environment." >&2
  return 2 2>/dev/null || exit 2
fi

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

export GNBM_REPO_DIR="${GNBM_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export GNBM_VENV="${GNBM_VENV:-$WORK/venvs/vascular-graph-extraction-h100-torch231}"
export PATH="$WORK/tools/uv:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/.cache/uv}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK/.cache/torch-extensions/gnbm-h100-torch231-sm90}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK/.cache/matplotlib}"
export PYVISTA_OFF_SCREEN="true"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

# IDRIS' PyTorch modules normally expect Slurm rank metadata. Allow intentional
# CPU-only imports on a login node, while retaining normal checks inside jobs.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export IDR_DEBUG="${IDR_DEBUG:-WARN}"
fi

if [[ -n "${JOBSCRATCH:-}" ]]; then
  export TMPDIR="$JOBSCRATCH"
else
  export TMPDIR="${TMPDIR:-$SCRATCH/tmp/$USER}"
fi

mkdir -p \
  "$UV_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$TMPDIR"

if [[ ! -x "$GNBM_VENV/bin/python" ]]; then
  echo "Missing Jean Zay environment: $GNBM_VENV" >&2
  echo "Run cluster/jean_zay/setup_environment.sh first." >&2
  return 2 2>/dev/null || exit 2
fi

# The venv was created with --system-site-packages so it inherits the loaded
# Jean Zay PyTorch/CUDA module while keeping project packages under $WORK.
source "$GNBM_VENV/bin/activate"
cd "$GNBM_REPO_DIR"
