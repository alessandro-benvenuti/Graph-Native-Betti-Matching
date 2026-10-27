#!/usr/bin/env bash
# Source this file before using the project on Jean Zay A100 nodes.

if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH must be defined by the Jean Zay environment." >&2
  return 2 2>/dev/null || exit 2
fi

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0

export GNBM_REPO_DIR="${GNBM_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$GNBM_REPO_DIR/cluster/jean_zay/scratch_paths.sh"
export GNBM_VENV="${GNBM_A100_VENV:-$SCRATCH/venvs/vascular-graph-extraction-a100-torch230}"
source "$GNBM_REPO_DIR/cluster/jean_zay/wandb_env.sh"
export PATH="$WORK/tools/uv:$PATH"
export TORCH_EXTENSIONS_DIR="${GNBM_A100_EXTENSIONS_DIR:-$SCRATCH/.cache/torch-extensions/gnbm-a100-torch230-sm80}"
export PYVISTA_OFF_SCREEN="true"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED="1"
export PYTHONNOUSERSITE="1"
export TORCH_CUDA_ARCH_LIST="8.0"
export GNBM_DISTRIBUTED_TIMEOUT_MINUTES="${GNBM_DISTRIBUTED_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

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
  "$WANDB_CACHE_DIR" \
  "$WANDB_DATA_DIR" \
  "$TMPDIR"

if [[ ! -x "$GNBM_VENV/bin/python" ]]; then
  echo "Missing Jean Zay A100 environment: $GNBM_VENV" >&2
  echo "Run cluster/jean_zay/setup_environment_a100.sh first." >&2
  return 2 2>/dev/null || exit 2
fi

source "$GNBM_VENV/bin/activate"
cd "$GNBM_REPO_DIR"
