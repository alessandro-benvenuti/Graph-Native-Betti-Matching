#!/usr/bin/env bash
# submit_train.sh supplies QoS, walltime, working directory, and log paths.
#SBATCH --job-name=gnbm-train
#SBATCH --account=vnc@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

set -euo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"

args=(
  --config "$GNBM_CONFIG"
  --output-dir "$GNBM_OUTPUT_DIR"
  --run-name "$GNBM_RUN_NAME"
  --device cuda
)

if [[ -n "${GNBM_BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "$GNBM_BATCH_SIZE")
fi
if [[ -n "${GNBM_WORKERS:-}" ]]; then
  args+=(--workers "$GNBM_WORKERS")
fi

if [[ -n "${GNBM_RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume "$GNBM_RESUME_CHECKPOINT")
elif [[ -n "${GNBM_INITIAL_WEIGHTS:-}" ]]; then
  args+=(--initial-weights "$GNBM_INITIAL_WEIGHTS")
fi

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "config=$GNBM_CONFIG run=$GNBM_RUN_NAME output=$GNBM_OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --ntasks=1 python -u train.py "${args[@]}"
