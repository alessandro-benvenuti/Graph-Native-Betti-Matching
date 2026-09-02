#!/usr/bin/env bash
# One-GPU batch entry point for the bounded three-model visualization workflow.
#SBATCH --job-name=gnbm-eval-random10
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env.sh"
set -u

required=(
  GNBM_EVAL_BASELINE_CHECKPOINT
  GNBM_EVAL_NODE_FOCAL_CHECKPOINT
  GNBM_EVAL_COMBINED_CHECKPOINT
  GNBM_EVAL_OUTPUT_DIR
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing batch environment variable: $name" >&2
    exit 2
  fi
done

echo "job=$SLURM_JOB_ID host=$(hostname) gpu_count=${SLURM_GPUS_ON_NODE:-1}"
echo "output=$GNBM_EVAL_OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

bash "$repo_dir/scripts/evaluate_random_patches_3d.sh" \
  "$GNBM_EVAL_BASELINE_CHECKPOINT" \
  "$GNBM_EVAL_NODE_FOCAL_CHECKPOINT" \
  "$GNBM_EVAL_COMBINED_CHECKPOINT" \
  "$GNBM_EVAL_OUTPUT_DIR"
