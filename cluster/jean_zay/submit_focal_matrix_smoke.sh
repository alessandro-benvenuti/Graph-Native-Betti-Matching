#!/usr/bin/env bash
# Production-style W&B smoke test for the hardest focal-matrix recipe.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
for name in WORK SCRATCH PLANTS_DATASET SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
for entry in "$PLANTS_DATASET" "$SYNTHETIC_MRI_DATASET"; do
  if [[ ! -d "$entry" ]]; then
    echo "Dataset directory does not exist: $entry" >&2
    exit 2
  fi
done

config="configs/experiments/focal_matrix_600/smoke_combined_immediate.yaml"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="focal_matrix_smoke_combined_immediate_${timestamp}"
run_dir="$GNBM_OUTPUT_DIR/$run_name"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite smoke output: $run_dir" >&2
  exit 2
fi

export GNBM_REPO_DIR="$repo_dir"
export GNBM_CONFIG="$config"
export GNBM_RUN_NAME="$run_name"
export WANDB_RUN_GROUP="focal-matrix-600-smoke"
unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"

submission="$(sbatch \
  --chdir="$repo_dir" \
  --job-name=gnbm-matrix-smoke \
  --qos=qos_gpu_h100-dev \
  --time=00:45:00 \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/train_h100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Run:   $run_name"
echo "Queue: squeue -j $job_id"
echo "Log:   $log_dir/gnbm-matrix-smoke-$job_id.out"
echo "Error: $log_dir/gnbm-matrix-smoke-$job_id.err"
echo "W&B:   entity=${WANDB_ENTITY:-from-wandb_env.sh} project=${WANDB_PROJECT:-from-wandb_env.sh} group=$WANDB_RUN_GROUP"
