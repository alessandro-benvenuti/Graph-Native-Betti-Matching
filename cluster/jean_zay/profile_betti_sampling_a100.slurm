#!/usr/bin/env bash
#SBATCH --job-name=gnbm-betti-profile
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00

set -euo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
python_bin="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/python"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "checkpoint=$GNBM_PROFILE_CHECKPOINT"
echo "output=$GNBM_PROFILE_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

exec "$python_bin" -u "$repo_dir/scripts/profile_betti_sampling.py" \
  --config "$repo_dir/configs/experiments/staged_betti_activation/subsampled_epoch300.yaml" \
  --checkpoint "$GNBM_PROFILE_CHECKPOINT" \
  --output "$GNBM_PROFILE_OUTPUT" \
  --warmup-steps "${GNBM_PROFILE_WARMUP_STEPS:-10}" \
  --measured-steps "${GNBM_PROFILE_MEASURED_STEPS:-100}" \
  --gradient-repeats "${GNBM_PROFILE_GRADIENT_REPEATS:-12}" \
  --batch-size "${GNBM_PROFILE_BATCH_SIZE:-16}" \
  --workers "${GNBM_PROFILE_WORKERS:-4}"

