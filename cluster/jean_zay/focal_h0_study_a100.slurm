#!/usr/bin/env bash
#SBATCH --job-name=gnbm-focal-h0
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
set -eo pipefail
cd "$GNBM_REPO_DIR"
source cluster/jean_zay/env_a100.sh
set -u
export WANDB_PROJECT=gnbm WANDB_RUN_GROUP=edge-focal-g05-h0-study WANDB_MODE=offline
echo "job=$SLURM_JOB_ID host=$(hostname) action=$GNBM_H0_ACTION family=$GNBM_H0_FAMILY endpoint=$GNBM_H0_ENDPOINT"
echo "output=$GNBM_OUTPUT_DIR wandb_project=$WANDB_PROJECT mode=$WANDB_MODE"
if [[ "$GNBM_H0_ACTION" == gpu-smoke ]]; then
  exec "$GNBM_VENV/bin/python" -u -m scripts.optimize_focal_h0 gpu-smoke --output "$GNBM_OUTPUT_DIR"
fi
exec "$GNBM_VENV/bin/python" -u -m scripts.optimize_focal_h0 worker \
  --output "$GNBM_OUTPUT_DIR" --family "$GNBM_H0_FAMILY" \
  --endpoint "$GNBM_H0_ENDPOINT" --resume
