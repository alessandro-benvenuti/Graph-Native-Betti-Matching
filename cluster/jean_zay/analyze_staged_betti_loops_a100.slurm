#!/usr/bin/env bash
#SBATCH --job-name=gnbm-loop-audit
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
set -eo pipefail
cd "${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source cluster/jean_zay/env_a100.sh
set -u
exec "$GNBM_VENV/bin/python" -u -m scripts.analyze_staged_betti_loops \
  --campaign-root "$GNBM_OUTPUT_DIR" \
  --dataset-root "$SYNTHETIC_MRI_DATASET" \
  --output-dir "$GNBM_LOOP_AUDIT_OUTPUT" \
  --epoch 400 --scope "$GNBM_LOOP_AUDIT_SCOPE" \
  --batch-size 32 --workers 8
