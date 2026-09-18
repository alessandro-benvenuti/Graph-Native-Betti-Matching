#!/usr/bin/env bash
# Frozen-checkpoint node/edge Betti diagnostic on one Jean-Zay A100.
#SBATCH --job-name=gnbm-node-edge-betti-diagnostic
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

set -eo pipefail

repo_dir="${GNBM_REPO_DIR:-$SLURM_SUBMIT_DIR}"
source "$repo_dir/cluster/jean_zay/env_a100.sh"
set -u

required=(
  GNBM_NODE_EDGE_BETTI_CHECKPOINT
  GNBM_NODE_EDGE_BETTI_CONFIG
  GNBM_NODE_EDGE_BETTI_OUTPUT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing batch environment variable: $name" >&2
    exit 2
  fi
done

echo "job=$SLURM_JOB_ID host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python "$repo_dir/scripts/diagnose_node_edge_betti.py" \
  --config "$GNBM_NODE_EDGE_BETTI_CONFIG" \
  --checkpoint "$GNBM_NODE_EDGE_BETTI_CHECKPOINT" \
  --output-dir "$GNBM_NODE_EDGE_BETTI_OUTPUT" \
  --dataset synthetic_mri \
  --split test \
  --batch-size 1 \
  --workers 4 \
  --device cuda \
  --aggregation hybrid \
  --alpha 0.5 \
  --unmatched-object-threshold 0.25 \
  --max-active-unmatched 8
