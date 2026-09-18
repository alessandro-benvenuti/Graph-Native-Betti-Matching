#!/usr/bin/env bash
# Submit the fixed ten-patch node/edge Betti audit from a Jean-Zay login node.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_checkpoint="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g20_seed364505/models/best_metric_checkpoint.pt"
checkpoint="${1:-$default_checkpoint}"
config_output_root="${GNBM_OUTPUT_DIR:-${SCRATCH:-}/experiments/gnbm}"
output="${2:-$config_output_root/node-edge-betti-diagnostic-g20-seed364505}"
config="configs/experiments/boundary_gamma_sweep_500/finetune_node_edge_focal_g20.yaml"
qos="${GNBM_DIAGNOSTIC_QOS:-qos_gpu_a100-dev}"
walltime="${GNBM_DIAGNOSTIC_WALLTIME:-02:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean-Zay login node." >&2
  exit 2
fi
if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH are not defined." >&2
  exit 2
fi
if [[ -z "${SYNTHETIC_MRI_DATASET:-}" ]]; then
  echo "SYNTHETIC_MRI_DATASET must point to the corrected boundary dataset." >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Output already exists; refusing to overwrite it: $output" >&2
  exit 2
fi
case "$qos" in
  qos_gpu_a100-dev|qos_gpu_a100-t3) ;;
  *) echo "Unsupported A100 QoS: $qos" >&2; exit 2 ;;
esac

export GNBM_REPO_DIR="$repo_dir"
export GNBM_OUTPUT_DIR="$config_output_root"
export GNBM_NODE_EDGE_BETTI_CHECKPOINT="$checkpoint"
export GNBM_NODE_EDGE_BETTI_CONFIG="$config"
export GNBM_NODE_EDGE_BETTI_OUTPUT="$output"

log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir"
submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/diagnose_node_edge_betti_a100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Output: $output"
echo "Log:    $log_dir/gnbm-node-edge-betti-diagnostic-$job_id.out"
