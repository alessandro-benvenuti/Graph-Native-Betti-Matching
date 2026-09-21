#!/usr/bin/env bash
# Submit paired best-checkpoint inference on all GT-loop validation patches.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_pilot_root="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-node-edge-betti-pretrained-pilot4000-a100"
pilot_root="${1:-${GNBM_LOOP_PILOT_ROOT:-$default_pilot_root}}"
output="${2:-${GNBM_LOOP_OUTPUT:-$pilot_root/loop-patch-validation-best}}"
control_run="$pilot_root/node_edge_betti_pilot4000_control_e100_seed364505"
betti_run="$pilot_root/node_edge_betti_pilot4000_nodeaware_e100_seed364505"
qos="${GNBM_LOOP_QOS:-qos_gpu_a100-dev}"
walltime="${GNBM_LOOP_WALLTIME:-02:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
if [[ -z "${WORK:-}" || -z "${SYNTHETIC_MRI_DATASET:-}" ]]; then
  echo "WORK and SYNTHETIC_MRI_DATASET must be set." >&2
  exit 2
fi
for split in val; do
  for folder in raw seg vtp; do
    if [[ ! -d "$SYNTHETIC_MRI_DATASET/$split/$folder" ]]; then
      echo "Invalid corrected dataset; missing: $SYNTHETIC_MRI_DATASET/$split/$folder" >&2
      exit 2
    fi
  done
done
for path in \
  "$control_run/resolved-config.yaml" \
  "$control_run/models/best_metric_checkpoint.pt" \
  "$betti_run/resolved-config.yaml" \
  "$betti_run/models/best_metric_checkpoint.pt"; do
  if [[ ! -f "$path" ]]; then
    echo "Required pilot artifact is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$output" ]]; then
  echo "Output already exists; refusing to overwrite it: $output" >&2
  exit 2
fi
case "$qos" in
  qos_gpu_a100-dev|qos_gpu_a100-t3) ;;
  *) echo "Unsupported A100 QoS: $qos" >&2; exit 2 ;;
esac

export GNBM_REPO_DIR="$repo_dir"
export GNBM_LOOP_CONTROL_RUN="$control_run"
export GNBM_LOOP_BETTI_RUN="$betti_run"
export GNBM_LOOP_OUTPUT="$output"

log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir"
submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/evaluate_loop_patches_a100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Output: $output"
echo "Log:    $log_dir/gnbm-loop-val-$job_id.out"
