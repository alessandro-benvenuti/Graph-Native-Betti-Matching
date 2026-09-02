#!/usr/bin/env bash
# Submit the bounded random-patch three-model evaluation from a login node.
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "Usage: $0 BASELINE_CHECKPOINT NODE_FOCAL_CHECKPOINT NODE_EDGE_FOCAL_CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
baseline_checkpoint="$1"
node_focal_checkpoint="$2"
combined_checkpoint="$3"
output_dir="${4:-${GNBM_OUTPUT_DIR:-${SCRATCH:-}/experiments/gnbm}/random10_three_models}"
qos="${GNBM_EVAL_QOS:-qos_gpu_h100-dev}"
walltime="${GNBM_EVAL_WALLTIME:-02:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this submission wrapper on a Jean-Zay login node." >&2
  exit 2
fi
if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH are not defined by the current Jean-Zay shell." >&2
  exit 2
fi
if [[ -z "${SYNTHETIC_MRI_DATASET:-}" ]]; then
  echo "Set SYNTHETIC_MRI_DATASET to the new_patches directory." >&2
  exit 2
fi
for checkpoint in "$baseline_checkpoint" "$node_focal_checkpoint" "$combined_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Output already exists; refusing to overwrite it: $output_dir" >&2
  exit 2
fi
case "$qos" in
  qos_gpu_h100-dev|qos_gpu_h100-t3|qos_gpu_h100-t4) ;;
  *) echo "GNBM_EVAL_QOS must be qos_gpu_h100-dev, qos_gpu_h100-t3, or qos_gpu_h100-t4." >&2; exit 2 ;;
esac
if [[ ! "$walltime" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "GNBM_EVAL_WALLTIME must use HH:MM:SS." >&2
  exit 2
fi

export GNBM_REPO_DIR="$repo_dir"
export GNBM_EVAL_BASELINE_CHECKPOINT="$baseline_checkpoint"
export GNBM_EVAL_NODE_FOCAL_CHECKPOINT="$node_focal_checkpoint"
export GNBM_EVAL_COMBINED_CHECKPOINT="$combined_checkpoint"
export GNBM_EVAL_OUTPUT_DIR="$output_dir"

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir"
submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/evaluate_random_patches_3d.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Output: $output_dir"
echo "Log:    $log_dir/gnbm-eval-random10-$job_id.out"
echo "Errors: $log_dir/gnbm-eval-random10-$job_id.err"
