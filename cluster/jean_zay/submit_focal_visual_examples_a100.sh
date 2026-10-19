#!/usr/bin/env bash
# Submit reproducible full-test focal-loss qualitative screening.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SCRATCH SYNTHETIC_MRI_DATASET; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi

campaign_root="${GNBM_EXTENSION_OUTPUT_DIR:-$SCRATCH/checkpoints/gnbm-boundary-gamma-extension-750-a100}"
source_root="${GNBM_EXTENSION_SOURCE_ROOT:-$SCRATCH/checkpoints/gnbm-boundary-gamma-sweep-500-a100}"
output="${GNBM_FOCAL_FIGURE_OUTPUT:-$SCRATCH/experiments/gnbm-focal-paper-figure-current-best}"
export GNBM_FOCAL_FIGURE_OUTPUT="$output"
export GNBM_EXTENSION_SOURCE_ROOT="$source_root"
export GNBM_OUTPUT_DIR="$campaign_root"
baseline_source="${GNBM_FOCAL_BASELINE_CHECKPOINT:-$campaign_root/finetune_boundary_mri750_baseline_tail250_seed364505/models/best_metric_checkpoint.pt}"
node_source="${GNBM_FOCAL_NODE_CHECKPOINT:-$campaign_root/finetune_boundary_mri750_node_focal_tail250_seed364505/models/best_metric_checkpoint.pt}"
node_edge_source="${GNBM_FOCAL_NODE_EDGE_CHECKPOINT:-$campaign_root/finetune_boundary_mri750_node_edge_focal_g05_tail250_seed364505/models/best_metric_checkpoint.pt}"
export GNBM_REPO_DIR="$repo_dir"

for checkpoint in \
  "$baseline_source" \
  "$node_source" \
  "$node_edge_source"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done
for split_dir in raw seg vtp; do
  if [[ ! -d "$SYNTHETIC_MRI_DATASET/test/$split_dir" ]]; then
    echo "Missing test dataset directory: $SYNTHETIC_MRI_DATASET/test/$split_dir" >&2
    exit 2
  fi
done
if [[ -f "$output/complete" ]]; then
  echo "Screening is already complete: $output" >&2
  exit 2
fi

log_dir="$WORK/logs/graph-native-betti-matching/a100-debug"
snapshot_dir="$output/checkpoint-snapshots"
mkdir -p "$log_dir" "$snapshot_dir"

# Freeze the exact model files before submission.  The baseline and combined
# training jobs may still replace their live best checkpoints while this
# qualitative evaluation is queued or running.
snapshot_checkpoint() {
  local source="$1"
  local destination="$2"
  if [[ ! -s "$destination" ]]; then
    cp -p "$source" "$destination"
  fi
}
snapshot_checkpoint "$baseline_source" "$snapshot_dir/baseline.pt"
snapshot_checkpoint "$node_source" "$snapshot_dir/node_focal.pt"
snapshot_checkpoint "$node_edge_source" "$snapshot_dir/node_edge_focal.pt"
export GNBM_FOCAL_BASELINE_CHECKPOINT="$snapshot_dir/baseline.pt"
export GNBM_FOCAL_NODE_CHECKPOINT="$snapshot_dir/node_focal.pt"
export GNBM_FOCAL_NODE_EDGE_CHECKPOINT="$snapshot_dir/node_edge_focal.pt"

submission="$(sbatch \
  --chdir="$repo_dir" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/focal_visual_examples_a100.slurm")"
echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Output: $output"
echo "Log:    $log_dir/gnbm-focal-figure-$job_id.out"
