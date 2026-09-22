#!/usr/bin/env bash
# Submit the paired node-focal topology-overfit mechanism test.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint_root="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100"
source_run="${GNBM_MECH_SOURCE_RUN:-$checkpoint_root/finetune_boundary_mri500_node_focal_seed364505}"
checkpoint="${GNBM_MECH_CHECKPOINT:-$source_run/models/best_metric_checkpoint.pt}"
source_config="${GNBM_MECH_SOURCE_CONFIG:-$repo_dir/configs/experiments/full_dataset_node_focal/finetune.yaml}"
default_output="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-node-focal-topology-overfit-a100"
output="${1:-${GNBM_MECH_OUTPUT:-$default_output}}"
qos="${GNBM_MECH_QOS:-qos_gpu_a100-dev}"
walltime="${GNBM_MECH_WALLTIME:-02:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
for name in WORK SYNTHETIC_MRI_DATASET; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
for path in "$checkpoint" "$source_config"; do
  if [[ ! -f "$path" ]]; then
    echo "Required node-focal artifact is missing: $path" >&2
    exit 2
  fi
done
if [[ -f "$output/experiment-complete" ]]; then
  echo "Experiment is already complete: $output" >&2
  exit 2
fi
case "$qos" in
  qos_gpu_a100-dev|qos_gpu_a100-t3) ;;
  *) echo "Unsupported A100 QoS: $qos" >&2; exit 2 ;;
esac

export GNBM_REPO_DIR="$repo_dir"
export GNBM_MECH_CHECKPOINT="$checkpoint"
export GNBM_MECH_SOURCE_CONFIG="$source_config"
export GNBM_MECH_OUTPUT="$output"

log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir"
submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/topology_overfit_node_focal_a100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue:  squeue -j $job_id"
echo "Output: $output"
echo "Log:    $log_dir/gnbm-topo-overfit-$job_id.out"
