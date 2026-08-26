#!/usr/bin/env bash
# Submit a one- or multi-GPU H100 training segment.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CONFIG RUN_NAME" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$1"
run_name="$2"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH are not defined." >&2
  exit 2
fi
if [[ ! -f "$repo_dir/$config" && ! -f "$config" ]]; then
  echo "Configuration does not exist: $config" >&2
  exit 2
fi
if ! python -c 'import medpy, nibabel, pyvista' >/dev/null 2>&1; then
  echo "The active Jean-Zay environment is missing dataset reader dependencies." >&2
  echo "Run cluster/jean_zay/setup_environment.sh, then source env.sh again." >&2
  exit 2
fi
if [[ -n "${GNBM_INITIAL_WEIGHTS:-}" && -n "${GNBM_RESUME_CHECKPOINT:-}" ]]; then
  echo "Set only one of GNBM_INITIAL_WEIGHTS and GNBM_RESUME_CHECKPOINT." >&2
  exit 2
fi
for name in SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
if [[ -n "${GNBM_INITIAL_WEIGHTS:-}" && ! -f "$GNBM_INITIAL_WEIGHTS" ]]; then
  echo "Initial checkpoint does not exist: $GNBM_INITIAL_WEIGHTS" >&2
  exit 2
fi
if [[ -n "${GNBM_RESUME_CHECKPOINT:-}" && ! -f "$GNBM_RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint does not exist: $GNBM_RESUME_CHECKPOINT" >&2
  exit 2
fi

qos="${GNBM_QOS:-qos_gpu_h100-t3}"
walltime="${GNBM_WALLTIME:-20:00:00}"
gpus="${GNBM_GPUS:-1}"
case "$gpus" in
  1|2|4) ;;
  *) echo "GNBM_GPUS must be 1, 2, or 4 (one H100 node)." >&2; exit 2 ;;
esac
nodes=1
gpus_per_node="$gpus"
case "$qos" in
  qos_gpu_h100-dev|qos_gpu_h100-t3|qos_gpu_h100-t4) ;;
  *)
    echo "GNBM_QOS must be qos_gpu_h100-dev, qos_gpu_h100-t3, or qos_gpu_h100-t4." >&2
    exit 2
    ;;
esac
if [[ ! "$walltime" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "GNBM_WALLTIME must use HH:MM:SS (for example 20:00:00)." >&2
  exit 2
fi
hours="${walltime%%:*}"
hours=$((10#$hours))
remainder="${walltime#*:}"
minutes="${remainder%%:*}"
seconds="${remainder##*:}"
total_seconds=$((hours * 3600 + 10#$minutes * 60 + 10#$seconds))
if [[ "$qos" == "qos_gpu_h100-dev" ]]; then
  if (( gpus > 2 )); then
    echo "Development jobs are deliberately limited here to at most 2 H100s." >&2
    exit 2
  fi
  if (( total_seconds > 7200 )); then
    echo "qos_gpu_h100-dev cannot exceed 2 hours." >&2
    exit 2
  fi
fi
if [[ "$qos" == "qos_gpu_h100-t3" && "$total_seconds" -gt 72000 ]]; then
  echo "qos_gpu_h100-t3 cannot exceed 20 hours; use qos_gpu_h100-t4." >&2
  exit 2
fi
if [[ "$qos" == "qos_gpu_h100-t4" && "$total_seconds" -gt 360000 ]]; then
  echo "qos_gpu_h100-t4 cannot exceed 100 hours." >&2
  exit 2
fi

export GNBM_REPO_DIR="$repo_dir"
export GNBM_CONFIG="$config"
export GNBM_RUN_NAME="$run_name"
export GNBM_GPUS="$gpus"
export GNBM_GPUS_PER_NODE="$gpus_per_node"

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"

submission="$(sbatch \
  --chdir="$repo_dir" \
  --nodes="$nodes" \
  --ntasks="$nodes" \
  --ntasks-per-node=1 \
  --gres="gpu:$gpus_per_node" \
  --cpus-per-task="$((10 * gpus_per_node))" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/train_h100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Log:   $log_dir/gnbm-train-$job_id.out"
echo "GPUs:  $gpus ($nodes node(s), $gpus_per_node GPU(s)/node)"
