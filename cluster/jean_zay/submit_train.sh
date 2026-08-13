#!/usr/bin/env bash
# Submit a single-GPU V100 training segment.
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

qos="${GNBM_QOS:-qos_gpu-t3}"
walltime="${GNBM_WALLTIME:-20:00:00}"
case "$qos" in
  qos_gpu-t3|qos_gpu-t4) ;;
  *) echo "GNBM_QOS must be qos_gpu-t3 or qos_gpu-t4." >&2; exit 2 ;;
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
if [[ "$qos" == "qos_gpu-t3" && "$total_seconds" -gt 72000 ]]; then
  echo "qos_gpu-t3 cannot exceed 20 hours; use qos_gpu-t4." >&2
  exit 2
fi
if [[ "$qos" == "qos_gpu-t4" && "$total_seconds" -gt 360000 ]]; then
  echo "qos_gpu-t4 cannot exceed 100 hours." >&2
  exit 2
fi

export GNBM_REPO_DIR="$repo_dir"
export GNBM_CONFIG="$config"
export GNBM_RUN_NAME="$run_name"

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"

submission="$(sbatch \
  --chdir="$repo_dir" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/train_v100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Log:   $log_dir/gnbm-train-$job_id.out"
