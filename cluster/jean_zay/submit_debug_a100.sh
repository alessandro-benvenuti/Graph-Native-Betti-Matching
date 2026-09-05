#!/usr/bin/env bash
# Submit the bounded one-A100 migration smoke test.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on Jean Zay." >&2
  exit 2
fi
if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH are not defined." >&2
  exit 2
fi
export GNBM_REPO_DIR="$repo_dir"
export GNBM_VENV="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
export GNBM_OUTPUT_DIR="${GNBM_OUTPUT_DIR:-$SCRATCH/experiments/gnbm-a100-debug}"
for name in SYNTHETIC_MRI_DATASET GNBM_MRI_CHECKPOINT; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
if [[ ! -f "$GNBM_MRI_CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $GNBM_MRI_CHECKPOINT" >&2
  exit 2
fi
log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
submission="$(sbatch \
  --chdir="$repo_dir" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/debug_a100.slurm")"
echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Log:   $log_dir/gnbm-a100-debug-$job_id.out"
echo "Error: $log_dir/gnbm-a100-debug-$job_id.err"
