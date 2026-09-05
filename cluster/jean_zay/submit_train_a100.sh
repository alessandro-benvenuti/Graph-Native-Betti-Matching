#!/usr/bin/env bash
# Submit one training segment to a single Jean Zay A100 node.
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

venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "A100 project Python is not executable: $python_bin" >&2
  echo "Run setup_environment_a100.sh and source env_a100.sh." >&2
  exit 2
fi
if ! "$python_bin" -c 'import medpy, nibabel, pyvista' >/dev/null 2>&1; then
  echo "The A100 environment is missing dataset dependencies." >&2
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
for split in train val; do
  for folder in raw seg vtp; do
    required="$SYNTHETIC_MRI_DATASET/$split/$folder"
    if [[ ! -d "$required" ]]; then
      echo "Invalid SYNTHETIC_MRI_DATASET; missing: $required" >&2
      exit 2
    fi
  done
done

echo "Running A100 configuration and dataset preflight on the login node..."
(
  cd "$repo_dir"
  "$python_bin" cluster/jean_zay/preflight_training.py --config "$config"
)
if [[ -n "${GNBM_INITIAL_WEIGHTS:-}" && ! -f "$GNBM_INITIAL_WEIGHTS" \
      && "${GNBM_ALLOW_PENDING_INITIAL_WEIGHTS:-0}" != "1" ]]; then
  echo "Initial checkpoint does not exist: $GNBM_INITIAL_WEIGHTS" >&2
  exit 2
fi
if [[ -n "${GNBM_RESUME_CHECKPOINT:-}" && ! -f "$GNBM_RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint does not exist: $GNBM_RESUME_CHECKPOINT" >&2
  exit 2
fi

qos="${GNBM_QOS:-qos_gpu_a100-t3}"
walltime="${GNBM_WALLTIME:-20:00:00}"
gpus="${GNBM_GPUS:-1}"
case "$gpus" in
  1|2|4|8) ;;
  *) echo "GNBM_GPUS must be 1, 2, 4, or 8 on one A100 node." >&2; exit 2 ;;
esac
case "$qos" in
  qos_gpu_a100-dev|qos_gpu_a100-t3) ;;
  *) echo "GNBM_QOS must be qos_gpu_a100-dev or qos_gpu_a100-t3." >&2; exit 2 ;;
esac
if [[ ! "$walltime" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "GNBM_WALLTIME must use HH:MM:SS." >&2
  exit 2
fi
hours=$((10#${walltime%%:*}))
rest="${walltime#*:}"
minutes="${rest%%:*}"
seconds="${rest##*:}"
total_seconds=$((hours * 3600 + 10#$minutes * 60 + 10#$seconds))
if [[ "$qos" == "qos_gpu_a100-dev" && "$total_seconds" -gt 7200 ]]; then
  echo "qos_gpu_a100-dev cannot exceed 2 hours." >&2
  exit 2
fi
if [[ "$qos" == "qos_gpu_a100-t3" && "$total_seconds" -gt 72000 ]]; then
  echo "qos_gpu_a100-t3 cannot exceed 20 hours." >&2
  exit 2
fi

export GNBM_REPO_DIR="$repo_dir"
export GNBM_CONFIG="$config"
export GNBM_RUN_NAME="$run_name"
export GNBM_GPUS="$gpus"
export GNBM_GPUS_PER_NODE="$gpus"
export GNBM_VENV="$venv"

dependency_args=()
if [[ -n "${GNBM_DEPENDENCY:-}" ]]; then
  if [[ ! "$GNBM_DEPENDENCY" =~ ^(afterok|afterany):[0-9]+$ ]]; then
    echo "GNBM_DEPENDENCY must look like afterok:12345 or afterany:12345." >&2
    exit 2
  fi
  dependency_args+=(--dependency="$GNBM_DEPENDENCY")
fi

log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
submission="$(sbatch \
  "${dependency_args[@]}" \
  --chdir="$repo_dir" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gres="gpu:$gpus" \
  --cpus-per-task="$((8 * gpus))" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  "$repo_dir/cluster/jean_zay/train_a100.slurm")"

echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Log:   $log_dir/gnbm-a100-train-$job_id.out"
echo "GPUs:  $gpus A100(s), global batch=$((gpus * ${GNBM_BATCH_SIZE:-32}))"
