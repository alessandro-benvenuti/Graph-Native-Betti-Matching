#!/usr/bin/env bash
# Submit paired 100-epoch control/node-aware Betti continuations on A100s.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_root="configs/experiments/node_edge_betti_pilot_4000"
default_checkpoint="/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g20_seed364505/models/best_metric_checkpoint.pt"
initial_weights="${GNBM_INITIAL_WEIGHTS:-$default_checkpoint}"
segments="${GNBM_PILOT_SEGMENTS:-2}"

for name in WORK SCRATCH SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
if [[ ! -f "$initial_weights" ]]; then
  echo "Initial checkpoint does not exist: $initial_weights" >&2
  exit 2
fi
case "$segments" in
  1|2|3) ;;
  *) echo "GNBM_PILOT_SEGMENTS must be 1, 2, or 3." >&2; exit 2 ;;
esac

configs=(
  "$config_root/control.yaml"
  "$config_root/node_aware_betti.yaml"
)
runs=(
  node_edge_betti_pilot4000_control_e100_seed364505
  node_edge_betti_pilot4000_nodeaware_e100_seed364505
)
for run in "${runs[@]}"; do
  if [[ -e "$GNBM_OUTPUT_DIR/$run" ]]; then
    echo "Run directory already exists; refusing a partial pair: $GNBM_OUTPUT_DIR/$run" >&2
    exit 2
  fi
done

venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "A100 project Python is not executable: $python_bin" >&2
  exit 2
fi

echo "Validating the paired configurations and deterministic dataset subset..."
(
  cd "$repo_dir"
  "$python_bin" -c \
    'import sys; from configs import load_config; [load_config(path) for path in sys.argv[1:]]' \
    "${configs[@]}"
  "$python_bin" cluster/jean_zay/preflight_training.py --config "${configs[0]}"
)

export GNBM_SKIP_PREFLIGHT=1
# This campaign always starts both arms from the same model-only checkpoint;
# never inherit full-state resume settings from a previous shell session.
unset GNBM_RESUME_CHECKPOINT GNBM_ALLOW_PENDING_RESUME
export GNBM_INITIAL_WEIGHTS="$initial_weights"
export GNBM_AUTO_RESUME=1
export GNBM_GPUS="${GNBM_GPUS:-1}"
case "$GNBM_GPUS" in
  1|2|4|8) ;;
  *) echo "GNBM_GPUS must be 1, 2, 4, or 8." >&2; exit 2 ;;
esac
export GNBM_BATCH_SIZE="${GNBM_BATCH_SIZE:-$((32 / GNBM_GPUS))}"
export GNBM_QOS="${GNBM_QOS:-qos_gpu_a100-t3}"
export GNBM_WALLTIME="${GNBM_WALLTIME:-20:00:00}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-node-edge-betti-pilot-4000}"
# Compute nodes can spend tens of minutes retrying a blocked W&B connection.
# Keep training independent of external connectivity; runs can be synced later.
export WANDB_MODE="${GNBM_PILOT_WANDB_MODE:-offline}"

submit_chain() {
  local config="$1"
  local run="$2"
  local previous_job=""
  local segment submission job
  unset GNBM_DEPENDENCY
  for ((segment = 1; segment <= segments; segment++)); do
    if [[ -n "$previous_job" ]]; then
      export GNBM_DEPENDENCY="afterany:$previous_job"
    fi
    submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
      "$config" "$run")"
    job="$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
    if [[ -z "$job" ]]; then
      echo "Could not parse job ID for $run segment $segment:" >&2
      printf '%s\n' "$submission" >&2
      exit 1
    fi
    printf '%s\n' "$submission"
    echo "arm=$run segment=$segment job=$job"
    previous_job="$job"
  done
  unset GNBM_DEPENDENCY
}

for index in "${!configs[@]}"; do
  submit_chain "${configs[$index]}" "${runs[$index]}"
done

echo "Submitted paired control and node-aware Betti runs."
echo "Each arm has $segments recoverable 20-hour segment(s); later segments exit immediately if training is complete."
echo "Queue: squeue -u $USER"
