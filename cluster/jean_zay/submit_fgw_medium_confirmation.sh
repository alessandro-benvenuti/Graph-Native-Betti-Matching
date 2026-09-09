#!/usr/bin/env bash
# Submit three full dependency trees for the 8,192-patch FGW confirmation.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_root="configs/experiments/fgw_medium_confirmation"
default_pretrained_checkpoint="/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/pretrain_full_mixed_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt"
initial_weights="${GNBM_INITIAL_WEIGHTS:-$default_pretrained_checkpoint}"

for name in WORK SCRATCH SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done

pilot_gpus="${GNBM_GPUS:-1}"
case "$pilot_gpus" in
  1|2|4|8) ;;
  *) echo "GNBM_GPUS must be 1, 2, 4, or 8." >&2; exit 2 ;;
esac
export GNBM_GPUS="$pilot_gpus"
export GNBM_BATCH_SIZE="${GNBM_BATCH_SIZE:-$((32 / pilot_gpus))}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fgw-medium-confirmation}"

seeds=(364505 364506 364507)
configs=()
runs=()
for seed in "${seeds[@]}"; do
  suffix=""
  if [[ "$seed" != "364505" ]]; then
    suffix="_seed$seed"
  fi
  configs+=(
    "$config_root/trunk_hungarian$suffix.yaml"
    "$config_root/continue_hungarian$suffix.yaml"
    "$config_root/continue_fgw_a04$suffix.yaml"
  )
  runs+=(
    "fgw_medium_pretrained_trunk_hungarian_e5_seed$seed"
    "fgw_medium_pretrained_continue_hungarian_e100_seed$seed"
    "fgw_medium_pretrained_continue_fgw_a04_e100_seed$seed"
  )
done

for run in "${runs[@]}"; do
  if [[ -e "$GNBM_OUTPUT_DIR/$run" ]]; then
    echo "Run directory already exists; refusing a partial matrix: $GNBM_OUTPUT_DIR/$run" >&2
    exit 2
  fi
done
if [[ ! -f "$initial_weights" ]]; then
  echo "Initial checkpoint does not exist: $initial_weights" >&2
  exit 2
fi

venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "A100 project Python is not executable: $python_bin" >&2
  exit 2
fi

echo "Validating every configuration before submitting any job..."
(
  cd "$repo_dir"
  "$python_bin" -c \
    'import sys; from configs import load_config; [load_config(path) for path in sys.argv[1:]]' \
    "${configs[@]}"
  # All nine configurations use the same dataset membership and transforms.
  "$python_bin" cluster/jean_zay/preflight_training.py --config "${configs[0]}"
)
export GNBM_SKIP_PREFLIGHT=1

for seed in "${seeds[@]}"; do
  suffix=""
  if [[ "$seed" != "364505" ]]; then
    suffix="_seed$seed"
  fi
  trunk_run="fgw_medium_pretrained_trunk_hungarian_e5_seed$seed"
  trunk_checkpoint="$GNBM_OUTPUT_DIR/$trunk_run/models/latest_checkpoint.pt"
  unset GNBM_RESUME_CHECKPOINT GNBM_DEPENDENCY GNBM_ALLOW_PENDING_RESUME
  export GNBM_INITIAL_WEIGHTS="$initial_weights"

  trunk_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
    "$config_root/trunk_hungarian$suffix.yaml" "$trunk_run")"
  trunk_job="$(printf '%s\n' "$trunk_submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
  if [[ -z "$trunk_job" ]]; then
    echo "Could not parse the trunk job ID for seed $seed:" >&2
    printf '%s\n' "$trunk_submission" >&2
    exit 1
  fi
  printf '%s\n' "$trunk_submission"

  unset GNBM_INITIAL_WEIGHTS
  export GNBM_RESUME_CHECKPOINT="$trunk_checkpoint"
  export GNBM_DEPENDENCY="afterok:$trunk_job"
  export GNBM_ALLOW_PENDING_RESUME=1
  for method in hungarian fgw_a04; do
    run="fgw_medium_pretrained_continue_${method}_e100_seed$seed"
    bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
      "$config_root/continue_${method}$suffix.yaml" "$run"
  done
  echo "seed=$seed trunk_job=$trunk_job dependent_branches=2"
done

echo "Submitted 3 trunks and 6 dependent continuations."
echo "Queue: squeue -u $USER"
