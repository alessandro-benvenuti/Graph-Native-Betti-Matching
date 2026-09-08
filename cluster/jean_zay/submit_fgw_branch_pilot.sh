#!/usr/bin/env bash
# Submit one Hungarian trunk and four dependent continuation branches on A100s.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_root="configs/experiments/fgw_branch_pilot"
trunk_run="fgw_pilot_pretrained_trunk_hungarian_e5_seed364505"
default_pretrained_checkpoint="/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/pretrain_full_mixed_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt"
if [[ -z "${GNBM_OUTPUT_DIR:-}" ]]; then
  echo "GNBM_OUTPUT_DIR is not set." >&2
  exit 2
fi
trunk_checkpoint="$GNBM_OUTPUT_DIR/$trunk_run/models/latest_checkpoint.pt"

branch_configs=(
  continue_hungarian.yaml
  continue_fgw_a04.yaml
  continue_fgw_a06.yaml
  continue_fgw_a08.yaml
)
branch_runs=(
  fgw_pilot_pretrained_continue_hungarian_e15_seed364505
  fgw_pilot_pretrained_continue_fgw_a04_e15_seed364505
  fgw_pilot_pretrained_continue_fgw_a06_e15_seed364505
  fgw_pilot_pretrained_continue_fgw_a08_e15_seed364505
)

for run in "$trunk_run" "${branch_runs[@]}"; do
  if [[ -e "$GNBM_OUTPUT_DIR/$run" ]]; then
    echo "Run directory already exists; refusing a partial pilot: $GNBM_OUTPUT_DIR/$run" >&2
    exit 2
  fi
done

pilot_gpus="${GNBM_GPUS:-1}"
case "$pilot_gpus" in
  1|2|4|8) ;;
  *) echo "GNBM_GPUS must be 1, 2, 4, or 8." >&2; exit 2 ;;
esac
export GNBM_GPUS="$pilot_gpus"
# Preserve global batch 32 unless the caller explicitly requests another size.
export GNBM_BATCH_SIZE="${GNBM_BATCH_SIZE:-$((32 / pilot_gpus))}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fgw-branch-pilot-seed364505}"
unset GNBM_RESUME_CHECKPOINT GNBM_DEPENDENCY GNBM_ALLOW_PENDING_RESUME
export GNBM_INITIAL_WEIGHTS="${GNBM_INITIAL_WEIGHTS:-$default_pretrained_checkpoint}"

trunk_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
  "$config_root/trunk_hungarian.yaml" "$trunk_run")"
trunk_job="$(printf '%s\n' "$trunk_submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
if [[ -z "$trunk_job" ]]; then
  echo "Could not parse the trunk job ID:" >&2
  printf '%s\n' "$trunk_submission" >&2
  exit 1
fi
printf '%s\n' "$trunk_submission"

unset GNBM_INITIAL_WEIGHTS
export GNBM_RESUME_CHECKPOINT="$trunk_checkpoint"
export GNBM_DEPENDENCY="afterok:$trunk_job"
export GNBM_ALLOW_PENDING_RESUME=1
for index in "${!branch_configs[@]}"; do
  bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
    "$config_root/${branch_configs[$index]}" "${branch_runs[$index]}"
done

echo "Trunk job: $trunk_job"
echo "The four continuations will start together after its epoch-5 checkpoint exists."
echo "Queue: squeue -u $USER"
