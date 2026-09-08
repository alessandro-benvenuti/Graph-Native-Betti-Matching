#!/usr/bin/env bash
# Submit two additional fixed-subset FGW replication trees on A100s.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_root="configs/experiments/fgw_branch_pilot"
default_pretrained_checkpoint="/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm/full-data-fresh250-20260831_160421/pretrain_full_mixed_nodefocal_edgefocal_mm_seed364505/models/best_checkpoint.pt"
initial_weights="${GNBM_INITIAL_WEIGHTS:-$default_pretrained_checkpoint}"

if [[ -z "${GNBM_OUTPUT_DIR:-}" ]]; then
  echo "GNBM_OUTPUT_DIR is not set." >&2
  exit 2
fi

pilot_gpus="${GNBM_GPUS:-1}"
case "$pilot_gpus" in
  1|2|4|8) ;;
  *) echo "GNBM_GPUS must be 1, 2, 4, or 8." >&2; exit 2 ;;
esac
export GNBM_GPUS="$pilot_gpus"
export GNBM_BATCH_SIZE="${GNBM_BATCH_SIZE:-$((32 / pilot_gpus))}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-fgw-replication-fixedsubset}"

seeds=(364506 364507)
methods=(hungarian fgw_a04 fgw_a08)
for seed in "${seeds[@]}"; do
  runs=("fgw_pilot_pretrained_trunk_hungarian_e5_seed$seed")
  for method in "${methods[@]}"; do
    runs+=("fgw_pilot_pretrained_continue_${method}_e15_seed$seed")
  done
  for run in "${runs[@]}"; do
    if [[ -e "$GNBM_OUTPUT_DIR/$run" ]]; then
      echo "Run directory already exists; refusing a partial replication: $GNBM_OUTPUT_DIR/$run" >&2
      exit 2
    fi
  done
done

for seed in "${seeds[@]}"; do
  trunk_run="fgw_pilot_pretrained_trunk_hungarian_e5_seed$seed"
  trunk_checkpoint="$GNBM_OUTPUT_DIR/$trunk_run/models/latest_checkpoint.pt"
  unset GNBM_RESUME_CHECKPOINT GNBM_DEPENDENCY GNBM_ALLOW_PENDING_RESUME
  export GNBM_INITIAL_WEIGHTS="$initial_weights"

  trunk_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
    "$config_root/trunk_hungarian_seed$seed.yaml" "$trunk_run")"
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
  for method in "${methods[@]}"; do
    run="fgw_pilot_pretrained_continue_${method}_e15_seed$seed"
    bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
      "$config_root/continue_${method}_seed$seed.yaml" "$run"
  done
  echo "seed=$seed trunk_job=$trunk_job dependent_branches=3"
done

echo "Queue: squeue -u $USER"
