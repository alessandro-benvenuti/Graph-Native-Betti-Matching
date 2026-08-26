#!/usr/bin/env bash
# Submit baseline and node-focal + matched-edge-focal full-data pipelines.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SCRATCH PLANTS_DATASET SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done

for split in train val test; do
  for folder in raw seg vtp; do
    if [[ ! -d "$SYNTHETIC_MRI_DATASET/$split/$folder" ]]; then
      echo "Missing full synthetic-MRI path: $SYNTHETIC_MRI_DATASET/$split/$folder" >&2
      exit 2
    fi
  done
done

gpus="${GNBM_GPUS:-4}"
case "$gpus" in
  1|2|4) ;;
  *) echo "GNBM_GPUS must be 1, 2, or 4." >&2; exit 2 ;;
esac
qos="${GNBM_QOS:-qos_gpu_h100-t4}"
walltime="${GNBM_WALLTIME:-48:00:00}"
group="${WANDB_RUN_GROUP:-full-data-loss-comparison-seed364505}"
log_dir="$WORK/logs/graph-native-betti-matching"
config_root="configs/experiments/full_dataset_comparison"

export GNBM_GPUS="$gpus"
export GNBM_BATCH_SIZE="$((32 / gpus))"
export GNBM_QOS="$qos"
export GNBM_WALLTIME="$walltime"
export WANDB_RUN_GROUP="$group"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

recipes=(baseline nodefocal_edgefocal_mm)
pretrain_runs=(
  pretrain_full_mixed_baseline_seed364505
  pretrain_full_mixed_nodefocal_edgefocal_mm_seed364505
)
finetune_runs=(
  finetune_full_mri_baseline_seed364505
  finetune_full_mri_nodefocal_edgefocal_mm_seed364505
)

for index in "${!recipes[@]}"; do
  if [[ -e "$GNBM_OUTPUT_DIR/${pretrain_runs[$index]}" || -e "$GNBM_OUTPUT_DIR/${finetune_runs[$index]}" ]]; then
    echo "Run directory already exists for ${recipes[$index]}; refusing to submit a partial matrix." >&2
    exit 2
  fi
done

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  pretrain_run="${pretrain_runs[$index]}"
  finetune_run="${finetune_runs[$index]}"
  pretrain_config="$config_root/pretrain_${recipe}.yaml"
  finetune_config="$config_root/finetune_${recipe}.yaml"

  pretrain_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train.sh" \
    "$pretrain_config" "$pretrain_run")"
  pretrain_job="$(printf '%s\n' "$pretrain_submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
  if [[ -z "$pretrain_job" ]]; then
    echo "Could not parse pretraining job ID for $recipe:" >&2
    printf '%s\n' "$pretrain_submission" >&2
    exit 1
  fi

  pretrain_checkpoint="$GNBM_OUTPUT_DIR/$pretrain_run/models/best_metric_checkpoint.pt"
  finetune_submission="$(sbatch \
    --dependency="afterok:$pretrain_job" \
    --kill-on-invalid-dep=yes \
    --chdir="$repo_dir" \
    --job-name="gnbm-ft-full-${recipe}" \
    --qos="$qos" \
    --time="$walltime" \
    --output="$log_dir/%x-%j.out" \
    --error="$log_dir/%x-%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gres="gpu:$gpus" \
    --cpus-per-task="$((10 * gpus))" \
    --export=ALL,GNBM_CONFIG="$finetune_config",GNBM_RUN_NAME="$finetune_run",GNBM_GPUS="$gpus",GNBM_GPUS_PER_NODE="$gpus",GNBM_INITIAL_WEIGHTS="$pretrain_checkpoint" \
    "$repo_dir/cluster/jean_zay/train_h100.slurm")"

  printf '%s\n' "$pretrain_submission"
  printf '%s\n' "$finetune_submission"
  echo "$recipe: pretraining job $pretrain_job, followed by MRI specialization"
done

echo "Queue: squeue -u $USER"
