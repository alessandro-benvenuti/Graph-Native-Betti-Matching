#!/usr/bin/env bash
# Submit full-data mixed pretraining and its dependent MRI specialization.
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
      echo "Synthetic-MRI root is not the generated new_patches directory." >&2
      echo "Missing: $SYNTHETIC_MRI_DATASET/$split/$folder" >&2
      exit 2
    fi
  done
done

echo "Synthetic-MRI dataset: $SYNTHETIC_MRI_DATASET"
for split in train val test; do
  count="$(find "$SYNTHETIC_MRI_DATASET/$split/vtp" -maxdepth 1 -type f -name '*_graph.vtp' | wc -l)"
  printf '  %-5s %8d patches\n' "$split" "$count"
  if (( count == 0 )); then
    echo "No graph patches discovered for split: $split" >&2
    exit 2
  fi
done

gpus="${GNBM_GPUS:-4}"
case "$gpus" in
  1|2|4) ;;
  *) echo "GNBM_GPUS must be 1, 2, or 4 (one H100 node)." >&2; exit 2 ;;
esac
qos="${GNBM_QOS:-qos_gpu_h100-t4}"
walltime="${GNBM_WALLTIME:-48:00:00}"
group="${WANDB_RUN_GROUP:-full-data-node-focal-seed364505}"
pretrain_run="pretrain_full_mixed_node_focal_seed364505"
finetune_run="finetune_full_mri_node_focal_seed364505"
pretrain_config="configs/experiments/full_dataset_node_focal/pretrain.yaml"
finetune_config="configs/experiments/full_dataset_node_focal/finetune.yaml"
log_dir="$WORK/logs/graph-native-betti-matching"

if [[ -e "$GNBM_OUTPUT_DIR/$pretrain_run" || -e "$GNBM_OUTPUT_DIR/$finetune_run" ]]; then
  echo "A full-data run directory already exists; use explicit resume instead." >&2
  exit 2
fi

export GNBM_GPUS="$gpus"
# Keep the global batch fixed at 32 for a controlled optimizer comparison.
export GNBM_BATCH_SIZE="$((32 / gpus))"
export GNBM_QOS="$qos"
export GNBM_WALLTIME="$walltime"
export WANDB_RUN_GROUP="$group"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

pretrain_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train.sh" \
  "$pretrain_config" "$pretrain_run")"
pretrain_job="$(printf '%s\n' "$pretrain_submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
if [[ -z "$pretrain_job" ]]; then
  echo "Could not parse pretraining job ID:" >&2
  printf '%s\n' "$pretrain_submission" >&2
  exit 1
fi

pretrain_checkpoint="$GNBM_OUTPUT_DIR/$pretrain_run/models/best_metric_checkpoint.pt"
export GNBM_INITIAL_WEIGHTS="$pretrain_checkpoint"
finetune_submission="$(sbatch \
  --dependency="afterok:$pretrain_job" \
  --kill-on-invalid-dep=yes \
  --chdir="$repo_dir" \
  --job-name="gnbm-ft-full-node-focal" \
  --qos="$qos" \
  --time="$walltime" \
  --output="$log_dir/%x-%j.out" \
  --error="$log_dir/%x-%j.err" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gres="gpu:$gpus" \
  --cpus-per-task="$((10 * gpus))" \
  --export=ALL,GNBM_CONFIG="$finetune_config",GNBM_RUN_NAME="$finetune_run",GNBM_GPUS="$gpus",GNBM_GPUS_PER_NODE="$gpus" \
  "$repo_dir/cluster/jean_zay/train_h100.slurm")"

printf '%s\n' "$pretrain_submission"
printf '%s\n' "$finetune_submission"
echo "Pretraining: $pretrain_run (job $pretrain_job)"
echo "Finetuning:  $finetune_run (starts after successful pretraining)"
echo "Queue:       squeue -u $USER"
