#!/usr/bin/env bash
# Submit seven mixed-pretraining jobs and seven dependent MRI specializations.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
for name in WORK SCRATCH PLANTS_DATASET SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done
if [[ ! -d "$PLANTS_DATASET" ]]; then
  echo "Plants dataset does not exist: $PLANTS_DATASET" >&2
  exit 2
fi
if [[ ! -d "$SYNTHETIC_MRI_DATASET" ]]; then
  echo "Synthetic-MRI dataset does not exist: $SYNTHETIC_MRI_DATASET" >&2
  exit 2
fi

pretrain_qos="${GNBM_PRETRAIN_QOS:-qos_gpu_h100-t3}"
pretrain_walltime="${GNBM_PRETRAIN_WALLTIME:-20:00:00}"
finetune_qos="${GNBM_FINETUNE_QOS:-qos_gpu_h100-t3}"
finetune_walltime="${GNBM_FINETUNE_WALLTIME:-20:00:00}"
dry_run="${GNBM_MATRIX_DRY_RUN:-0}"

validate_request() {
  local qos="$1"
  local walltime="$2"
  local stage="$3"
  case "$qos" in
    qos_gpu_h100-t3|qos_gpu_h100-t4) ;;
    *)
      echo "$stage QoS must be qos_gpu_h100-t3 or qos_gpu_h100-t4: $qos" >&2
      exit 2
      ;;
  esac
  if [[ ! "$walltime" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "$stage walltime must use HH:MM:SS: $walltime" >&2
    exit 2
  fi
  local hours="${walltime%%:*}"
  local remainder="${walltime#*:}"
  local minutes="${remainder%%:*}"
  local seconds="${remainder##*:}"
  local total_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
  if [[ "$qos" == "qos_gpu_h100-t3" && "$total_seconds" -gt 72000 ]]; then
    echo "$stage requests more than the 20-hour t3 limit; use t4." >&2
    exit 2
  fi
  if [[ "$qos" == "qos_gpu_h100-t4" && "$total_seconds" -gt 360000 ]]; then
    echo "$stage requests more than the 100-hour t4 limit." >&2
    exit 2
  fi
}

validate_request "$pretrain_qos" "$pretrain_walltime" pretraining
validate_request "$finetune_qos" "$finetune_walltime" finetuning

recipes=(
  baseline
  node_immediate
  node_curriculum
  edge_immediate
  edge_curriculum
  combined_immediate
  combined_curriculum
)

pretrain_runs=(
  pretrain_mixed_baseline_seed364505
  pretrain_mixed_node_focal_immediate_unweighted_seed364505
  pretrain_mixed_node_focal_curriculum_unweighted_seed364505
  pretrain_mixed_edge_focal_immediate_unweighted_seed364505
  pretrain_mixed_edge_focal_curriculum_unweighted_seed364505
  pretrain_mixed_combined_focal_immediate_unweighted_seed364505
  pretrain_mixed_combined_focal_curriculum_unweighted_seed364505
)

finetune_runs=(
  finetune_mri600_baseline_seed364505
  finetune_mri600_node_focal_immediate_unweighted_seed364505
  finetune_mri600_node_focal_curriculum_unweighted_seed364505
  finetune_mri600_edge_focal_immediate_unweighted_seed364505
  finetune_mri600_edge_focal_curriculum_unweighted_seed364505
  finetune_mri600_combined_focal_immediate_unweighted_seed364505
  finetune_mri600_combined_focal_curriculum_unweighted_seed364505
)

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  for stage in pretrain finetune; do
    config="$repo_dir/configs/experiments/focal_matrix_600/${stage}_${recipe}.yaml"
    if [[ ! -f "$config" ]]; then
      echo "Missing configuration: $config" >&2
      exit 2
    fi
  done
  for run_name in "${pretrain_runs[$index]}" "${finetune_runs[$index]}"; do
    if [[ -e "$GNBM_OUTPUT_DIR/$run_name" ]]; then
      echo "Refusing to reuse existing run directory: $GNBM_OUTPUT_DIR/$run_name" >&2
      exit 2
    fi
  done
done

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-focal-matrix-600-seed364505}"
# Jean Zay compute nodes cannot contact wandb.ai. Buffer each complete run for
# upload from a login node after training.
export WANDB_MODE=offline

submit_stage() {
  local stage="$1"
  local recipe="$2"
  local config="$3"
  local run_name="$4"
  local qos="$5"
  local walltime="$6"
  local dependency="${7:-}"
  local initial_weights="${8:-}"
  local job_name="gnbm-${stage}-${recipe}"

  if [[ "$dry_run" == "1" ]]; then
    echo "DRY-RUN $stage $recipe config=$config run=$run_name dependency=${dependency:-none}" >&2
    echo "dry-${stage}-${recipe}"
    return
  fi

  export GNBM_REPO_DIR="$repo_dir"
  export GNBM_CONFIG="$config"
  export GNBM_RUN_NAME="$run_name"
  unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT
  if [[ -n "$initial_weights" ]]; then
    export GNBM_INITIAL_WEIGHTS="$initial_weights"
  fi

  args=(
    --parsable
    --chdir="$repo_dir"
    --job-name="$job_name"
    --qos="$qos"
    --time="$walltime"
    --output="$log_dir/%x-%j.out"
    --error="$log_dir/%x-%j.err"
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  fi

  sbatch "${args[@]}" "$repo_dir/cluster/jean_zay/train_h100.slurm"
}

printf '%-22s %-14s %-14s\n' RECIPE PRETRAIN_JOB FINETUNE_JOB
printf '%-22s %-14s %-14s\n' ---------------------- -------------- --------------

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  pretrain_config="configs/experiments/focal_matrix_600/pretrain_${recipe}.yaml"
  finetune_config="configs/experiments/focal_matrix_600/finetune_${recipe}.yaml"
  pretrain_run="${pretrain_runs[$index]}"
  finetune_run="${finetune_runs[$index]}"

  pretrain_job="$(submit_stage \
    pre "$recipe" "$pretrain_config" "$pretrain_run" \
    "$pretrain_qos" "$pretrain_walltime")"
  initial_checkpoint="$GNBM_OUTPUT_DIR/$pretrain_run/models/best_checkpoint.pt"
  finetune_job="$(submit_stage \
    ft "$recipe" "$finetune_config" "$finetune_run" \
    "$finetune_qos" "$finetune_walltime" \
    "$pretrain_job" "$initial_checkpoint")"

  printf '%-22s %-14s %-14s\n' "$recipe" "$pretrain_job" "$finetune_job"
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete; no jobs were submitted."
else
  echo
  echo "All finetuning jobs depend on successful completion of their matching pretraining job."
  echo "Queue: squeue -u $USER"
  echo "Logs:  $log_dir/gnbm-{pre,ft}-*-JOB_ID.{out,err}"
fi
