#!/usr/bin/env bash
# Submit five boundary-data mixed-pretraining/MRI-finetuning pipelines.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SCRATCH PLANTS_DATASET SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is not set." >&2
    exit 2
  fi
done

dry_run="${GNBM_BOUNDARY_SWEEP_DRY_RUN:-0}"
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "GNBM_BOUNDARY_SWEEP_DRY_RUN must be 0 or 1." >&2
  exit 2
fi
if [[ "$dry_run" == "0" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi
if [[ ! -d "$PLANTS_DATASET" ]]; then
  echo "Plants dataset does not exist: $PLANTS_DATASET" >&2
  exit 2
fi
for split in train val test; do
  for folder in raw seg vtp; do
    required="$SYNTHETIC_MRI_DATASET/$split/$folder"
    if [[ ! -d "$required" ]]; then
      echo "Invalid boundary MRI dataset; missing: $required" >&2
      exit 2
    fi
  done
done

pretrain_gpus="${GNBM_PRETRAIN_GPUS:-1}"
finetune_gpus="${GNBM_FINETUNE_GPUS:-4}"
for gpus in "$pretrain_gpus" "$finetune_gpus"; do
  case "$gpus" in
    1|2|4) ;;
    *) echo "Stage GPU counts must be 1, 2, or 4." >&2; exit 2 ;;
  esac
done

pretrain_qos="${GNBM_PRETRAIN_QOS:-qos_gpu_h100-t4}"
finetune_qos="${GNBM_FINETUNE_QOS:-qos_gpu_h100-t4}"
pretrain_walltime="${GNBM_PRETRAIN_WALLTIME:-100:00:00}"
finetune_walltime="${GNBM_FINETUNE_WALLTIME:-100:00:00}"

validate_request() {
  local qos="$1"
  local walltime="$2"
  local stage="$3"
  case "$qos" in
    qos_gpu_h100-t3|qos_gpu_h100-t4) ;;
    *) echo "$stage QoS must be qos_gpu_h100-t3 or qos_gpu_h100-t4: $qos" >&2; exit 2 ;;
  esac
  if [[ ! "$walltime" =~ ^[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "$stage walltime must use HH:MM:SS: $walltime" >&2
    exit 2
  fi
  local hours="${walltime%%:*}"
  local rest="${walltime#*:}"
  local minutes="${rest%%:*}"
  local seconds="${rest##*:}"
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

config_root="configs/experiments/boundary_gamma_sweep_500"
recipes=(baseline node_focal node_edge_focal_g05 node_edge_focal_g10 node_edge_focal_g20)
pretrain_runs=(
  pretrain_boundary_mixed_baseline_seed364505
  pretrain_boundary_mixed_node_focal_seed364505
  pretrain_boundary_mixed_node_edge_focal_g05_seed364505
  pretrain_boundary_mixed_node_edge_focal_g10_seed364505
  pretrain_boundary_mixed_node_edge_focal_g20_seed364505
)
finetune_runs=(
  finetune_boundary_mri500_baseline_seed364505
  finetune_boundary_mri500_node_focal_seed364505
  finetune_boundary_mri500_node_edge_focal_g05_seed364505
  finetune_boundary_mri500_node_edge_focal_g10_seed364505
  finetune_boundary_mri500_node_edge_focal_g20_seed364505
)

venv="${GNBM_VENV:-$WORK/venvs/vascular-graph-extraction-h100-torch231}"
python_bin="$venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Project Python is not executable: $python_bin" >&2
  exit 2
fi

for index in "${!recipes[@]}"; do
  for stage in pretrain finetune; do
    config="$config_root/${stage}_${recipes[$index]}.yaml"
    if [[ ! -f "$repo_dir/$config" ]]; then
      echo "Missing configuration: $config" >&2
      exit 2
    fi
    (
      cd "$repo_dir"
      "$python_bin" -c \
        'import sys; from configs import load_config, validate_config; validate_config(load_config(sys.argv[1]))' \
        "$config"
    )
  done
  for run_name in "${pretrain_runs[$index]}" "${finetune_runs[$index]}"; do
    if [[ -e "$GNBM_OUTPUT_DIR/$run_name" ]]; then
      echo "Run directory already exists; refusing to overwrite: $GNBM_OUTPUT_DIR/$run_name" >&2
      exit 2
    fi
  done
done

log_dir="$WORK/logs/graph-native-betti-matching"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-boundary-gamma-sweep-500-seed364505}"
# Jean-Zay compute nodes cannot reach wandb.ai; synchronize from a login node.
export WANDB_MODE=offline

printf '%-24s %-16s %-16s\n' RECIPE PRETRAIN_JOB FINETUNE_JOB
printf '%-24s %-16s %-16s\n' ------------------------ ---------------- ----------------

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  pretrain_config="$config_root/pretrain_${recipe}.yaml"
  finetune_config="$config_root/finetune_${recipe}.yaml"
  pretrain_run="${pretrain_runs[$index]}"
  finetune_run="${finetune_runs[$index]}"

  if [[ "$dry_run" == "1" ]]; then
    printf '%-24s %-16s %-16s\n' "$recipe" dry-pretrain dry-finetune
    echo "  pretrain: $pretrain_config -> $pretrain_run"
    echo "  finetune: $finetune_config -> $finetune_run"
    continue
  fi

  export GNBM_GPUS="$pretrain_gpus"
  export GNBM_BATCH_SIZE="$((32 / pretrain_gpus))"
  export GNBM_QOS="$pretrain_qos"
  export GNBM_WALLTIME="$pretrain_walltime"
  unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

  pretrain_submission="$(bash "$repo_dir/cluster/jean_zay/submit_train.sh" \
    "$pretrain_config" "$pretrain_run")"
  pretrain_job="$(printf '%s\n' "$pretrain_submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
  if [[ -z "$pretrain_job" ]]; then
    echo "Could not parse pretraining job ID for $recipe:" >&2
    printf '%s\n' "$pretrain_submission" >&2
    exit 1
  fi

  initial_checkpoint="$GNBM_OUTPUT_DIR/$pretrain_run/models/best_metric_checkpoint.pt"
  finetune_submission="$(sbatch \
    --dependency="afterok:$pretrain_job" \
    --kill-on-invalid-dep=yes \
    --chdir="$repo_dir" \
    --job-name="gnbm-ft-boundary-${recipe}" \
    --qos="$finetune_qos" \
    --time="$finetune_walltime" \
    --output="$log_dir/%x-%j.out" \
    --error="$log_dir/%x-%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --gres="gpu:$finetune_gpus" \
    --cpus-per-task="$((10 * finetune_gpus))" \
    --export=ALL,GNBM_CONFIG="$finetune_config",GNBM_RUN_NAME="$finetune_run",GNBM_GPUS="$finetune_gpus",GNBM_GPUS_PER_NODE="$finetune_gpus",GNBM_BATCH_SIZE="$((32 / finetune_gpus))",GNBM_WALLTIME="$finetune_walltime",GNBM_INITIAL_WEIGHTS="$initial_checkpoint" \
    "$repo_dir/cluster/jean_zay/train_h100.slurm")"
  finetune_job="${finetune_submission##* }"

  printf '%-24s %-16s %-16s\n' "$recipe" "$pretrain_job" "$finetune_job"
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete; no jobs were submitted."
else
  echo "Every fine-tuning job depends on successful completion of its matching pretraining job."
  echo "W&B: project=${WANDB_PROJECT:-focal-loss} group=$WANDB_RUN_GROUP mode=$WANDB_MODE"
  echo "Queue: squeue -u $USER"
  echo "Logs:  $log_dir"
fi
