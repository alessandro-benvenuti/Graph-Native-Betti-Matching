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

pretrain_gpus="${GNBM_PRETRAIN_GPUS:-4}"
finetune_gpus="${GNBM_FINETUNE_GPUS:-2}"
for gpus in "$pretrain_gpus" "$finetune_gpus"; do
  case "$gpus" in
    1|2|4) ;;
    *) echo "Stage GPU counts must be 1, 2, or 4." >&2; exit 2 ;;
  esac
done

pretrain_qos="${GNBM_PRETRAIN_QOS:-qos_gpu_a100-t3}"
finetune_qos="${GNBM_FINETUNE_QOS:-qos_gpu_a100-t3}"
pretrain_walltime="${GNBM_PRETRAIN_WALLTIME:-20:00:00}"
finetune_walltime="${GNBM_FINETUNE_WALLTIME:-20:00:00}"
pretrain_segments="${GNBM_PRETRAIN_SEGMENTS:-1}"
finetune_segments="${GNBM_FINETUNE_SEGMENTS:-10}"
for segments in "$pretrain_segments" "$finetune_segments"; do
  if [[ ! "$segments" =~ ^[1-9][0-9]*$ ]]; then
    echo "Stage segment counts must be positive integers." >&2
    exit 2
  fi
done

validate_request() {
  local qos="$1"
  local walltime="$2"
  local stage="$3"
  case "$qos" in
    qos_gpu_a100-t3) ;;
    *) echo "$stage QoS must be qos_gpu_a100-t3: $qos" >&2; exit 2 ;;
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
  if [[ "$total_seconds" -gt 72000 ]]; then
    echo "$stage requests more than the 20-hour A100 t3 limit." >&2
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

venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
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
# This campaign belongs to the new project. Use the campaign-specific override
# deliberately instead of inheriting a stale WANDB_PROJECT from an old shell.
export WANDB_PROJECT="${GNBM_BOUNDARY_WANDB_PROJECT:-focal-loss}"
# Jean-Zay compute nodes cannot reach wandb.ai; synchronize from a login node.
export WANDB_MODE=offline

printf '%-24s %-24s %-24s\n' RECIPE PRETRAIN_CHAIN FINETUNE_CHAIN
printf '%-24s %-24s %-24s\n' ------------------------ ------------------------ ------------------------

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  pretrain_config="$config_root/pretrain_${recipe}.yaml"
  finetune_config="$config_root/finetune_${recipe}.yaml"
  pretrain_run="${pretrain_runs[$index]}"
  finetune_run="${finetune_runs[$index]}"

  if [[ "$dry_run" == "1" ]]; then
    printf '%-24s %-24s %-24s\n' "$recipe" \
      "dry-pretrain-x$pretrain_segments" "dry-finetune-x$finetune_segments"
    echo "  pretrain: $pretrain_config -> $pretrain_run ($pretrain_segments segments)"
    echo "  finetune: $finetune_config -> $finetune_run ($finetune_segments segments)"
    continue
  fi

  export GNBM_GPUS="$pretrain_gpus"
  export GNBM_BATCH_SIZE="$((32 / pretrain_gpus))"
  export GNBM_QOS="$pretrain_qos"
  export GNBM_WALLTIME="$pretrain_walltime"
  unset GNBM_INITIAL_WEIGHTS GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME \
    GNBM_DEPENDENCY GNBM_ALLOW_PENDING_INITIAL_WEIGHTS

  pretrain_first=""
  pretrain_job=""
  for ((segment = 1; segment <= pretrain_segments; segment++)); do
    if (( segment > 1 )); then
      export GNBM_DEPENDENCY="afterany:$pretrain_job"
      export GNBM_AUTO_RESUME=1
    fi
    submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
      "$pretrain_config" "$pretrain_run")"
    next_job="$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
    if [[ -z "$next_job" ]]; then
      echo "Could not parse pretraining segment $segment job ID for $recipe:" >&2
      printf '%s\n' "$submission" >&2
      exit 1
    fi
    pretrain_job="$next_job"
    pretrain_first="${pretrain_first:-$pretrain_job}"
  done

  initial_checkpoint="$GNBM_OUTPUT_DIR/$pretrain_run/models/best_metric_checkpoint.pt"
  export GNBM_GPUS="$finetune_gpus"
  export GNBM_BATCH_SIZE="$((32 / finetune_gpus))"
  export GNBM_QOS="$finetune_qos"
  export GNBM_WALLTIME="$finetune_walltime"
  export GNBM_INITIAL_WEIGHTS="$initial_checkpoint"
  export GNBM_ALLOW_PENDING_INITIAL_WEIGHTS=1
  export GNBM_DEPENDENCY="afterok:$pretrain_job"
  unset GNBM_RESUME_CHECKPOINT GNBM_AUTO_RESUME

  finetune_first=""
  finetune_job=""
  for ((segment = 1; segment <= finetune_segments; segment++)); do
    if (( segment > 1 )); then
      export GNBM_DEPENDENCY="afterany:$finetune_job"
      export GNBM_AUTO_RESUME=1
    fi
    submission="$(bash "$repo_dir/cluster/jean_zay/submit_train_a100.sh" \
      "$finetune_config" "$finetune_run")"
    next_job="$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
    if [[ -z "$next_job" ]]; then
      echo "Could not parse fine-tuning segment $segment job ID for $recipe:" >&2
      printf '%s\n' "$submission" >&2
      exit 1
    fi
    finetune_job="$next_job"
    finetune_first="${finetune_first:-$finetune_job}"
  done

  printf '%-24s %-24s %-24s\n' "$recipe" \
    "$pretrain_first..$pretrain_job" "$finetune_first..$finetune_job"
done

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run complete; no jobs were submitted."
else
  echo "Every stage is a resumable Slurm chain; TIMEOUT advances to the next segment."
  echo "Completed runs create training-complete, so surplus segments exit immediately."
  echo "W&B: project=${WANDB_PROJECT:-focal-loss} group=$WANDB_RUN_GROUP mode=$WANDB_MODE"
  echo "Queue: squeue -u $USER"
  echo "Logs:  $log_dir"
fi
