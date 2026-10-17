#!/usr/bin/env bash
# Continue three selected epoch-500 runs with a fresh 250-epoch LR tail.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SCRATCH SYNTHETIC_MRI_DATASET; do
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
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this on a Jean Zay login node." >&2
  exit 2
fi

dry_run="${GNBM_EXTENSION_DRY_RUN:-0}"
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "GNBM_EXTENSION_DRY_RUN must be 0 or 1." >&2
  exit 2
fi

hardware="${GNBM_EXTENSION_HARDWARE:-a100}"
case "$hardware" in
  a100)
    continuation_gpus=4
    batch_per_gpu=8
    default_output_root="$SCRATCH/checkpoints/gnbm-boundary-gamma-extension-750-a100"
    venv_dir="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
    python_bin="$venv_dir/bin/python"
    submit_helper="$repo_dir/cluster/jean_zay/submit_train_a100.sh"
    qos=qos_gpu_a100-t3
    default_walltime=20:00:00
    default_segments=6
    default_group=boundary-gamma-extension-750-seed364505
    ;;
  h100)
    continuation_gpus=2
    batch_per_gpu=16
    default_output_root="$SCRATCH/checkpoints/gnbm-boundary-gamma-extension-750-h100"
    venv_dir="${GNBM_H100_VENV:-$WORK/venvs/vascular-graph-extraction-h100-torch231}"
    python_bin="$venv_dir/bin/python"
    submit_helper="$repo_dir/cluster/jean_zay/submit_train.sh"
    qos=qos_gpu_h100-t3
    default_walltime=06:00:00
    default_segments=8
    default_group=boundary-gamma-extension-750-h100-seed364505
    ;;
  *)
    echo "GNBM_EXTENSION_HARDWARE must be a100 or h100." >&2
    exit 2
    ;;
esac

source_root="${GNBM_EXTENSION_SOURCE_ROOT:-$SCRATCH/checkpoints/gnbm-boundary-gamma-sweep-500-a100}"
output_root="${GNBM_EXTENSION_OUTPUT_DIR:-$default_output_root}"
segments="${GNBM_EXTENSION_SEGMENTS:-$default_segments}"
if [[ ! "$segments" =~ ^[1-9][0-9]*$ ]]; then
  echo "GNBM_EXTENSION_SEGMENTS must be a positive integer." >&2
  exit 2
fi

config_root="configs/experiments/boundary_gamma_extension_750"
recipes=(baseline node_focal node_edge_focal_g05)
source_runs=(
  finetune_boundary_mri500_baseline_seed364505
  finetune_boundary_mri500_node_focal_seed364505
  finetune_boundary_mri500_node_edge_focal_g05_seed364505
)
destination_runs=(
  finetune_boundary_mri750_baseline_tail250_seed364505
  finetune_boundary_mri750_node_focal_tail250_seed364505
  finetune_boundary_mri750_node_edge_focal_g05_tail250_seed364505
)

if [[ ! -x "$python_bin" ]]; then
  echo "The $hardware Python environment is missing: $python_bin" >&2
  exit 2
fi

for index in "${!recipes[@]}"; do
  source_run="$source_root/${source_runs[$index]}"
  destination_run="$output_root/${destination_runs[$index]}"
  config="$config_root/${recipes[$index]}.yaml"
  for path in \
    "$source_run/training-status.json" \
    "$source_run/models/latest_checkpoint.pt" \
    "$source_run/models/best_metric_checkpoint.pt" \
    "$source_run/models/best_node_f1_checkpoint.pt" \
    "$source_run/models/best_edge_f1_checkpoint.pt" \
    "$source_run/best-metric.json" \
    "$source_run/best-node-f1.json" \
    "$source_run/best-edge-f1.json" \
    "$source_run/performance.jsonl" \
    "$source_run/validation-metrics.jsonl" \
    "$source_run/canonical-wandb-history.jsonl" \
    "$source_run/canonical-wandb-manifest.json"; do
    if [[ ! -s "$path" ]]; then
      echo "Missing or empty source artifact: $path" >&2
      exit 2
    fi
  done
  if [[ ! -f "$source_run/training-complete" ]]; then
    echo \
      "Warning: completion marker is missing for ${source_runs[$index]}; validating status and checkpoint instead." \
      >&2
  fi
  "$python_bin" -c '
import inspect, json, sys, torch
status = json.load(open(sys.argv[1], encoding="utf-8"))
options = {"map_location": "cpu"}
if "weights_only" in inspect.signature(torch.load).parameters:
    options["weights_only"] = False
checkpoint = torch.load(sys.argv[2], **options)
history = [
    json.loads(line)
    for line in open(sys.argv[3], encoding="utf-8")
    if line.strip()
]
if status.get("reason") != "max_epochs" or int(status.get("epoch", -1)) != 500:
    raise SystemExit("source run is not a completed 500-epoch run: " + repr(status))
if int(checkpoint.get("epoch", -1)) != 500:
    raise SystemExit("latest checkpoint is not epoch 500")
if int((checkpoint.get("trainer_state") or {}).get("world_size", -1)) != 2:
    raise SystemExit("source checkpoint was not trained with world size 2")
iteration = int(checkpoint.get("iteration", checkpoint.get("global_step", -1)))
if iteration <= 0 or iteration % 500:
    raise SystemExit("source checkpoint has an invalid iteration count")
iterations_per_epoch = iteration // 500
epochs = [int(row.get("epoch", -1)) for row in history]
if epochs != list(range(1, 501)):
    raise SystemExit("canonical W&B history is not contiguous for epochs 1..500")
for row in history:
    epoch = int(row["epoch"])
    for key in ("train/epoch", "performance/epoch"):
        if int(row.get(key, -1)) != epoch:
            raise SystemExit(f"canonical W&B history has invalid {key} at epoch {epoch}")
    if int(row.get("train/iteration", -1)) != epoch * iterations_per_epoch:
        raise SystemExit(f"canonical W&B history has invalid iteration at epoch {epoch}")
' "$source_run/training-status.json" "$source_run/models/latest_checkpoint.pt" \
  "$source_run/canonical-wandb-history.jsonl"
  if [[ -e "$destination_run" ]]; then
    echo "Destination already exists; refusing to overwrite: $destination_run" >&2
    exit 2
  fi
  export GNBM_EXTENSION_SOURCE_RUN="$source_run"
  (
    cd "$repo_dir"
    "$python_bin" -c \
      'import sys; from configs import load_config, validate_config; validate_config(load_config(sys.argv[1]))' \
      "$config"
  )
done

echo "Source:      $source_root"
echo "Destination: $output_root"
echo "Hardware:    $continuation_gpus ${hardware^^}(s) (batch $batch_per_gpu/GPU, global batch 32)"
echo "Schedule:    epoch 500 -> at most 750, fresh polynomial LR 1e-5 -> 0"
echo "Stopping:    reset edge-mAP patience, 50 epochs"
echo "Segments:    $segments x $default_walltime per recipe"

if [[ "$dry_run" == "1" ]]; then
  echo "Dry run passed; no directories or jobs were created."
  exit 0
fi

mkdir -p "$output_root"

seed_destination() {
  local source_run="$1"
  local destination_run="$2"
  mkdir -p "$destination_run/models"
  local name
  for name in \
    best_metric_checkpoint.pt \
    best_node_f1_checkpoint.pt \
    best_edge_f1_checkpoint.pt; do
    if ! ln "$source_run/models/$name" "$destination_run/models/$name" 2>/dev/null; then
      cp -p "$source_run/models/$name" "$destination_run/models/$name"
    fi
  done
  if [[ -s "$source_run/models/best_checkpoint.pt" ]]; then
    if ! ln "$source_run/models/best_checkpoint.pt" \
      "$destination_run/models/best_checkpoint.pt" 2>/dev/null; then
      cp -p "$source_run/models/best_checkpoint.pt" \
        "$destination_run/models/best_checkpoint.pt"
    fi
  fi
  cp -p \
    "$source_run/best-metric.json" \
    "$source_run/best-node-f1.json" \
    "$source_run/best-edge-f1.json" \
    "$source_run/performance.jsonl" \
    "$source_run/validation-metrics.jsonl" \
    "$destination_run/"
  "$python_bin" -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[2]) / "extension-source.json"
path.write_text(json.dumps({
    "source_run": sys.argv[1],
    "resume_checkpoint": sys.argv[1] + "/models/latest_checkpoint.pt",
    "source_epoch": 500,
    "target_epoch": 750,
    "scheduler_restart_base_lr": 1.0e-5,
    "scheduler_duration_epochs": 250,
    "early_stopping_reset": True,
    "source_world_size": 2,
    "continuation_world_size": int(sys.argv[4]),
    "continuation_hardware": sys.argv[3],
    "global_batch_size": 32,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$source_run" "$destination_run" "$hardware" "$continuation_gpus"
}

export GNBM_OUTPUT_DIR="$output_root"
if [[ "$hardware" == "h100" ]]; then
  export GNBM_VENV="$venv_dir"
fi
export GNBM_GPUS="$continuation_gpus"
export GNBM_BATCH_SIZE="$batch_per_gpu"
export GNBM_QOS="$qos"
export GNBM_WALLTIME="${GNBM_EXTENSION_WALLTIME:-$default_walltime}"
export GNBM_SKIP_PREFLIGHT=1
export WANDB_MODE=offline
export WANDB_PROJECT="${GNBM_EXTENSION_WANDB_PROJECT:-focal-loss}"
export WANDB_RUN_GROUP="${GNBM_EXTENSION_WANDB_GROUP:-$default_group}"
unset GNBM_INITIAL_WEIGHTS

printf '%-24s %-28s\n' RECIPE JOB_CHAIN
printf '%-24s %-28s\n' ------------------------ ----------------------------

for index in "${!recipes[@]}"; do
  recipe="${recipes[$index]}"
  source_run="$source_root/${source_runs[$index]}"
  destination_name="${destination_runs[$index]}"
  destination_run="$output_root/$destination_name"
  config="$config_root/$recipe.yaml"
  seed_destination "$source_run" "$destination_run"
  export GNBM_EXTENSION_SOURCE_RUN="$source_run"

  first_job=""
  previous_job=""
  for ((segment = 1; segment <= segments; segment++)); do
    if (( segment == 1 )); then
      export GNBM_RESUME_CHECKPOINT="$source_run/models/latest_checkpoint.pt"
      unset GNBM_AUTO_RESUME GNBM_DEPENDENCY GNBM_ALLOW_PENDING_RESUME
      export GNBM_RESET_SCHEDULER_ON_RESUME=1
      export GNBM_RESET_EARLY_STOPPING_ON_RESUME=1
      if (( continuation_gpus != 2 )); then
        export GNBM_ALLOW_WORLD_SIZE_CHANGE_ON_RESUME=1
      else
        unset GNBM_ALLOW_WORLD_SIZE_CHANGE_ON_RESUME
      fi
    else
      export GNBM_RESUME_CHECKPOINT="$destination_run/models/latest_checkpoint.pt"
      export GNBM_ALLOW_PENDING_RESUME=1
      export GNBM_DEPENDENCY="afterany:$previous_job"
      unset GNBM_AUTO_RESUME
      unset GNBM_RESET_SCHEDULER_ON_RESUME
      unset GNBM_RESET_EARLY_STOPPING_ON_RESUME
      unset GNBM_ALLOW_WORLD_SIZE_CHANGE_ON_RESUME
    fi
    submission="$(bash "$submit_helper" \
      "$config" "$destination_name")"
    job_id="$(printf '%s\n' "$submission" | sed -n 's/^Submitted batch job //p' | head -n 1)"
    if [[ -z "$job_id" ]]; then
      echo "Could not parse job ID for $recipe segment $segment:" >&2
      printf '%s\n' "$submission" >&2
      exit 1
    fi
    previous_job="$job_id"
    first_job="${first_job:-$job_id}"
  done
  printf '%-24s %s..%s\n' "$recipe" "$first_job" "$previous_job"
done

echo "Queue: squeue -u $USER --format='%.18i %.55j %.2t %.10M %.30R'"
echo "Logs:  $WORK/logs/graph-native-betti-matching"
echo "The first segment resets the scheduler and patience; later segments strictly resume the extension state."
