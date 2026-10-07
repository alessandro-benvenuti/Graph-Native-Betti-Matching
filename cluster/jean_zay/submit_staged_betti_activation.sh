#!/usr/bin/env bash
# Submit and inspect the staged Betti activation campaign.
set -euo pipefail

usage() {
  echo "Usage: $0 prefix|status|summarize | prepare [BRANCH] | screen|resume BRANCH [WORKERS] | final|test BRANCH TRIAL" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 3 ]] || usage
action="$1"
branch="${2:-}"
value="${3:-}"
case "$action" in
  prefix|status|summarize) [[ -z "$branch" && -z "$value" ]] || usage ;;
  prepare) [[ -z "$value" && ( -z "$branch" || "$branch" =~ ^[0-9]+$ ) ]] || usage ;;
  screen|resume) [[ "$branch" =~ ^[0-9]+$ ]] || usage ;;
  final|test) [[ "$branch" =~ ^[0-9]+$ && "$value" =~ ^[0-9]+$ ]] || usage ;;
  *) usage ;;
esac

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR GNBM_INITIAL_WEIGHTS; do
  [[ -n "${!name:-}" ]] || { echo "$name is not set." >&2; exit 2; }
done
venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
[[ -f "$GNBM_INITIAL_WEIGHTS" ]] || { echo "Missing checkpoint: $GNBM_INITIAL_WEIGHTS" >&2; exit 2; }

prefix_config="configs/experiments/staged_betti_activation/shared_prefix.yaml"
screen_config="configs/experiments/staged_betti_activation/screening.yaml"
config="$screen_config"
[[ "$action" == "prefix" ]] && config="$prefix_config"
if [[ -n "${GNBM_STAGED_CONFIG_OVERRIDE:-}" ]]; then
  config="$GNBM_STAGED_CONFIG_OVERRIDE"
fi
[[ -f "$repo_dir/$config" ]] || { echo "Missing staged config: $repo_dir/$config" >&2; exit 2; }
if [[ "$action" == "prefix" && -z "${GNBM_STAGED_CONFIG_OVERRIDE:-}" ]]; then
  [[ -n "${GNBM_STAGED_SMOKE_OUTPUT:-}" ]] || {
    echo "GNBM_STAGED_SMOKE_OUTPUT must point to a passed staged smoke before the full prefix." >&2
    exit 2
  }
  smoke_marker="$GNBM_STAGED_SMOKE_OUTPUT/summaries/smoke-validation.json"
  [[ -f "$smoke_marker" ]] || { echo "Missing smoke validation: $smoke_marker" >&2; exit 2; }
  "$python_bin" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("passed") is True' "$smoke_marker" || {
    echo "Staged smoke or timeout-resumption validation has not passed." >&2
    exit 2
  }
fi
common=(--config "$config" --output "$GNBM_OUTPUT_DIR" --initial-weights "$GNBM_INITIAL_WEIGHTS")
cd "$repo_dir"
"$python_bin" scripts/staged_betti_activation.py preflight "${common[@]}"

case "$action" in
  prepare)
    prepare_args=()
    [[ -z "$branch" ]] || prepare_args=(--branch "$branch")
    exec "$python_bin" scripts/staged_betti_activation.py prepare "${common[@]}" "${prepare_args[@]}"
    ;;
  summarize)
    exec "$python_bin" scripts/staged_betti_activation.py summarize "${common[@]}"
    ;;
  status)
    squeue -u "$USER" -n gnbm-staged-betti \
      --format="%.18i %.2t %.10M %.20R" || true
    exec "$python_bin" scripts/staged_betti_activation.py status "${common[@]}"
    ;;
esac

workers=1
if [[ "$action" == "screen" || "$action" == "resume" ]]; then
  workers="${value:-4}"
  (( workers >= 1 && workers <= 4 )) || { echo "WORKERS must be 1..4" >&2; exit 2; }
fi

export GNBM_REPO_DIR="$repo_dir" GNBM_VENV="$venv" GNBM_STAGED_CONFIG="$config"
export GNBM_STAGED_BRANCH="$branch" GNBM_STAGED_RESUME=0
case "$action" in
  prefix) export GNBM_STAGED_ACTION=prefix ;;
  screen) export GNBM_STAGED_ACTION=screen ;;
  resume) export GNBM_STAGED_ACTION=screen GNBM_STAGED_RESUME=1 ;;
  final) export GNBM_STAGED_ACTION=final GNBM_STAGED_TRIAL="$value" ;;
  test) export GNBM_STAGED_ACTION=test GNBM_STAGED_TRIAL="$value" ;;
esac
export WANDB_PROJECT=gnbm WANDB_RUN_GROUP=staged-betti-activation
# Jean Zay compute nodes cannot reach wandb.ai.  Every segment writes an
# offline transaction with a stable run ID; sync_wandb_offline.sh appends the
# segments to the same cloud run from a login node.
export WANDB_MODE="${GNBM_STAGED_WANDB_MODE:-offline}"

gpus="${GNBM_STAGED_GPUS:-2}"
case "$gpus" in
  1|2|4) ;;
  *) echo "GNBM_STAGED_GPUS must be 1, 2, or 4." >&2; exit 2 ;;
esac
global_batch_size="${GNBM_STAGED_GLOBAL_BATCH_SIZE:-32}"
[[ "$global_batch_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "GNBM_STAGED_GLOBAL_BATCH_SIZE must be a positive integer." >&2
  exit 2
}
(( global_batch_size % gpus == 0 )) || {
  echo "Global batch size $global_batch_size is not divisible by $gpus GPUs." >&2
  exit 2
}
export GNBM_STAGED_GPUS="$gpus"
export GNBM_STAGED_GLOBAL_BATCH_SIZE="$global_batch_size"

log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
array_args=()
if [[ "$action" == "screen" || "$action" == "resume" ]]; then
  array_args=(--array="0-$((workers - 1))")
fi
default_qos="qos_gpu_a100-t3"
default_walltime="20:00:00"
if [[ "$config" == *"/smoke.yaml" || "$config" == "smoke.yaml" ]]; then
  default_qos="qos_gpu_a100-dev"
  default_walltime="02:00:00"
fi
if [[ "$config" == *"/smoke.yaml" || "$config" == "smoke.yaml" ]]; then
  default_segments=1
else
  case "$action" in
    prefix) default_segments=12 ;;
    screen|resume) default_segments=4 ;;
    final) default_segments=8 ;;
    test) default_segments=1 ;;
  esac
fi
segments="${GNBM_STAGED_CHAIN_SEGMENTS:-$default_segments}"
[[ "$segments" =~ ^[1-9][0-9]*$ ]] || {
  echo "GNBM_STAGED_CHAIN_SEGMENTS must be a positive integer." >&2
  exit 2
}
(( segments <= 32 )) || {
  echo "GNBM_STAGED_CHAIN_SEGMENTS must not exceed 32." >&2
  exit 2
}

previous_job="${GNBM_STAGED_AFTER_JOB_ID:-}"
job_ids=()
for ((segment = 1; segment <= segments; segment++)); do
  # A successor screening segment must recover stale RUNNING trials before it
  # allocates WAITING/new trials.  Prefix and final actions discover their own
  # latest checkpoint and completion marker automatically.
  if [[ "$action" == "screen" || "$action" == "resume" ]]; then
    if (( segment > 1 )) || [[ "$action" == "resume" ]]; then
      export GNBM_STAGED_RESUME=1
    else
      export GNBM_STAGED_RESUME=0
    fi
  fi
  dependency_args=()
  if [[ -n "$previous_job" ]]; then
    dependency_args=(--dependency="afterany:$previous_job")
  fi
  submission="$(sbatch "${array_args[@]}" "${dependency_args[@]}" \
    --qos="${GNBM_QOS:-$default_qos}" \
    --gres="gpu:$gpus" --cpus-per-task="$((8 * gpus))" \
    --time="${GNBM_WALLTIME:-$default_walltime}" --chdir="$repo_dir" \
    --output="$log_dir/%x-%A_%a.out" --error="$log_dir/%x-%A_%a.err" \
    "$repo_dir/cluster/jean_zay/staged_betti_activation_a100.slurm")"
  echo "$submission (segment $segment/$segments)"
  job_id="${submission##* }"
  job_ids+=("$job_id")
  previous_job="$job_id"
done

job_list="$(IFS=,; echo "${job_ids[*]}")"
echo "Chain: $job_list"
echo "Queue: squeue -j $job_list"
echo "Accounting: sacct -j $job_list --format=JobID,State,Elapsed,ExitCode,MaxRSS"
echo "Logs: $log_dir/gnbm-staged-betti-{${job_list}}_*.{out,err}"
echo "Training: gpus=$gpus global_batch=$global_batch_size per_gpu_batch=$((global_batch_size / gpus))"
echo "W&B: project=$WANDB_PROJECT group=$WANDB_RUN_GROUP mode=$WANDB_MODE"
