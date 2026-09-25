#!/usr/bin/env bash
# Submit/control/summarize the node/edge Betti Pareto campaigns.
set -euo pipefail

usage() { echo "Usage: $0 control|worker|array|resume|summarize|recover smoke|study-a|study-b [WORKERS]" >&2; exit 2; }
[[ $# -ge 2 && $# -le 3 ]] || usage
action="$1"; campaign="$2"; workers="${3:-1}"
case "$action" in control|worker|array|resume|summarize|recover) ;; *) usage ;; esac
case "$campaign" in
  smoke) config="configs/experiments/node_edge_betti_optuna/smoke.yaml"; study="node-edge-betti-pareto-smoke" ;;
  study-a) config="configs/experiments/node_edge_betti_optuna/study_a.yaml"; study="node-edge-betti-pareto-study-a" ;;
  study-b) config="configs/experiments/node_edge_betti_optuna/study_b_proposed.yaml"; study="node-edge-betti-pareto-study-b" ;;
  *) usage ;;
esac
if (( workers < 1 || workers > 4 )); then echo "WORKERS must be 1..4" >&2; exit 2; fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for name in WORK SYNTHETIC_MRI_DATASET GNBM_OUTPUT_DIR GNBM_INITIAL_WEIGHTS; do
  [[ -n "${!name:-}" ]] || { echo "$name is not set." >&2; exit 2; }
done
venv="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}"
python_bin="$venv/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
[[ -f "$GNBM_INITIAL_WEIGHTS" ]] || { echo "Missing checkpoint: $GNBM_INITIAL_WEIGHTS" >&2; exit 2; }
[[ -f "$repo_dir/$config" ]] || { echo "Missing config: $repo_dir/$config" >&2; exit 2; }
storage="sqlite"; [[ "$action" == "array" || "$workers" -gt 1 ]] && storage="journal"

common=(--config "$config" --output "$GNBM_OUTPUT_DIR" --initial-weights "$GNBM_INITIAL_WEIGHTS" --storage "$storage")
cd "$repo_dir"
"$python_bin" -c 'import optuna; print("optuna=" + optuna.__version__)'
export GNBM_OPTUNA_WORKERS="$workers"
"$python_bin" scripts/optimize_node_edge_betti.py preflight "${common[@]}"

if [[ "$action" == "summarize" ]]; then
  exec "$python_bin" scripts/summarize_node_edge_betti_optuna.py \
    --output "$GNBM_OUTPUT_DIR" --study-name "$study" --storage "$storage"
fi
if [[ "$action" == "recover" ]]; then
  echo "Run recovery only when no workers are active."
  exec "$python_bin" scripts/optimize_node_edge_betti.py recover "${common[@]}"
fi
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is unavailable" >&2; exit 2; }

export GNBM_REPO_DIR="$repo_dir" GNBM_VENV="$venv" GNBM_OPTUNA_CONFIG="$config"
export GNBM_OPTUNA_STUDY_NAME="$study" GNBM_OPTUNA_STORAGE="$storage"
if [[ "$action" == "array" || "$action" == "resume" ]]; then export GNBM_OPTUNA_ACTION="worker"
else export GNBM_OPTUNA_ACTION="$action"; fi
GNBM_OPTUNA_MAX_TRIALS="$($python_bin -c \
  'import sys; from pathlib import Path; from scripts.optimize_node_edge_betti import load_campaign; print(load_campaign(Path(sys.argv[1]))[1]["n_trials"])' "$config")"
export GNBM_OPTUNA_MAX_TRIALS
export WANDB_MODE=offline

if [[ "$campaign" == "smoke" ]]; then qos="${GNBM_QOS:-qos_gpu_a100-dev}"; walltime="${GNBM_WALLTIME:-02:00:00}"
else qos="${GNBM_QOS:-qos_gpu_a100-t3}"; walltime="${GNBM_WALLTIME:-20:00:00}"; fi
log_dir="$WORK/logs/graph-native-betti-matching/a100"; mkdir -p "$log_dir" "$GNBM_OUTPUT_DIR"
array_args=(); [[ "$action" == "array" || ( "$action" == "resume" && "$workers" -gt 1 ) ]] && array_args=(--array="0-$((workers - 1))")
submission="$(sbatch "${array_args[@]}" --qos="$qos" --time="$walltime" --chdir="$repo_dir" \
  --output="$log_dir/%x-%A_%a.out" --error="$log_dir/%x-%A_%a.err" \
  "$repo_dir/cluster/jean_zay/node_edge_betti_optuna_a100.slurm")"
echo "$submission"
job_id="${submission##* }"
echo "Queue: squeue -j $job_id"
echo "Accounting: sacct -j $job_id --format=JobID,State,Elapsed,ExitCode,MaxRSS"
echo "Logs: $log_dir/gnbm-betti-optuna-${job_id}_*.{out,err}"
