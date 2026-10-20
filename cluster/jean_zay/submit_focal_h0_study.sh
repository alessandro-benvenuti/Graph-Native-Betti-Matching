#!/usr/bin/env bash
# Prepare a new H0 study; never submit or modify the existing focal control.
set -euo pipefail
usage() {
  echo "Usage: $0 prepare|test|smoke|status|summary|h1-summary | screen|resume|h1-screen [WORKERS] | promote ENDPOINT IDS | extend ENDPOINT [WORKERS] | h1-prepare PARENT" >&2
  exit 2
}
[[ $# -ge 1 && $# -le 3 ]] || usage
action="$1"
export GNBM_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GNBM_REPO_DIR"
[[ -n "${WORK:-}" ]] || { echo "WORK is not set" >&2; exit 2; }
python_bin="${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing $python_bin" >&2; exit 2; }
if [[ "$action" == test ]]; then
  "$python_bin" -c 'import optuna; print("Optuna", optuna.__version__)'
  exec "$python_bin" -m unittest tests.test_focal_h0_study
fi
export GNBM_OUTPUT_DIR="${GNBM_OUTPUT_DIR:-/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-edge-focal-g05-h0-study-a100}"
source_run="${GNBM_H0_SOURCE_RUN:-/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-sweep-500-a100/finetune_boundary_mri500_node_edge_focal_g05_seed364505}"
reference_run="${GNBM_H0_REFERENCE_RUN:-/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-boundary-gamma-extension-750-a100/finetune_boundary_mri750_node_edge_focal_g05_tail250_seed364505}"
settings="${GNBM_H0_SETTINGS:-$GNBM_REPO_DIR/configs/experiments/focal_h0_study.yaml}"
export GNBM_H0_FAMILY="${GNBM_H0_FAMILY:-h0}"
case "$action" in
  prepare)
    exec "$python_bin" -m scripts.optimize_focal_h0 prepare --output "$GNBM_OUTPUT_DIR" \
      --settings "$settings" --source-run "$source_run" --reference-run "$reference_run" ;;
  summary|h1-summary|status)
    [[ "$action" != h1-summary ]] || export GNBM_H0_FAMILY=h1
    if [[ "$action" == status ]]; then
      squeue -u "$USER" -n gnbm-focal-h0 -o "%.18i %.2t %.10M %.30R"
    fi
    exec "$python_bin" -m scripts.optimize_focal_h0 summary --output "$GNBM_OUTPUT_DIR" --family "$GNBM_H0_FAMILY" ;;
  promote)
    [[ $# == 3 ]] || usage
    exec "$python_bin" -m scripts.optimize_focal_h0 promote --output "$GNBM_OUTPUT_DIR" \
      --family "$GNBM_H0_FAMILY" --endpoint "$2" --trials "$3" ;;
  h1-prepare)
    [[ $# == 2 ]] || usage
    exec "$python_bin" -m scripts.optimize_focal_h0 h1-prepare --output "$GNBM_OUTPUT_DIR" --parent "$2" ;;
  smoke|screen|resume|extend|h1-screen) ;;
  *) usage ;;
esac
[[ -f "$GNBM_OUTPUT_DIR/campaign.json" ]] || { echo "Run prepare first" >&2; exit 2; }
gpus="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["gpus"])' "$GNBM_OUTPUT_DIR/campaign.json")"
endpoint="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["endpoints"][0])' "$GNBM_OUTPUT_DIR/campaign.json")"
workers="${2:-2}"
if [[ "$action" == extend ]]; then
  [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || usage
  endpoint="$2"
  workers="${3:-2}"
  [[ -f "$GNBM_OUTPUT_DIR/promotions/$GNBM_H0_FAMILY/epoch_$(printf '%04d' "$endpoint").json" ]] || { echo "Run promote first" >&2; exit 2; }
fi
[[ "$action" != h1-screen ]] || export GNBM_H0_FAMILY=h1
export GNBM_H0_ACTION=worker GNBM_H0_ENDPOINT="$endpoint"
segments="${GNBM_H0_SEGMENTS:-6}"
walltime="${GNBM_WALLTIME:-06:00:00}"
qos="${GNBM_QOS:-qos_gpu_a100-t3}"
if [[ "$action" == smoke ]]; then
  workers=1
  segments=1
  walltime="${GNBM_SMOKE_WALLTIME:-02:00:00}"
  qos="${GNBM_SMOKE_QOS:-qos_gpu_a100-dev}"
  export GNBM_H0_ACTION=gpu-smoke
else
  [[ -f "$GNBM_OUTPUT_DIR/gpu-smoke/passed.json" ]] || { echo "Run and pass the GPU smoke first" >&2; exit 2; }
  "$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("passed") is True, "GPU smoke did not pass"' "$GNBM_OUTPUT_DIR/gpu-smoke/passed.json"
fi
[[ "$workers" =~ ^[1-4]$ && "$segments" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be 1..4 and SEGMENTS positive" >&2; exit 2; }
log_dir="$WORK/logs/graph-native-betti-matching/a100"
mkdir -p "$log_dir"
previous="${GNBM_H0_AFTER_JOB_ID:-}"
for ((segment=1; segment<=segments; segment++)); do
  dependencies=()
  [[ -z "$previous" ]] || dependencies=(--dependency="afterany:$previous")
  submission="$(sbatch "${dependencies[@]}" --array="0-$((workers-1))" \
    --gres="gpu:$gpus" --cpus-per-task="$((8*gpus))" --qos="$qos" --time="$walltime" \
    --chdir="$GNBM_REPO_DIR" --output="$log_dir/%x-%A_%a.out" --error="$log_dir/%x-%A_%a.err" \
    "$GNBM_REPO_DIR/cluster/jean_zay/focal_h0_study_a100.slurm")"
  echo "$submission (segment $segment/$segments)"
  previous="${submission##* }"
done
echo "Each array worker uses $gpus GPUs; workers=$workers; endpoint=$endpoint; family=$GNBM_H0_FAMILY"
if [[ "$action" == extend ]]; then
  echo "Resume this stage with: GNBM_H0_FAMILY=$GNBM_H0_FAMILY bash cluster/jean_zay/submit_focal_h0_study.sh extend $endpoint $workers"
elif [[ "$action" == smoke ]]; then
  echo "Repeat smoke after timeout if needed; inspect gpu-smoke/passed.json before screening."
elif [[ "$GNBM_H0_FAMILY" == h1 ]]; then
  echo "Resume later with: bash cluster/jean_zay/submit_focal_h0_study.sh h1-screen $workers"
else
  echo "Resume later with: bash cluster/jean_zay/submit_focal_h0_study.sh resume $workers"
fi
