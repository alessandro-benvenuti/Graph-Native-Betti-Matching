#!/usr/bin/env bash
# Read-only post-analysis on the login node. No sbatch, model, or optimizer.
set -Eeo pipefail
: "${WORK:?WORK is required}"
repo_dir="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
root=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
campaign=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-edge-focal-g05-h0-study-a100
output="${1:-$root/main-analysis}"
[[ ! -e "$output" ]] || { echo "Analysis output already exists: $output" >&2; exit 2; }
[[ -f "$root/main-opt/COMPLETE.json" && -f "$root/main-cache/CACHE_COMPLETE.json" ]] || {
  echo 'Completed existing main run and cache are required' >&2; exit 2;
}
[[ -f "$campaign/runs/h1/trial_0000/resolved-config.yaml" && -f "$campaign/h1-plan.json" ]] || {
  echo 'The specified current campaign reference and frozen H1 plan are required' >&2; exit 2;
}
module purge
module load pytorch-gpu/py3/2.3.0
source "${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/activate"
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
export MPLCONFIGDIR="$output/.matplotlib"
options=(--root "$root" --output "$output" --campaign-config "$campaign/runs/h1/trial_0000/resolved-config.yaml" --h1-plan "$campaign/h1-plan.json")
if [[ -f "$campaign/runs/h1/trial_0002/resolved-config.yaml" ]]; then
  options+=(--weight-match-config "$campaign/runs/h1/trial_0002/resolved-config.yaml")
fi
python "$repo_dir/experiments/cached_logits/analyze_saved.py" "${options[@]}"
printf 'Read-only analysis report: %s/report.md\n' "$output"
