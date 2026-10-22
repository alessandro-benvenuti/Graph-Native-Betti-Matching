#!/usr/bin/env bash
#SBATCH --job-name=gnbm-cached-logits-cpu
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
set -eo pipefail
source "$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test/experiments/cached_logits/cluster/guard.sh"
module purge
module load pytorch-gpu/py3/2.3.0
source "${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/activate"
set -u
cached_logit_require_git
cd "$GNBM_REPO_DIR"
[[ "$PWD" -ef "$GNBM_REPO_DIR" ]] || { echo 'Wrong disposable working directory' >&2; exit 2; }
[[ -f "$CACHED_LOGIT_ROOT/smoke-opt/SMOKE_VERIFIED.json" ]] || exit 2
case "${CACHED_LOGIT_CPU_STAGE:-benchmark}" in
  benchmark)
    python -m experiments.cached_logits benchmark --cache "$CACHED_LOGIT_ROOT/main-cache/predictions.pt" --output "$CACHED_LOGIT_ROOT/cpu-benchmark" --steps 5 --threads 1 --max-seconds 300
    ;;
  main|resume)
    python - "$CACHED_LOGIT_ROOT/cpu-benchmark/benchmark.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['complete'], 'Benchmark incomplete: inspect before proceeding'
assert r['projected_32_patch_200_step_seconds'] <= 3300, 'CPU estimate exceeds bounded one-hour job: inspect before changing resources'
PY
    options=()
    if [[ "${CACHED_LOGIT_CPU_STAGE}" == resume ]]; then options+=(--resume); fi
    python -m experiments.cached_logits optimize --cache "$CACHED_LOGIT_ROOT/main-cache/predictions.pt" --output "$CACHED_LOGIT_ROOT/main-opt" --steps 200 --threads 1 --max-seconds 3300 "${options[@]}"
    ;;
  *) exit 2 ;;
esac
