#!/usr/bin/env bash
#SBATCH --job-name=gnbm-cached-logits-cpu
#SBATCH --account=vnc@cpu
#SBATCH --qos=qos_cpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
set -Eeo pipefail
trap 'rc=$?; echo "Cached-logit job failed at line $LINENO (exit $rc)" >&2' ERR
echo "Cached-logit phase: isolated path checks"
source "$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test/experiments/cached_logits/cluster/guard.sh"
module purge
module load pytorch-gpu/py3/2.3.0
source "${GNBM_A100_VENV:-$WORK/venvs/vascular-graph-extraction-a100-torch230}/bin/activate"
set -u
cd "$GNBM_REPO_DIR"
[[ "$PWD" -ef "$GNBM_REPO_DIR" ]] || { echo 'Wrong disposable working directory' >&2; exit 2; }
echo "Cached-logit phase: verify login-node code snapshot (no Git)"
python "$GNBM_REPO_DIR/experiments/cached_logits/provenance.py" verify \
  --repo "$GNBM_REPO_DIR" --manifest "$GNBM_CODE_PROVENANCE_FILE"
[[ -f "$CACHED_LOGIT_ROOT/smoke-opt/SMOKE_VERIFIED.json" ]] || exit 2
case "${CACHED_LOGIT_CPU_STAGE:-benchmark}" in
  benchmark)
    python -m experiments.cached_logits benchmark --cache "$CACHED_LOGIT_ROOT/main-cache/predictions.pt" --output "$CACHED_LOGIT_ROOT/cpu-benchmark" --steps 5 --threads 1 --max-seconds 300
    ;;
  main|resume)
    cpu_budget="${CACHED_LOGIT_CPU_BUDGET_SECONDS:-3300}"
    python "$GNBM_REPO_DIR/experiments/cached_logits/runtime.py" \
      --benchmark "$CACHED_LOGIT_ROOT/cpu-benchmark/benchmark.json" --budget "$cpu_budget"
    options=(--steps 200 --threads 1 --max-seconds "$cpu_budget")
    if [[ "${CACHED_LOGIT_CPU_STAGE}" == resume ]]; then options+=(--resume); fi
    python -m experiments.cached_logits optimize --cache "$CACHED_LOGIT_ROOT/main-cache/predictions.pt" --output "$CACHED_LOGIT_ROOT/main-opt" "${options[@]}"
    ;;
  *) exit 2 ;;
esac
