#!/usr/bin/env bash
#SBATCH --job-name=gnbm-cached-logits
#SBATCH --account=vnc@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
set -eo pipefail
source "$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test/experiments/cached_logits/cluster/guard.sh"
source "$GNBM_REPO_DIR/cluster/jean_zay/env_a100.sh"
set -u
[[ "$PWD" == "$GNBM_REPO_DIR" ]] || exit 2
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -c 'import torch; from experiments.cached_logits.core import Objective; from models import build_model; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'
case "${CACHED_LOGIT_STAGE:-smoke}" in
  smoke)
    python -m experiments.cached_logits select --count 2 --output "$CACHED_LOGIT_ROOT/smoke-manifest.json"
    python -m experiments.cached_logits cache --manifest "$CACHED_LOGIT_ROOT/smoke-manifest.json" --output "$CACHED_LOGIT_ROOT/smoke-cache"
    # Cache is closed before optimizer starts. Optimization does not import model/images.
    python -m experiments.cached_logits optimize --cache "$CACHED_LOGIT_ROOT/smoke-cache/predictions.pt" --output "$CACHED_LOGIT_ROOT/smoke-opt" --steps 20 --threads 1 --max-seconds 600
    python -m experiments.cached_logits verify-smoke --cache "$CACHED_LOGIT_ROOT/smoke-cache/predictions.pt" --run "$CACHED_LOGIT_ROOT/smoke-opt"
    ;;
  main)
    [[ -f "$CACHED_LOGIT_ROOT/smoke-opt/SMOKE_VERIFIED.json" ]] || { echo 'Smoke has not passed' >&2; exit 2; }
    python -m experiments.cached_logits select --count 64 --output "$CACHED_LOGIT_ROOT/main-pool-manifest.json"
    python -m experiments.cached_logits cache --manifest "$CACHED_LOGIT_ROOT/main-pool-manifest.json" --select-count 32 --output "$CACHED_LOGIT_ROOT/main-cache"
    ;;
  *) echo 'Stage must be smoke or main' >&2; exit 2 ;;
esac
