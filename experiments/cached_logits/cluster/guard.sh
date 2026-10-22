#!/usr/bin/env bash
# Compute-node checks use the immutable snapshot captured on the login node.
: "${WORK:?}"
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
export CACHED_LOGIT_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
export WANDB_MODE=disabled
[[ -d "$GNBM_REPO_DIR/.git" && -f "$GNBM_REPO_DIR/.git/cached-logit-disposable" ]] || { echo 'Not the independent disposable clone' >&2; exit 2; }
[[ -n "${GNBM_CODE_PROVENANCE_FILE:-}" && -f "$GNBM_CODE_PROVENANCE_FILE" ]] || { echo 'Missing code provenance snapshot; submit on the login node with bash experiments/cached_logits/cluster/submit.sh smoke' >&2; exit 2; }
export MPLCONFIGDIR="$CACHED_LOGIT_ROOT/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
