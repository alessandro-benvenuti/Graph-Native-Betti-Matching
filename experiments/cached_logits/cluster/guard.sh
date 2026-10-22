#!/usr/bin/env bash
# Source before module setup. A100 env script must see this exact export.
: "${WORK:?}"
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
export CACHED_LOGIT_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
export WANDB_MODE=disabled
[[ -d "$GNBM_REPO_DIR/.git" && -f "$GNBM_REPO_DIR/.git/cached-logit-disposable" ]] || { echo 'Not the independent disposable clone' >&2; exit 2; }
[[ "$(git -C "$GNBM_REPO_DIR" branch --show-current)" == codex/cached-logit-betti-test ]] || exit 2
[[ -z "$(git -C "$GNBM_REPO_DIR" status --porcelain)" ]] || { echo 'Commit changes before caching' >&2; exit 2; }
export MPLCONFIGDIR="$CACHED_LOGIT_ROOT/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
