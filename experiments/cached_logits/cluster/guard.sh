#!/usr/bin/env bash
# Export the isolated paths before module setup; verify Git again after purges.
: "${WORK:?}"
export GNBM_REPO_DIR="$WORK/projects/Graph-Native-Betti-Matching-cached-logit-test"
export CACHED_LOGIT_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/checkpoints/gnbm-cached-logit-betti-test-a100
export WANDB_MODE=disabled
source "${BASH_SOURCE[0]%/*}/git_env.sh"
cached_logit_require_git || exit 2
[[ -d "$GNBM_REPO_DIR/.git" && -f "$GNBM_REPO_DIR/.git/cached-logit-disposable" ]] || { echo 'Not the independent disposable clone' >&2; exit 2; }
cached_logit_branch="$(git -C "$GNBM_REPO_DIR" branch --show-current)" || { echo 'Cannot read disposable branch' >&2; exit 2; }
[[ "$cached_logit_branch" == codex/cached-logit-betti-test ]] || { echo 'Wrong disposable branch' >&2; exit 2; }
cached_logit_status="$(git -C "$GNBM_REPO_DIR" status --porcelain)" || { echo 'Cannot inspect disposable checkout' >&2; exit 2; }
[[ -z "$cached_logit_status" ]] || { echo 'Commit changes before caching' >&2; exit 2; }
unset cached_logit_branch cached_logit_status
export MPLCONFIGDIR="$CACHED_LOGIT_ROOT/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
