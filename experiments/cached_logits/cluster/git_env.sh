#!/usr/bin/env bash
# Git may be a module on compute nodes. Recheck after any module purge.
cached_logit_require_git() {
  if ! command -v git >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
      echo 'Git is unavailable and the module command is not initialized.' >&2
      return 2
    fi
    if ! module load "${CACHED_LOGIT_GIT_MODULE:-git}"; then
      echo 'Cannot load Git. Check module avail git and set CACHED_LOGIT_GIT_MODULE if a version is required.' >&2
      return 2
    fi
  fi
  if ! command -v git >/dev/null 2>&1 || ! git --version >/dev/null 2>&1; then
    echo 'Git is still unavailable after environment setup.' >&2
    return 2
  fi
}
