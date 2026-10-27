#!/usr/bin/env bash
# Shared storage defaults; source from setup/environment entry points.
# Keep repositories, logs and the small uv executable in WORK. Rebuildable
# environments and caches belong in SCRATCH to avoid the shared WORK inode cap.

if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "WORK and SCRATCH must be defined before sourcing scratch_paths.sh." >&2
  return 2 2>/dev/null || exit 2
fi

_gnbm_set_scratch_paths() {
  local variable current
  # Login shells and old job exports may still contain the former defaults,
  # including the isolated FGW environment. Translate these paths as well.
  for variable in GNBM_VENV GNBM_A100_VENV GNBM_H100_VENV; do
    current="${!variable:-}"
    case "$current" in
      "$WORK"/venvs/*)
        printf -v "$variable" '%s' "$SCRATCH/venvs/${current#"$WORK/venvs/"}"
        export "$variable"
        ;;
    esac
  done
  for variable in UV_CACHE_DIR XDG_CACHE_HOME MPLCONFIGDIR PIP_CACHE_DIR \
    TORCH_EXTENSIONS_DIR GNBM_A100_EXTENSIONS_DIR GNBM_H100_EXTENSIONS_DIR \
    TORCH_HOME HF_HOME WANDB_CACHE_DIR WANDB_DATA_DIR; do
    current="${!variable:-}"
    case "$current" in
      "$WORK"/.cache|"$WORK"/.cache/*)
        printf -v "$variable" '%s' "$SCRATCH/${current#"$WORK/"}"
        export "$variable"
        ;;
    esac
  done

  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
  export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
  export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$SCRATCH/.local/share/wandb}"
}

_gnbm_set_scratch_paths
unset -f _gnbm_set_scratch_paths
