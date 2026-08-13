#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build the extension in the project environment." >&2
  exit 2
fi
if [[ -z "${VIRTUAL_ENV:-}" || ! -x "$VIRTUAL_ENV/bin/python" ]]; then
  echo "Activate the project virtual environment before building." >&2
  exit 2
fi

if [[ "${GNBM_FORCE_REBUILD_OPS:-0}" != "1" ]] && python - <<'PY'
import MultiScaleDeformableAttention3D  # noqa: F401
PY
then
  echo "MultiScaleDeformableAttention3D is already installed; skipping rebuild."
  exit 0
fi

uv pip install \
  --python "$VIRTUAL_ENV/bin/python" \
  --reinstall \
  --no-build-isolation \
  models/ops \
  --verbose
