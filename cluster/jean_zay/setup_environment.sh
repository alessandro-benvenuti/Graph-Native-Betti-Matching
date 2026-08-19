#!/usr/bin/env bash
# One-time login-node setup. This downloads Python wheels but does not compile
# the CUDA extension or run GPU work.
set -eo pipefail

if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "Run this script on a Jean Zay login node." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="${GNBM_VENV:-$WORK/venvs/vascular-graph-extraction-h100-torch231}"
uv_bin="$WORK/tools/uv/uv"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1
set -u
export IDR_DEBUG="${IDR_DEBUG:-WARN}"

if [[ ! -x "$uv_bin" ]]; then
  echo "uv is missing at $uv_bin" >&2
  echo "Install uv under \$WORK before running this script." >&2
  exit 2
fi

mkdir -p "$WORK/venvs" "$WORK/.cache/uv"
export UV_CACHE_DIR="$WORK/.cache/uv"

if [[ ! -x "$venv/bin/python" ]]; then
  "$uv_bin" venv \
    --python "$(command -v python)" \
    --system-site-packages \
    "$venv"
fi

if [[ -e "$repo_dir/.venv" && ! -L "$repo_dir/.venv" ]]; then
  echo "$repo_dir/.venv exists but is not a symbolic link; refusing to replace it." >&2
  exit 2
fi
if [[ -L "$repo_dir/.venv" ]]; then
  linked="$(readlink -f "$repo_dir/.venv")"
  expected="$(readlink -f "$venv")"
  if [[ "$linked" != "$expected" ]]; then
    ln -sfn "$venv" "$repo_dir/.venv"
  fi
else
  ln -s "$venv" "$repo_dir/.venv"
fi

source "$venv/bin/activate"

# This is a complete freeze of the non-PyTorch Gardenia environment. --no-deps
# prevents MONAI from installing another PyTorch build over the cluster module.
"$uv_bin" pip install \
  --python "$venv/bin/python" \
  --no-deps \
  --requirements "$repo_dir/requirements/jean-zay.txt"

python - <<'PY'
import sys
import torch
import monai
import numpy
import scipy
import yaml

print("Python executable:", sys.executable)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("MONAI:", monai.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyYAML:", yaml.__version__)

if torch.__version__.split("+")[0] != "2.3.1":
    raise SystemExit("Expected Jean Zay PyTorch 2.3.1")
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit("Expected the Jean Zay H100 CUDA 12 PyTorch build")
PY

"$uv_bin" pip check --python "$venv/bin/python"

echo
echo "Environment ready: $venv"
echo "Repository link:  $repo_dir/.venv"
echo "Next: submit the bounded development job with submit_debug.sh"
