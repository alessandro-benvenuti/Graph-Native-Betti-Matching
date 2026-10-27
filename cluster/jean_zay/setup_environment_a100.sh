#!/usr/bin/env bash
# One-time A100 login-node setup. This installs Python packages but leaves the
# sm_80 CUDA extension build to the first bounded A100 development job.
set -eo pipefail

if [[ -z "${WORK:-}" || -z "${SCRATCH:-}" ]]; then
  echo "Run this script on a Jean Zay login node." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_dir/cluster/jean_zay/scratch_paths.sh"
venv="${GNBM_A100_VENV:-$SCRATCH/venvs/vascular-graph-extraction-a100-torch230}"
uv_bin="$WORK/tools/uv/uv"

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
set -u
export IDR_DEBUG="${IDR_DEBUG:-WARN}"

if [[ ! -x "$uv_bin" ]]; then
  echo "uv is missing at $uv_bin" >&2
  echo "Install uv under \$WORK before running this script." >&2
  exit 2
fi

mkdir -p "$(dirname "$venv")" "$UV_CACHE_DIR"

if [[ ! -x "$venv/bin/python" ]]; then
  "$uv_bin" venv \
    --python "$(command -v python)" \
    --system-site-packages \
    "$venv"
fi

source "$venv/bin/activate"

# Preserve experiment extras (e.g. Optuna) from the former WORK environment.
# Inspect only its own distribution metadata, never the inherited module stack.
record="$WORK/environment-records/$(basename "$venv").requirements.txt"
legacy_venv="$WORK/venvs/$(basename "$venv")"
if [[ -f "$legacy_venv/pyvenv.cfg" ]]; then
  python "$repo_dir/cluster/jean_zay/snapshot_environment_requirements.py" \
    --venv "$legacy_venv" --output "$record"
fi

# Keep the module-provided PyTorch/CUDA stack and install only project packages.
"$uv_bin" pip install \
  --python "$venv/bin/python" \
  --no-deps \
  --requirements "$repo_dir/requirements/jean-zay.txt"

"$uv_bin" pip install \
  --python "$venv/bin/python" \
  "wandb==0.28.1"

if [[ -s "$record" ]]; then
  "$uv_bin" pip install --python "$venv/bin/python" --no-deps --requirements "$record"
fi

python - <<'PY'
import sys
import torch
import monai
import medpy
import numpy
import scipy
import wandb
import yaml

print("Python executable:", sys.executable)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("MONAI:", monai.__version__)
print("MedPy:", medpy.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("W&B:", wandb.__version__)
print("PyYAML:", yaml.__version__)

if torch.__version__.split("+")[0] != "2.3.0":
    raise SystemExit("Expected Jean Zay A100 PyTorch 2.3.0")
if str(torch.version.cuda) != "12.2":
    raise SystemExit("Expected the Jean Zay A100 CUDA 12.2 build")
if wandb.__version__ != "0.28.1":
    raise SystemExit("Expected W&B 0.28.1")
PY

"$uv_bin" pip check --python "$venv/bin/python"

echo
echo "A100 environment ready: $venv"
echo "Next: source cluster/jean_zay/env_a100.sh"
