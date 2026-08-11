#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-build-isolation --editable models/ops --verbose
