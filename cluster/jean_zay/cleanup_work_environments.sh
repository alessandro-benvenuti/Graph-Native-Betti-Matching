#!/usr/bin/env bash
# Dry-run by default. The Python helper archives and verifies before removal.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_dir/cleanup_work_environments.py" "$@"
