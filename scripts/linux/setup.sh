#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
# Reuse the installed training Python only to create an isolated teleop venv.
base_python="${DOBOT_BASE_PYTHON:-/home/yxy/data/projects/dobot-act/env/bin/python}"
"$base_python" -m venv .venv-teleop
.venv-teleop/bin/python -m pip install -r requirements-teleop-linux.txt
if [[ ! -e config/linux.local.ini ]]; then cp config/linux.example.ini config/linux.local.ini; fi
printf '%s\n' 'Fill config/linux.local.ini, then source scripts/linux/activate.sh'
