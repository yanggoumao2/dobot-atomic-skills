#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"
VENV="$ROOT/.venv-calibration"

if [[ -x "$VENV/Scripts/python.exe" ]]; then
  PYTHON="$VENV/Scripts/python.exe"
elif [[ -x "$VENV/bin/python" ]]; then
  PYTHON="$VENV/bin/python"
else
  BASE=""
  for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      BASE="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$BASE" ]]; then
    printf '%s\n' "需要 Python 3.11 或 3.12；未在 PATH 中找到。" >&2
    exit 2
  fi
  "$BASE" -m venv "$VENV"
  if [[ -x "$VENV/Scripts/python.exe" ]]; then
    PYTHON="$VENV/Scripts/python.exe"
  else
    PYTHON="$VENV/bin/python"
  fi
fi

"$PYTHON" -m pip install --disable-pip-version-check -r requirements-calibration.txt
if [[ $# -eq 0 ]]; then
  set -- --open-browser
fi
exec "$PYTHON" -X utf8 -B -m task_skills.run_calibration_ui "$@"
