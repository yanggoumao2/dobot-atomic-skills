# Source from any directory: source /path/to/repo/scripts/linux/activate.sh
DOBOT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$DOBOT_REPO/.venv-teleop/bin/activate"
export PYTHONNOUSERSITE=1
export DOBOT_CONFIG="${DOBOT_CONFIG:-$DOBOT_REPO/config/linux.local.ini}"
export DOBOT_DEVICE_LOCK_PATH="${DOBOT_DEVICE_LOCK_PATH:-$(dirname "$DOBOT_REPO")/.local/runtime/device-owner}"
export DOBOT_ACT_RECORDING=1
cd "$DOBOT_REPO"
