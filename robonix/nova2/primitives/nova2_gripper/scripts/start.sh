#!/usr/bin/env bash
# Start this provider locally; Robonix handles Atlas and transport sockets.
set -eo pipefail

PKG_ROOT="$RBNX_PACKAGE_ROOT"
if [[ -z "$PKG_ROOT" ]]; then
  PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
CODEGEN_ROOT="$PKG_ROOT/rbnx-build/codegen"
NOVA2_ROOT="$(cd "$PKG_ROOT/../.." && pwd)"
export PYTHONPATH="$PKG_ROOT:$NOVA2_ROOT:$CODEGEN_ROOT/proto_gen:$CODEGEN_ROOT/robonix_mcp_types:$PYTHONPATH"

cd "$PKG_ROOT"
exec python3 -m nova2_gripper.driver
