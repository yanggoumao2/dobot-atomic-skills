#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -eo pipefail

PKG_ROOT="$RBNX_PACKAGE_ROOT"
if [[ -z "$PKG_ROOT" ]]; then
  PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$PKG_ROOT"
NOVA2_ROOT="$(cd "$PKG_ROOT/../.." && pwd)"
export PYTHONPATH="$(rbnx path robonix-api):$NOVA2_ROOT:$PKG_ROOT:$PKG_ROOT/rbnx-build/codegen/proto_gen:$PYTHONPATH"

exec python3 -m nova2_arm.driver
