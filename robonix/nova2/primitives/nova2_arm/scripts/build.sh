#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -eo pipefail

PKG_ROOT="$RBNX_PACKAGE_ROOT"
if [[ -z "$PKG_ROOT" ]]; then
  PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

if [[ "$RBNX_BUILD_CLEAN" == "1" ]]; then
  rbnx codegen -p "$PKG_ROOT" --clean
else
  rbnx codegen -p "$PKG_ROOT"
fi
echo "[nova2_arm/build] done"
