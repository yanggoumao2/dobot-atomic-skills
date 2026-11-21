#!/usr/bin/env bash
# Generate the shared Robonix gRPC stubs and MCP dataclasses for this package.
set -eo pipefail

PKG_ROOT="$RBNX_PACKAGE_ROOT"
if [[ -z "$PKG_ROOT" ]]; then
  PKG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

CODEGEN_FLAGS=(--mcp)
[[ "$RBNX_BUILD_CLEAN" == "1" ]] && CODEGEN_FLAGS+=(--clean)
CODEGEN_PYTHON=()
if [[ -n "$RBNX_CODEGEN_PYTHON" ]]; then
  CODEGEN_PYTHON=(--python "$RBNX_CODEGEN_PYTHON")
fi
exec rbnx codegen -p "$PKG_ROOT" "${CODEGEN_FLAGS[@]}" "${CODEGEN_PYTHON[@]}"
