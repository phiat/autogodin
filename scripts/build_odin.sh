#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

mkdir -p build
OUT="build/libalpha_go_odin.so"
ODIN_OPT="${ODIN_OPT:-}"

odin build odin/alpha_go \
	-build-mode:shared \
	-out:"$OUT" \
	$ODIN_OPT "$@"

echo "built: $HERE/$OUT"
nm -D "$OUT" 2>/dev/null | awk '$2 == "T" {print "  export: " $3}' | head -20 || true
