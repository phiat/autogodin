#!/usr/bin/env bash
# Fetch upstream autogo at the pinned SHA and apply our build_cpp patch.
#
# autogo/ is intentionally a sibling clone (with its own .git) rather than a
# submodule: we patch its build script (libpython hardcode fix) and a real
# submodule would force that patch upstream first. Pinning lives in
# autogo.pin at the repo root.
#
# Idempotent:
#   - if autogo/ doesn't exist, clones it
#   - if autogo/ exists at the pinned SHA, only re-applies the patch (no-op
#     if already applied)
#   - if autogo/ exists at a different SHA, refuses to touch it and prints
#     the discrepancy so the user can decide
#
# Usage: scripts/setup_autogo.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PIN_FILE="$REPO_ROOT/autogo.pin"
AUTOGO_DIR="$REPO_ROOT/autogo"
PATCH_SCRIPT="$REPO_ROOT/tools/patches/upstream_build_cpp_fix.py"

if [ ! -f "$PIN_FILE" ]; then
  echo "error: $PIN_FILE not found" >&2
  exit 1
fi

PINNED_URL=$(awk -F': +' '/^upstream:/ {print $2}' "$PIN_FILE")
PINNED_SHA=$(awk -F': +' '/^commit:/   {print $2}' "$PIN_FILE")

if [ -z "$PINNED_URL" ] || [ -z "$PINNED_SHA" ]; then
  echo "error: failed to parse upstream/commit from $PIN_FILE" >&2
  exit 1
fi

echo "autogo pin: $PINNED_URL @ $PINNED_SHA"

if [ ! -d "$AUTOGO_DIR/.git" ]; then
  echo "==> cloning autogo into $AUTOGO_DIR"
  git clone "$PINNED_URL" "$AUTOGO_DIR"
  ( cd "$AUTOGO_DIR" && git checkout --detach "$PINNED_SHA" )
else
  CURRENT_SHA=$(cd "$AUTOGO_DIR" && git rev-parse HEAD)
  if [ "$CURRENT_SHA" != "$PINNED_SHA" ]; then
    echo "warning: $AUTOGO_DIR is at $CURRENT_SHA, pin wants $PINNED_SHA" >&2
    echo "         leaving it alone — run 'cd autogo && git checkout $PINNED_SHA' if you want to sync" >&2
  else
    echo "==> autogo/ already at pinned SHA $PINNED_SHA"
  fi
fi

if [ -f "$PATCH_SCRIPT" ] && [ -f "$AUTOGO_DIR/scripts/build_cpp.sh" ]; then
  # Patch is idempotent at the line-removal level; the script's assertions
  # fail loudly if the lines aren't present (i.e. patch already applied).
  if grep -q 'libpython3.10.so' "$AUTOGO_DIR/scripts/build_cpp.sh"; then
    echo "==> applying build_cpp libpython-hardcode patch"
    ( cd "$AUTOGO_DIR" && python3 "$PATCH_SCRIPT" )
  else
    echo "==> build_cpp.sh already patched (libpython3.10.so reference absent)"
  fi
fi

echo
echo "autogo ready at $AUTOGO_DIR"
echo "Next: see README 'Optional: building the C++ backend' for the venv + build steps."
