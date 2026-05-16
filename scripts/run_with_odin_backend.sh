#!/usr/bin/env bash
# Run any command with `import alpha_go_cpp` aliased to the Odin backend.
#
# Mechanism: prepends python/odin_backend/ (which contains a shim
# alpha_go_cpp.py re-exporting alpha_go_odin) to PYTHONPATH so the shim
# wins over any installed pybind11 alpha_go_cpp. Sets ALPHAGO_BACKEND=odin
# as a sentinel for code that wants to branch on it.
#
# Usage:
#   scripts/run_with_odin_backend.sh <command> [args...]
# Examples:
#   scripts/run_with_odin_backend.sh python -c "import alpha_go_cpp; print(alpha_go_cpp.__file__)"
#   scripts/run_with_odin_backend.sh autogo/.venv-cpponly/bin/python autogo/tests/test_smoke.py

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

export PYTHONPATH="$REPO_ROOT/python/odin_backend:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export ALPHAGO_BACKEND=odin

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "  e.g.: $0 python -c 'import alpha_go_cpp; print(alpha_go_cpp.__file__)'" >&2
  exit 2
fi

exec "$@"
