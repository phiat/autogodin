#!/usr/bin/env bash
# Bootstrap a JarvisLabs PyTorch instance for autogodin GPU runs.
#
# Assumptions about the JL image:
#   - Python 3.x + torch (CUDA) preinstalled
#   - apt + sudo available
#   - User home writable
#   - No Odin, no uv (we install both)
#
# This is the docker-less path: we build alpha_go_odin (.so) and
# alpha_go_cpp (pybind11 wheel) directly on the instance, set up the
# /nfs symlink that train.py/pre_collect_random.py expect, and leave a
# python venv ready to run.
#
# Usage on the JL instance (after `ssh <user>@<host>`):
#   curl -fsSL <this-script-url> | bash
# or
#   scp scripts/jl_bootstrap.sh <host>:/tmp/bootstrap.sh
#   ssh <host> 'bash /tmp/bootstrap.sh'
#
# Idempotent: re-running is safe.

set -euo pipefail

REPO_URL="${AUTOGODIN_REPO_URL:-https://github.com/phiat/autogodin.git}"
REPO_BRANCH="${AUTOGODIN_BRANCH:-main}"
WORK_DIR="${WORK_DIR:-$HOME/autogodin-work}"
NFS_LOCAL="${NFS_LOCAL:-$HOME/nfs-local}"

echo "===== autogodin JL bootstrap ====="
echo "  repo:      $REPO_URL ($REPO_BRANCH)"
echo "  work dir:  $WORK_DIR"
echo "  /nfs ->    $NFS_LOCAL"
echo

# -- 1. system deps for building alpha_go_cpp ---------------------------
echo "[1/6] apt deps (build-essential, cmake, eigen3, python3-dev)..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential cmake \
    clang \
    libeigen3-dev zlib1g-dev \
    python3-dev \
    git curl ca-certificates \
    >/dev/null

# -- 2. uv (Python project manager) -------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[2/6] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[2/6] uv already present: $(uv --version)"
fi
export PATH="$HOME/.local/bin:$PATH"

# -- 3. Odin nightly (via direct download — avoids mise on a one-shot box)
ODIN_VERSION="${ODIN_VERSION:-dev-2026-05}"
ODIN_DIR="$HOME/.local/odin"
if [ ! -x "$ODIN_DIR/odin" ]; then
    echo "[3/6] installing Odin $ODIN_VERSION via mise..."
    mkdir -p "$ODIN_DIR"
    if ! command -v mise >/dev/null 2>&1; then
        curl -fsSL https://mise.run | sh >/dev/null
        export PATH="$HOME/.local/bin:$PATH"
    fi
    # Mise's odin plugin (asdf:jtakakura/asdf-odin) resolves `dev-YYYY-MM`
    # to the matching dated nightly. `nightly` itself isn't a valid release
    # tag on the odin-lang/Odin github releases page (404).
    mise install "odin@$ODIN_VERSION"
    ODIN_BIN="$(mise where "odin@$ODIN_VERSION")/odin"
    ln -sf "$ODIN_BIN" "$ODIN_DIR/odin"
else
    echo "[3/6] Odin already present"
fi
export PATH="$ODIN_DIR:$PATH"
"$ODIN_DIR/odin" version

# -- 4. clone autogodin + upstream autogo -------------------------------
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
if [ ! -d autogodin ]; then
    echo "[4/6] cloning autogodin..."
    git clone --depth=1 --branch="$REPO_BRANCH" "$REPO_URL" autogodin
fi
cd "$WORK_DIR/autogodin"
git fetch origin "$REPO_BRANCH"
git reset --hard "origin/$REPO_BRANCH"

# Fetch + patch upstream autogo per autogo.pin.
bash scripts/setup_autogo.sh

# -- 5. build alpha_go_odin + alpha_go_cpp ------------------------------
echo "[5/6] building alpha_go_odin (-o:speed)..."
ODIN_OPT="-o:speed" bash scripts/build_odin.sh

echo "[5/6] setting up autogo venv + building alpha_go_cpp wheel..."
cd "$WORK_DIR/autogodin/autogo"
uv sync
bash scripts/build_cpp.sh

# Sanity: confirm both backends importable.
PYBIN="$WORK_DIR/autogodin/autogo/.venv/bin/python"
PYTHONPATH="$WORK_DIR/autogodin/python:$WORK_DIR/autogodin/autogo/src" \
    "$PYBIN" -c "
import alpha_go_cpp, alpha_go_odin, torch, numpy
print('alpha_go_cpp:', alpha_go_cpp.__file__)
print('alpha_go_odin:', alpha_go_odin.__file__)
print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available())
assert alpha_go_cpp.MCTSTree is not alpha_go_odin.MCTSTree, 'shim alias bug'
"

# -- 6. /nfs symlink so train.py / pre_collect_random.py work locally ---
echo "[6/6] /nfs symlink -> $NFS_LOCAL ..."
mkdir -p "$NFS_LOCAL/game_data_root" "$NFS_LOCAL/checkpoints"
if [ ! -L /nfs ] && [ ! -d /nfs ]; then
    sudo ln -s "$NFS_LOCAL" /nfs
elif [ -L /nfs ]; then
    echo "  /nfs symlink exists -> $(readlink /nfs)"
else
    echo "  /nfs is a real dir, leaving alone"
fi

echo
echo "===== bootstrap complete ====="
echo "  work_dir:    $WORK_DIR/autogodin"
echo "  python:      $PYBIN"
echo "  PYTHONPATH:  $WORK_DIR/autogodin/python:$WORK_DIR/autogodin/autogo/src"
echo "  GAME_DATA_DIR=$NFS_LOCAL/game_data_root  (or /nfs/game_data_root)"
echo
echo "Run the GPU experiment with:"
echo "  cd $WORK_DIR/autogodin"
echo "  PYTHONPATH=\"\$PWD/python:autogo/src\" GAME_DATA_DIR=$NFS_LOCAL/game_data_root \\"
echo "    autogo/.venv/bin/python experiments/2026-05-16_17-21-ydh1-phaseB-direct/run.py"
