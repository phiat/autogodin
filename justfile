# autogodin developer commands. `just` (no args) lists recipes.
#
# Loads .env if present so per-machine overrides stay out of the tracked tree.
# Copy .env.example -> .env to customize.

set dotenv-load := true
set dotenv-required := false

odin_opt := env_var_or_default('ODIN_OPT', '-o:speed')
test_opt := env_var_or_default('ODIN_TEST_OPT', '-debug')
smoke    := env_var_or_default('SMOKE_TEST', 'do_undo_simple_place')

bench_dir := 'experiments/2026-05-16_05-40-mcts-bench-cpp-vs-odin'

# List available recipes.
default:
    @just --list --unsorted

# Build the Odin shared lib (build/libalpha_go_odin.so).
build:
    ODIN_OPT={{odin_opt}} ./scripts/build_odin.sh

# Full Odin test suite (single-threaded so the memory tracker is per-test stable).
test:
    timeout 30s odin test odin/tests {{test_opt}} -define:ODIN_TEST_THREADS=1

# Run one named test (faster smoke after a code change).
smoke name=smoke:
    timeout 10s odin test odin/tests {{test_opt}} -define:ODIN_TEST_THREADS=1 -define:ODIN_TEST_NAMES={{name}}

# Parity fingerprint of N seeded random games against the committed fixture.
parity:
    python python/parity/random_games.py --check python/parity/fixtures/random_games_v0.json

# MCTSTree readout contract parity (requires alpha_go_cpp; see CONTRIBUTING.md).
parity-readouts:
    PYTHONPATH=python autogo/.venv/bin/python python/parity/readouts_dual.py

# ydh.2 throughput bench (3 trials, 1600 sims x 32 moves). Writes CSV to /tmp.
bench backend='odin' trials='3' sims='1600' moves='32':
    timeout 120s python {{bench_dir}}/bench.py \
        --backend {{backend}} --trials {{trials}} --warmup 1 \
        --num-sims {{sims}} --num-moves {{moves}} --out /tmp/bench.csv

# Remove build artifacts.
clean:
    rm -rf build

# Pre-push gate: build + tests + parity. Run before `git push`.
check: build test parity

# Beads quick-look (ready queue).
ready:
    bd ready

# Show a beads issue by id, e.g. `just show autogodin-dsi`.
show id:
    bd show {{id}}
