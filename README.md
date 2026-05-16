# autogodin

Odin port of the C++ MCTS + Go-board core from [ericjang/autogo](https://github.com/ericjang/autogo), plus a workspace for experiments and research around it.

The Python + training side of autogo stays unchanged in its own repo; this one houses the Odin reimplementation of the C++ hot path, a ctypes shim that exposes the same surface as the upstream `alpha_go_cpp` pybind11 module, and a parity / benchmark harness.

## What's here

```
odin/alpha_go/      Odin source: GoBoard + MCTS + C-ABI exports
odin/tests/         Odin @(test) procs (31 cases ported from gtest)
python/alpha_go_odin/   ctypes wrapper mirroring alpha_go_cpp's OO API
python/parity/      Deterministic Zobrist-fingerprint parity harness
scripts/build_odin.sh   builds build/libalpha_go_odin.so
```

`autogo/` is **not** in this repo — clone it as a sibling when needed for cross-language testing (see *Getting started* below).

## Status

**Phase 1 (Odin port) — done.** **Phase 2 (foundation + parity) — done.**

- GoBoard: Zobrist-incremental positional superko, KataGo-aligned no-suicide rule, Tromp-Taylor area scoring.
- MCTS: working-board do_move/undo_move descent (no per-node board clone), per-tree virtual arena, leaf-parallel batched with virtual loss, Dirichlet noise, PCR.
- 37/37 Odin `@(test)` cases pass clean under the memory tracker.
- 42 `alphago_*` C-ABI symbols in `libalpha_go_odin.so`; Python ctypes shim mirrors upstream `alpha_go_cpp`'s OO API.

### Correctness

- **Board parity** (`python/parity/random_games_dual.py`): Odin and upstream C++ produce a byte-identical SHA-256 fingerprint `109bd08a…` over 10 seeded games × ~200 moves.
- **MCTS-layer A/B** (`experiments/2026-05-16_11-25-mcts-ab-odin-vs-cpp/`): 100 games of Odin-MCTS vs C++-MCTS at 200 sims/move, uniform-policy evaluator, alternating colors. Outcome: 50–50, Wilson 95% CI [0.404, 0.596]. MCTS is semantically equivalent at this evaluator class.

### Throughput

ydh.2 micro-bench (9×9, 1600 sims/move × 32 moves, single-thread, NN-free):

| Backend                         | sims/sec     | vs C++  |
|---------------------------------|--------------|---------|
| `alpha_go_cpp` (upstream)       | ~8,500       | 1.00×   |
| `alpha_go_odin` (post-foundation) | **7,927 ± 12** | 0.93×   |
| autogodin pre-foundation        | 2,859        | 0.34×   |

Foundation refactor (`bd close autogodin-4rw`) lifted Odin from 0.34× to 0.93× C++. The remaining gap is dominated by Python-callback marshalling per leaf; see `bd show autogodin-ci2` for the next perf direction (potential adoption of [mcts-odin](https://github.com/phiat/mcts-odin), a generic Odin MCTS package extracted from this work).

**Phase 3** — experimentation, training A/Bs, optional GPU runs. See `bd ready`.

## Getting started

```bash
# Prerequisites: Odin nightly, gcc/clang, Python 3.10+, just (mise install just),
# optional uv for the upstream autogo env.
git clone https://github.com/phiat/autogodin.git
cd autogodin

# Optional: sibling clone of the upstream autogo for parity / Python tests.
git clone https://github.com/ericjang/autogo.git autogo

# Per-machine env overrides (gitignored). Defaults work out of the box.
cp .env.example .env

# Common commands:
just            # list recipes
just build      # build/libalpha_go_odin.so
just test       # full Odin test suite
just smoke      # single-test smoke (override with: just smoke <name>)
just parity     # Zobrist-fingerprint parity check vs committed fixture
just bench      # ydh.2 MCTS throughput bench (just bench cpp ... for cpp backend)
just check      # pre-push gate: build + test + parity
```

Underlying scripts still work directly (`./scripts/build_odin.sh`, `odin test odin/tests`, etc.) — `just` is convenience, not a wrapper requirement. Build flags can be overridden via `ODIN_OPT` in `.env` or inline.

## Parity harness

`python/parity/random_games.py` plays N seeded random games, captures the per-move Zobrist hash + ko_point + score / winner, and SHA-256 fingerprints the trace. `random_games_dual.py` does the same but on both backends side-by-side, requiring an importable `alpha_go_cpp` (see *Optional: building the C++ backend* below).

```bash
just parity                                                                # check fingerprint against committed fixture
python python/parity/random_games.py --emit /tmp/trace.json                # write the full trace
PYTHONPATH=python autogo/.venv-cpponly/bin/python \
  python/parity/random_games_dual.py --backend both                        # cross-language diff
```

## Optional: building the C++ backend

For cross-language tests / strength A/Bs, build upstream `alpha_go_cpp` against a minimal Python venv (no torch needed for the .so itself):

```bash
cd autogo
uv venv -p 3.12 .venv-cpponly
uv pip install --python .venv-cpponly/bin/python numpy
# Patch hardcoded libpython3.10.so out of build_cpp.sh:
python3 ../tools/patches/upstream_build_cpp_fix.py
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv-cpponly" bash scripts/build_cpp.sh
```

End-to-end on a clean host: ~3 min, no GPU. The .so installs into `.venv-cpponly/lib/python3.12/site-packages/`.

## Workflow

Tasks live in `bd` (beads); the Dolt-backed DB syncs over the same git remote via `bd dolt push/pull` (a separate `refs/dolt/data` ref, kept out of the working tree). Start with `bd ready` to see the unblocked queue. Project conventions (compute hosts, GPU policy, parallel-agent rules) live in `AGENTS.md` for collaborators.

## Acknowledgements

Upstream codebase: [ericjang/autogo](https://github.com/ericjang/autogo). All algorithms ported here are direct translations of that C++; this repo's contribution is the Odin port + ctypes shim + parity tooling.
