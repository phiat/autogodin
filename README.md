# autogodin

Odin port of the C++ MCTS + Go-board core from [ericjang/autogo](https://github.com/ericjang/autogo), plus a workspace for experiments and research around it.

The Python + training side of autogo stays unchanged in its own repo; this one houses the Odin reimplementation of the C++ hot path, a ctypes shim that exposes the same surface as the upstream `alpha_go_cpp` pybind11 module, and a parity / benchmark harness.

## What's here

```
odin/alpha_go/          Odin source: GoBoard + thin Game-vtable adapter + C-ABI exports
odin/vendor/mcts-odin/  Pinned copy of mcts-odin/mcts/ (commit + license + sync date)
odin/tests/             Odin @(test) procs (37 cases: board + adapter integration)
python/alpha_go_odin/   ctypes wrapper mirroring alpha_go_cpp's OO API
python/parity/          Deterministic Zobrist-fingerprint parity harness
scripts/build_odin.sh   builds build/libalpha_go_odin.so
```

`autogo/` is **not** in this repo — clone it as a sibling when needed for cross-language testing (see *Getting started* below).

## Status

**Phase 1 (Odin port) — done.** **Phase 2 (foundation + MCTS vendor migration + batched FFI) — done.**

- GoBoard: Zobrist-incremental positional superko, KataGo-aligned no-suicide rule, Tromp-Taylor area scoring.
- MCTS: vendored from [mcts-odin](https://github.com/phiat/mcts-odin) (`odin/vendor/mcts-odin/`, pinned commit; see `VERSION`). Packed-slot nodes, branchless PUCT, linear-space priors, FPU (parent-Q with reduction), per-tree scratch arena, leaf-parallel batched with virtual loss, Dirichlet noise, PCR, subtree reuse, root-parallel threading. The local `go_adapter.odin` is a ~140-LOC Game vtable bridging GoBoard.
- 37/37 Odin `@(test)` cases pass clean under the memory tracker.
- 44 `alphago_*` C-ABI symbols in `libalpha_go_odin.so`; Python ctypes shim mirrors upstream `alpha_go_cpp`'s OO API plus a new `MCTSTree.run_simulations_batched(num_sims, batch_size, batched_evaluator)` for NN-eval workloads.

### Correctness

- **Board parity** (`python/parity/random_games_dual.py`): Odin and upstream C++ produce a byte-identical SHA-256 fingerprint `109bd08a…` over 10 seeded games × ~200 moves.
- **MCTS-layer A/B**: 100 games of Odin-MCTS vs C++-MCTS at 200 sims/move, uniform-policy evaluator. Pre-vendor result was 50–50, Wilson 95% CI [0.404, 0.596]. Post-vendor, this regime is dominated by the FPU concentration-vs-spread tradeoff documented in `odin/vendor/mcts-odin/mcts/mcts.odin` (Config.fpu_reduction): under uniform priors with very low sim budgets, FPU's correct-but-thin spread can lose to C++'s accidental concentration. A/B parity is the gate that matters under NN evaluators (where priors are informative); queued under `experiments/` for the next NN-eval pass.

### Throughput

9×9 micro-bench (1600 sims/move × 32 moves, single-thread, miniwini host, vendor v0.4.0, post-`zkq` legality probe).

**Sequential evaluator** (one leaf at a time):

| Backend                                   | sims/sec        | vs C++  |
|-------------------------------------------|-----------------|---------|
| `alpha_go_cpp` (upstream)                 | 8,713 ± 66      | 1.00×   |
| `alpha_go_odin` Python ctypes shim        | 44,374 ± 270    | 5.09×   |
| `alpha_go_odin` in-process Odin evaluator | **69,234 ± 187**  | **7.95×** |

**Batched evaluator** (`run_simulations_batched`, key cells; full grid in `experiments/2026-05-16_13-30-ydh.3-batched-sweep/`):

| latency      | batch=1 | batch=128 | speedup | Python tax |
|--------------|---------|-----------|---------|------------|
| 0us          | 23,268  | 32,132    | 1.4×    | 29%        |
| 100us        | 4,499   | 30,504    | 6.8×    | 28%        |
| 1ms          | 771     | 24,065    | **31×** | 24%        |

Sequential numbers track the phase-2 progression: pre-foundation 2,859 / pre-vendor 7,927 / pre-FPU 13,613 / post-FPU 25,618 → post-`zkq` 69,234 in-process. The `zkq` jump (commit `pending`) replaced `is_legal_flat`'s clone-and-simulate path with an in-place probe + restore — kills ~30-35% of CPU that was going through the Odin runtime map machinery on a discarded clone of `seen_hashes`. The batched table below is pre-`zkq`; expect similar uplift on a re-run.

**Phase 3** — experimentation, training A/Bs, optional GPU runs.

## Getting started

```bash
# Prerequisites: Odin nightly, gcc/clang, Python 3.10+, just (mise install just),
# optional uv for the upstream autogo env.
git clone https://github.com/phiat/autogodin.git
cd autogodin

# Optional: sibling clone of the upstream autogo for parity / Python tests.
# Pins the upstream SHA from autogo.pin and applies our build_cpp patch.
./scripts/setup_autogo.sh

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
# scripts/setup_autogo.sh (above) clones the pinned SHA and applies our
# build_cpp.sh libpython-hardcode patch — start from here:
cd autogo
uv venv -p 3.12 .venv-cpponly
uv pip install --python .venv-cpponly/bin/python numpy
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv-cpponly" bash scripts/build_cpp.sh
```

End-to-end on a clean host: ~3 min, no GPU. The .so installs into `.venv-cpponly/lib/python3.12/site-packages/`.

## Workflow

Project conventions (compute hosts, GPU policy, parallel-agent rules, build/test/parity gates) live in `AGENTS.md` for collaborators.

## Acknowledgements

Upstream codebase: [ericjang/autogo](https://github.com/ericjang/autogo). All algorithms ported here are direct translations of that C++; this repo's contribution is the Odin port + ctypes shim + parity tooling.
