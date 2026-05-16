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

**Phase 1 (Odin port) — done.**

- GoBoard: Zobrist-incremental positional superko, KataGo-aligned no-suicide rule, Tromp-Taylor area scoring.
- MCTS: recursive AlphaZero playout, leaf-parallel batched with virtual loss, Marsaglia-Tsang Dirichlet noise, PCR, fast_rollout.
- 31/31 ported gtest cases pass, clean under Odin's memory tracker.
- 42 `alphago_*` C-ABI symbols in `libalpha_go_odin.so`; Python ctypes shim exposes the upstream pybind11 OO API (single-state MCTS callback; batched callback is a follow-up).
- Parity harness fingerprint (`python/parity/random_games.py`) is self-consistent across runs; cross-language diff against upstream C++ is queued behind getting the autogo dev-env buildable here.

**Phase 2** — experimentation + C++ comparison. See `bd ready`.

## Getting started

```bash
# Prerequisites: Odin nightly, gcc/clang, Python 3.10+, optional uv for the upstream autogo env.
git clone https://github.com/phiat/autogodin.git
cd autogodin

# Optional: sibling clone of the upstream autogo for parity / Python tests.
git clone https://github.com/ericjang/autogo.git autogo

# Build the Odin shared lib (-> build/libalpha_go_odin.so).
./scripts/build_odin.sh

# Run Odin unit tests (31 cases).
odin test odin/tests

# Smoke-test the Python ctypes shim + run the parity fingerprint.
python python/parity/random_games.py --check python/parity/fixtures/random_games_v0.json
```

Build flags can be overridden via `ODIN_OPT`, e.g.:

```bash
ODIN_OPT="-o:speed -no-bounds-check" ./scripts/build_odin.sh
```

## Parity harness

`python/parity/random_games.py` plays N seeded random games, captures the per-move Zobrist hash + ko_point + score / winner, and SHA-256 fingerprints the trace. The same script will diff against the upstream C++ once autogo's pybind11 module is buildable on the same host (tracked under `autogodin-ydh.1`).

```bash
python python/parity/random_games.py                                       # print fingerprint
python python/parity/random_games.py --emit /tmp/trace.json                # write the full trace
python python/parity/random_games.py --check python/parity/fixtures/...    # exits 2 on drift
```

## Workflow

Tasks live in `bd` (beads); the Dolt-backed DB syncs over the same git remote via `bd dolt push/pull` (a separate `refs/dolt/data` ref, kept out of the working tree). Start with `bd ready` to see the unblocked queue. Project conventions (compute hosts, GPU policy, parallel-agent rules) live in `AGENTS.md` for collaborators.

## Acknowledgements

Upstream codebase: [ericjang/autogo](https://github.com/ericjang/autogo). All algorithms ported here are direct translations of that C++; this repo's contribution is the Odin port + ctypes shim + parity tooling.
