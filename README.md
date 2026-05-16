# autogodin

Odin port of the C++ MCTS + Go-board core from [ericjang/autogo](https://github.com/ericjang/autogo), plus a workspace for experiments and research around it.

The Python + training side of autogo stays unchanged in its own repo; this one houses the Odin reimplementation of the C++ hot path, a ctypes shim that exposes the same surface as the upstream `alpha_go_cpp` pybind11 module, and a parity / benchmark harness.

## What's here

```
odin/alpha_go/          Odin source: GoBoard + thin Game-vtable adapter + C-ABI exports
odin/vendor/mcts-odin/  Pinned copy of mcts-odin/mcts/ (commit + license + sync date)
odin/tests/             Odin @(test) procs (37 cases: board + adapter integration)
python/alpha_go_odin/   ctypes wrapper mirroring alpha_go_cpp's OO API
python/odin_backend/    drop-in shim: `import alpha_go_cpp` → alpha_go_odin (used by autogo)
python/parity/          Deterministic Zobrist-fingerprint parity harness
scripts/build_odin.sh   builds build/libalpha_go_odin.so
```

`autogo/` is **not** in this repo — clone it as a sibling when needed for cross-language testing (see *Getting started* below).

## Status

**Phase 1 (Odin port) — done.** **Phase 2 (foundation + MCTS vendor migration + batched FFI) — done.**

- GoBoard: Zobrist-incremental positional superko, KataGo-aligned no-suicide rule, Tromp-Taylor area scoring.
- MCTS: vendored from [mcts-odin](https://github.com/phiat/mcts-odin) (`odin/vendor/mcts-odin/`, pinned commit; see `VERSION`). Packed-slot nodes, branchless PUCT, linear-space priors, FPU (parent-Q with reduction), per-tree scratch arena, leaf-parallel batched with virtual loss, Dirichlet noise, PCR, subtree reuse, root-parallel threading. The local `go_adapter.odin` is a ~140-LOC Game vtable bridging GoBoard.
- 37/37 Odin `@(test)` cases pass clean under the memory tracker.
- 43 `alphago_*` C-ABI symbols in `libalpha_go_odin.so`; Python ctypes shim mirrors upstream `alpha_go_cpp`'s OO API plus `MCTSTree.run_simulations_batched` (leaf-parallel + virtual loss), `run_simulations_batched_flat` (no-dict scratch-ndarray evaluator for the batched path; `cg0`), `run_simulations_threaded` (root-parallel worker pool), and `run_simulations_flat` (no-dict scratch-ndarray for the sequential path; `cz9`).

### Correctness

- **Board parity** (`python/parity/random_games_dual.py`): Odin and upstream C++ produce a byte-identical SHA-256 fingerprint `109bd08a…` over 10 seeded games × ~200 moves.
- **MCTS-layer A/B**: 100 games of Odin-MCTS vs C++-MCTS at 200 sims/move, uniform-policy evaluator. Pre-vendor result was 50–50, Wilson 95% CI [0.404, 0.596]. Post-vendor, this regime is dominated by the FPU concentration-vs-spread tradeoff documented in `odin/vendor/mcts-odin/mcts/mcts.odin` (Config.fpu_reduction): under uniform priors with very low sim budgets, FPU's correct-but-thin spread can lose to C++'s accidental concentration. A/B parity is the gate that matters under NN evaluators (where priors are informative); queued under `experiments/` for the next NN-eval pass.

### Throughput

9×9 micro-bench (1600 sims/move × 32 moves, single-thread, miniwini host, vendor v0.4.0, post-`zkq`/`373`/`5km`/`cz9`).

**Sequential evaluator** (one leaf at a time):

| Backend                                                      | sims/sec        | vs C++  |
|--------------------------------------------------------------|-----------------|---------|
| `alpha_go_cpp` (upstream)                                    | 8,713 ± 66      | 1.00×   |
| `alpha_go_odin` Python ctypes shim, legacy dict evaluator    | 48,019 ± 307    | 5.51×   |
| `alpha_go_odin` Python ctypes shim, flat evaluator (`cz9`)   | 54,541 ± 616    | **6.26×** |
| `alpha_go_odin` in-process Odin evaluator                    | **76,159 ± 481**  | **8.74×** |

**Batched evaluator** (`run_simulations_batched`, in-process key cells; full grid in `experiments/2026-05-16_13-30-ydh.3-batched-sweep/`):

| latency      | batch=1 | batch=128 | speedup | Python tax |
|--------------|---------|-----------|---------|------------|
| 0us          | 62,482  | 87,281    | 1.4×    | 40%        |
| 100us        | 5,313   | 78,889    | 14.9×   | 37%        |
| 1ms          | 828     | 47,284    | **57×** | 23%        |

**Threaded evaluator** (`run_simulations_threaded`, miniwini, post-i5d):

| evaluator                    | n=0 (seq) | n=2    | n=4    | n=8    | best speedup |
|------------------------------|----------:|-------:|-------:|-------:|-------------:|
| Pure Python (GIL held)       | 24,798    | 20,752 | 17,952 | 17,574 | 0.84× (regress) |
| Python + `time.sleep(200µs)` | 2,949     | 6,380  | 11,533 | 12,740 | **4.32×** |

The threaded path adds tree-mutex / virtual-loss / worker-pool overhead
that's a net loss when the evaluator holds the GIL (every leaf serializes
in Python anyway). The win arrives when the evaluator drops into a
GIL-releasing C extension — numpy ops, torch inference, an HTTP-RPC
client — which the sleep cell stands in for. For batched NN workloads,
`run_simulations_batched` is still the right tool; threading shines on
single-leaf evaluators that yield the GIL. See
`experiments/2026-05-16_16-52-i5d-threaded-mcts/` for the sweep.

Sequential numbers track the phase-2 progression: pre-foundation 2,859 / pre-vendor 7,927 / pre-FPU 13,613 / post-FPU 25,618 / post-`zkq` 69,234 / post-`373` 74,899 → post-`5km` 76,159 in-process. The ydh.6 perf profile identified three hot-paths in legality; all three fixed: `zkq` (commit `25e0230`) replaced `is_legal_flat`'s clone-and-simulate with in-place probe + restore; `373` (commit `a700960`) replaced the capture-probe's full liberty enumeration with `would_capture_group_at` (bail on first off-candidate liberty); `5km` (commit `0c52ea8`) added `fill_legal_moves_flat` for caller-owned-buffer legal enumeration. Batched table re-run post-ydh.6: 1ms × batch=128 went 24,065 → 47,284 sims/sec (+96%); 1ms × batch=1 barely moved (771 → 828, dominated by sleep), so the within-row amortization speedup grew from 31× to 57×. Python tax (in-process vs ctypes) widened at low latency because Python's per-batch callback overhead didn't shrink with the Odin work — see `experiments/2026-05-16_13-30-ydh.3-batched-sweep/results_python.md`.

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
