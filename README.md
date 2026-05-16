# autogodin

An Odin port of the C++ MCTS + Go-board hot path from
[ericjang/autogo](https://github.com/ericjang/autogo), exposed to Python
via ctypes as a drop-in replacement for upstream `alpha_go_cpp`.

What we've measured:

- **Byte-identical** board behaviour vs the C++ reference (SHA-256 Zobrist fingerprint over 10 seeded games × ~200 moves).
- **Equal strength** under a random-init `SizeInvariantGoResNet(32ch × 4b)` evaluator over 100 MCTS games at 200 sims/move (Odin 53 / C++ 47; Wilson 95% CI [0.433, 0.625] brackets 0.5).

Throughput is workload-dependent. We have numbers on a uniform-policy micro-bench (where MCTS-internal work dominates) and on a CPU NN-eval grid in isolation. We have **not** run a head-to-head Odin-vs-C++ comparison with a real NN evaluator; the realistic-workload ratio is therefore unknown. See [Throughput](#throughput) for what we did measure and how to read it.

## Quick example

```python
from alpha_go_odin import GoBoard, MCTSTree, MCTSConfig, PASS_ACTION

cfg = MCTSConfig()
cfg.c_puct = 1.0
cfg.temperature = 1.0

tree = MCTSTree(GoBoard(9, 7.5), cfg)

def uniform(board):
    legal = board.get_legal_moves_flat()
    p = 1.0 / (len(legal) + 1)
    policy = {a: p for a in legal}
    policy[PASS_ACTION] = p
    return policy, 0.0  # (policy_dict, value)

tree.run_simulations(100, uniform)
print("Chosen move:", tree.select_action(temperature=0.0))
```

Drop-in replacement for the upstream `alpha_go_cpp` pybind11 module:
`scripts/run_with_odin_backend.sh <your-python-command>` reroutes
`import alpha_go_cpp` to this backend with zero source changes upstream.

## What's here

```
odin/alpha_go/          GoBoard + Game-vtable adapter + C-ABI exports
odin/vendor/mcts-odin/  Pinned vendor of mcts-odin (algorithm core)
odin/tests/             Odin @(test) procs (37 cases)
python/alpha_go_odin/   ctypes wrapper mirroring alpha_go_cpp's OO API
python/odin_backend/    `import alpha_go_cpp` → alpha_go_odin shim
python/parity/          Zobrist-fingerprint parity harness
scripts/                build, parity, autogo-setup, backend-swap shims
experiments/            self-contained benches + reports per change
```

## Status

The Go-board port, the vendored MCTS core, the C-ABI export surface, and the Python ctypes shim (including batched and threaded paths) are complete and have cleared the parity and strength gates documented below. Current work focuses on training A/B experiments and optional GPU runs.

- GoBoard: Zobrist-incremental positional superko, KataGo-aligned no-suicide rule, Tromp-Taylor area scoring.
- MCTS: vendored from [mcts-odin](https://github.com/phiat/mcts-odin) (`odin/vendor/mcts-odin/`, pinned commit; see `VERSION`). Packed-slot nodes, branchless PUCT, linear-space priors, FPU (parent-Q with reduction), per-tree scratch arena, leaf-parallel batched with virtual loss, Dirichlet noise, PCR, subtree reuse, root-parallel threading. The local `go_adapter.odin` is a ~140-LOC Game vtable bridging GoBoard.
- 37/37 Odin `@(test)` cases pass clean under the memory tracker.
- 43 `alphago_*` C-ABI symbols in `libalpha_go_odin.so`. Python ctypes shim mirrors upstream `alpha_go_cpp`'s OO API plus `run_simulations_batched` (leaf-parallel + virtual loss), `run_simulations_batched_flat` (no-dict scratch-ndarray evaluator; `cg0`), `run_simulations_threaded` (root-parallel worker pool), and `run_simulations_flat` (no-dict sequential; `cz9`).

### Correctness

- **Board parity** (`python/parity/random_games_dual.py`): Odin and upstream C++ produce a byte-identical SHA-256 fingerprint `109bd08a…` over 10 seeded games × ~200 moves.
- **MCTS-layer A/B under a real NN evaluator** (`7v8`): 100 games of Odin-MCTS vs C++-MCTS at 200 sims/move, `SizeInvariantGoResNet(32ch × 4b)` random-init evaluator passed to both backends. **Result: Odin 53 – C++ 47 – 0 draws, Wilson 95% CI [0.433, 0.625] brackets 0.5.** Parity-complete under realistic priors. See `experiments/2026-05-16_18-41-7v8-nn-strength-ab/`.

### Throughput

**Read this section carefully.** The headline ratios below are from a synthetic micro-bench (uniform-policy + value 0). In that regime per-leaf evaluator cost is essentially zero, so MCTS-internal cost dominates and any speed difference in the MCTS implementation gets fully exposed. In a realistic workload the per-leaf cost is the NN forward, which dominates so completely that the MCTS-implementation delta gets amortized away — the ratio you'd see in production is smaller, possibly close to 1×. **We have not measured the realistic-workload ratio**; running a head-to-head C++ comparison under a real NN evaluator is open follow-up work.

9×9 micro-bench: 1600 sims/move × 32 moves, single-thread, miniwini host, vendor v0.4.0. Evaluator: uniform-policy + value 0. Both backends invoked through the same Python callback signature.

| Backend                                                      | sims/sec        | vs C++  |
|--------------------------------------------------------------|-----------------|---------|
| `alpha_go_cpp` (upstream)                                    | 8,713 ± 66      | 1.00×   |
| `alpha_go_odin` Python ctypes shim, legacy dict evaluator    | 48,019 ± 307    | 5.51×   |
| `alpha_go_odin` Python ctypes shim, flat evaluator (`cz9`)   | 54,541 ± 616    | 6.26×   |

Additional Odin-only regime (no Python callback in the leaf — available only if your evaluator is also Odin-side; upstream's pybind11 surface has no equivalent so this row has no C++ counterpart):

| Backend                                                      | sims/sec        |
|--------------------------------------------------------------|-----------------|
| `alpha_go_odin` in-process Odin evaluator                    | 76,159 ± 481    |

<details>
<summary><b>Batched / threaded / NN-eval grids</b></summary>

**Batched evaluator** (`run_simulations_batched`, in-process key cells; full grid in `experiments/2026-05-16_13-30-ydh.3-batched-sweep/`):

| latency      | batch=1 | batch=128 | speedup | Python tax |
|--------------|---------|-----------|---------|------------|
| 0us          | 62,482  | 87,281    | 1.4×    | 40%        |
| 100us        | 5,313   | 78,889    | 14.9×   | 37%        |
| 1ms          | 828     | 47,284    | **57×** | 23%        |

**Threaded evaluator** (`run_simulations_threaded`):

| evaluator                          | n=0 (seq) | n=2    | n=4    | n=8    | best speedup |
|------------------------------------|----------:|-------:|-------:|-------:|-------------:|
| Pure Python (GIL held)             | 24,798    | 20,752 | 17,952 | 17,574 | 0.84× (regress) |
| Python + `time.sleep(200µs)` (i5d) | 2,949     | 6,380  | 11,533 | 12,740 | **4.32×** |
| Real torch CPU forward (441)       | 1,408     | 588    | 105    | 68     | 0.05× (collapse) |

The threaded path adds tree-mutex / virtual-loss / worker-pool overhead that's a net loss when the evaluator holds the GIL. The sleep cell shows the path is real and scales when the evaluator yields the GIL for its entire body. The real-torch cell shows it does NOT scale for actual NN evaluators — torch's CPU dispatcher serializes across Python threads on top of the per-leaf GIL hold. **For Python NN evaluators, use `run_simulations_batched` (5.8× at bs=128) — not threaded.**

**Real CPU NN-eval** (`SizeInvariantGoResNet 32ch×4b`, post-441):

| path                       | sims/sec    | vs seq |
|----------------------------|------------:|-------:|
| seq dict / flat            | ~1,294      | 1.00×  |
| batched dict bs=128        | 7,503       | 5.8×   |
| batched flat bs=128 (cg0)  | **7,888**   | **6.1×** |

Sequential progression: pre-foundation 2,859 / pre-vendor 7,927 / pre-FPU 13,613 / post-FPU 25,618 / post-`zkq` 69,234 / post-`373` 74,899 → post-`5km` 76,159 in-process. The `ydh.6` perf profile identified three hot-paths in legality; all three fixed: `zkq` (in-place legality probe), `373` (early-exit capture probe), `5km` (caller-owned legal-moves buffer). Batched table re-run post-`ydh.6`: 1ms × batch=128 went 24,065 → 47,284 sims/sec.
</details>

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: `just check` is the pre-push gate; MCTS algorithm work goes upstream at [phiat/mcts-odin](https://github.com/phiat/mcts-odin), not here; perf claims need a host + CI.

## License

MIT (this repo, see `LICENSE`). Vendored mcts-odin is MIT (`odin/vendor/mcts-odin/LICENSE`). All algorithms ported here are direct translations of [ericjang/autogo](https://github.com/ericjang/autogo) (MIT, Copyright Eric Jang); this repo's contribution is the Odin port + ctypes shim + parity tooling.
