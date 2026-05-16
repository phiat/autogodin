# autogodin

An Odin port of the C++ MCTS + Go-board hot path from
[ericjang/autogo](https://github.com/ericjang/autogo), exposed to Python
via ctypes as a drop-in replacement for upstream `alpha_go_cpp`.

Correctness checks that we run against the upstream C++:

- **Board parity**: SHA-256 Zobrist fingerprint over 10 seeded games × ~200 moves matches byte-for-byte. `just parity` re-runs the check against the committed fixture.
- **MCTS strength**: 100 MCTS games at 200 sims/move with a random-init `SizeInvariantGoResNet(32ch × 4b)` shared between backends — Odin 53 / C++ 47 / 0 draws, Wilson 95% CI [0.433, 0.625] brackets 0.5. Raw data in `experiments/2026-05-16_18-41-7v8-nn-strength-ab/`.

Throughput characterizations live under `experiments/` per change. We don't put summary numbers in this README — they shift with host state, and the per-experiment reports are the source of truth.

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
