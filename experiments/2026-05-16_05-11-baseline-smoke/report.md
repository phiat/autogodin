# Baseline smoke — autogo env bootstrap + cross-language parity

**Date:** 2026-05-16 (PST)
**Bead:** `autogodin-ydh.1` (partial; GPU training step deferred)
**Host:** CPU-only Linux box (16 cores, 19 GB RAM, no GPU)

## What was done

Phase A of `ydh.1`: bootstrap the autogo Python + C++ environment on a fresh CPU host and prove the cross-language parity contract end-to-end. Phase B (actually completing `run_iteration.sh 0 1` with a real GPU training step) is filed as a follow-up.

Steps:

1. `curl https://astral.sh/uv/install.sh | sh` → uv 0.11.14.
2. `git clone https://github.com/ericjang/autogo` (public; phiat/autogodin is our repo).
3. `uv sync` (~3 min, downloaded torch 2.11+cu130, scipy, pandas, ...).
4. `cmake` direct invocation building `alpha_go_cpp.cpython-311-...so`. Standard `scripts/build_cpp.sh` hardcodes `libpython3.10.so`; bypassed by computing the version from `sys.version_info` and passing `-DPython3_LIBRARY=...` manually.
5. `uv run python -m pytest tests/ --ignore=tests/test_gpu_lease.py` → **101 passed, 24 skipped** (the skipped tests are GPU-gated; the ignored file is broken upstream — see friction below).
6. `rsync` the autogodin tree (sans `.git`, `autogo/`, `.beads/`) → miniwini.
7. Run `python/parity/random_games_dual.py --backend both` (added under this bead) which runs the same seeded random-game harness against `alpha_go_odin` (ctypes shim) and `alpha_go_cpp` (pybind11), comparing per-move board content via sha1.

## Key result — cross-language parity

```
{
  "odin": "109bd08aa3578ec029a3342a1f3749f4b42fa1b9d93113fde94a1e84d6a0994c",
  "cpp":  "109bd08aa3578ec029a3342a1f3749f4b42fa1b9d93113fde94a1e84d6a0994c",
  "match": true,
  "games": 10, "max_moves": 200, "size": 9
}
```

10 games × ~200 moves each = ~2000 move-level state comparisons, all matching. This is the contract the Odin port was built to satisfy. `autogodin-3xv.12` was committed against a self-consistency fingerprint; this report upgrades it to cross-language.

## Environment friction (notes for future runs)

- **`scripts/build_cpp.sh` hardcodes `libpython3.10.so`.** uv's auto-installed Python on a fresh host is 3.11.15, so the script's `sysconfig.get_config_var('LIBDIR') + 'libpython3.10.so'` resolves to a non-existent path and CMake errors out with "Could NOT find Python3 (missing: Development...)". Workaround: compute the version dynamically (`f"libpython{sys.version_info.major}.{sys.version_info.minor}.so"`) or pass `-DPython3_LIBRARY=...` directly to cmake. Worth a PR upstream.
- **`tests/test_gpu_lease.py` is broken upstream.** It imports `_read_priorities` and `_write_priorities_stub` from `infra.remote_exec`, neither of which exists in the current `remote_exec.py`. Skipped via `--ignore`. Not fatal, but a real test gap.
- **`pyproject.toml` requires-python = ">=3.10"** — works fine on 3.11; 3.12 untested.

## Timing

- uv sync: ~3 min.
- CMake configure + build: ~30 sec (12 cores, `cmake --build -j12`).
- pytest 101 tests: 2.75 s.
- Parity harness (10 games, both backends): ~1.5 s.

End-to-end from a clean host: ~5 min, no GPU.

## What's NOT covered here (and where it goes)

| Item | Bead |
|------|------|
| `pre_collect_random.py` exercise on CPU | will run as part of GPU-side smoke (`ydh.1` Phase B) |
| `run_iteration.sh 0 1` actual training (iter0 + iter1) | needs GPU — pending user approval per `compute-gpu-policy` |
| MCTS C++ vs Odin throughput bench | `autogodin-ydh.2` (unblocked) |
| Self-play A/B Odin vs C++ at fixed sims | `autogodin-ydh.4` (unblocked) |
| Upstream patch for hardcoded `libpython3.10` | not filed; mention if we send a PR |

## Artifacts

- `python/parity/random_games_dual.py` — the dual-backend harness.
- This report.
- All beads updated; `bd ready` queue rolls forward to `ydh.2`.
