# Tier-1 rebench: Odin vs C++ MCTS throughput

Date: 2026-05-16 · Bead: autogodin-4fl

## Why this exists

The README's prior throughput numbers were not traceable to a single
committed experiment (the C++ baseline `8,713` does not appear in any
tracked run; the Odin `76,159` was from cz9 but on miniwini in a state
that today reproduces at roughly half-throughput). The README also
admitted we had never run an Odin↔C++ head-to-head with a real NN
evaluator — a gap left over from earlier commit bodies claiming "C++
has no batched API," which is wrong (both backends expose
`run_simulations_batched`).

This experiment resolves both: one host state, one script, both
backends, two workloads (uniform-policy and a real
`SizeInvariantGoResNet`), sequential and batched.

## Setup

- Host: miniwini (12th Gen Intel i7-12650H, 16 cores, 19 GiB RAM)
- Repo at f98071a (6rz)
- Python 3.11.15
- torch 2.11.0+cu130 — run on CPU only (no `.cuda()` calls); `torch.get_num_threads()` = 8
- numpy 2.4.4
- `alpha_go_cpp` is the pybind11 wheel from `~/autogodin-work/autogo/.venv`
- `alpha_go_odin` is the source package, lib at `build/libalpha_go_odin.so` (`-o:speed`)
- Same evaluator body for both backends per workload; only the batched
  return-signature differs (C++ returns `list[(policy, value)]`, Odin
  returns `(list[policy], list[value])`) and is handled by separate
  factory functions in `bench.py`
- Both backends go through their full Python ctypes / pybind11 surface
  — no in-process shortcut

Script: `bench.py`. Canonical config:
```
--uniform-sims 1600 --uniform-moves 32 --uniform-trials 3
--nn-sims 800 --nn-moves 16 --nn-trials 3
--batch 128
```

Raw output: [`runs/miniwini_canonical.log`](runs/miniwini_canonical.log)

## Results

| Cell                     | Rate (sims/sec) | 95% CI | Odin/C++ |
|--------------------------|----------------:|-------:|---------:|
| cpp_seq_uniform          |           6,956 |   ±9   |          |
| odin_seq_uniform         |          23,271 |   ±212 | **3.35×**|
| cpp_seq_nn               |           1,452 |   ±15  |          |
| odin_seq_nn              |           1,086 |   ±16  |   0.75×  |
| cpp_batched_nn bs=128    |           5,666 |   ±162 |          |
| odin_batched_nn bs=128   |           6,932 |   ±140 |   1.22×  |

CIs are 1.96·σ/√n on n=3 trials per cell (after one warmup), so each
ratio's uncertainty is well under ±5% relative.

## What this means

The picture is workload-dependent. Don't take any single ratio as "the
answer."

- **Uniform-policy (3.35×)** is the cleanest measurement of the MCTS
  inner loop itself — the per-leaf evaluator is cheap, so most cycles
  go to tree traversal, expansion, and backprop. Odin's flat-evaluator
  path (cz9) avoids the dict-iteration per leaf that the C++ binding
  pays on every Python callback, and the win shows up here. But this
  cell does *not* model any real training/play workload.

- **Sequential NN (0.75×)** has the per-leaf cost dominated by torch
  forward, not tree work. Here Odin loses: the Python evaluator
  callback runs once per leaf, and the round-trip overhead through
  Odin's ctypes shim is heavier than pybind11's, enough to be visible
  when each callback already costs ~600µs. The 25% deficit is real but
  reflects callback marshalling, not algorithm cost.

- **Batched NN bs=128 (1.22×)** is what matters for self-play. The
  batched API amortises the per-leaf overhead across 128 leaves, and
  Odin pulls ahead again — its leaf-parallel collection (virtual loss
  + batched expansion) appears slightly tighter than the C++ path's,
  enough for a ~22% lift. This is the cell closest to what a training
  loop actually does.

## Honest summary

- For pure tree-walking workloads (e.g. tablebase generation, evaluator
  ablations with cheap policies): Odin is materially faster (~3×).
- For sequential per-leaf NN: Odin is slightly slower (~0.75×); the
  callback overhead dominates and C++ pybind11 wins this micro.
- For batched NN (the real workload): Odin is modestly faster (~1.22×).

We will not headline a single "Odin is Nx faster" number — the ratio
depends entirely on what's in the evaluator.

## Reproducing

Local (laptop with `autogo/.venv-cpponly`):
```sh
PYTHONPATH="autogo/.venv-cpponly/lib/python3.12/site-packages:python:autogo/src" \
  autogo/.venv/bin/python -u experiments/2026-05-16_19-55-cpp-vs-odin-rebench/bench.py
```

Miniwini:
```sh
ssh phiat@miniwini-1.tail08f675.ts.net 'cd ~/autogodin-work/autogodin && \
  PYTHONPATH="$HOME/autogodin-work/autogo/.venv/lib/python3.11/site-packages:$PWD/python:$HOME/autogodin-work/autogo/src" \
  ~/autogodin-work/autogo/.venv/bin/python -u experiments/2026-05-16_19-55-cpp-vs-odin-rebench/bench.py'
```

## Caveats

- All numbers are CPU; we have not yet measured under a GPU evaluator,
  where torch forward latency changes shape and the batched lift would
  likely grow. See bead [[autogodin-ydh.8]] for the GPU run plan.
- `bench.py` runs the bench from a freshly-constructed `MCTSTree` per
  move; no reuse of warmed-up trees across moves. This mirrors the
  current Python API surface, not a hypothetical lower-level test.
- The 0.75× sequential-NN deficit is a known cost of the Python
  callback path; both backends are bottlenecked here on roughly the
  same torch forward time, so the gap is callback marshalling, not
  MCTS proper.
