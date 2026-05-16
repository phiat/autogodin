# i5d: expose `run_simulations_threaded` — threading is real but GIL-gated

## What landed

- C-ABI export `alphago_mcts_tree_run_simulations_threaded` in
  `odin/alpha_go/exports.odin` (~20 LOC) — thin wrapper around
  `mcts.run_simulations_threaded` from the v0.4.0 vendor.
- Python binding `MCTSTree.run_simulations_threaded(num_sims, n_threads,
  evaluator)` in `python/alpha_go_odin/__init__.py`. CFUNCTYPE callbacks
  auto-acquire the GIL when foreign threads enter Python, so no manual
  PyGILState_Ensure needed on the Odin side. Smoke-tested with 200 sims
  × 4 threads → tree_size=194, root_visits=200.

## Setup

ydh.2-style: 9×9 Go, 1600 sims/move × 32 moves × 3 trials, miniwini host,
post-cz9 build. Two evaluator shapes:

- **Pure Python** uniform evaluator (dict path). GIL held throughout each
  leaf — the *worst case* for threading.
- **GIL-releasing**: same evaluator + `time.sleep(200us)` per leaf, which
  releases the GIL while waiting. Simulates an NN-eval call that yields
  the GIL during a C-extension forward pass.

## Results

### Pure-Python evaluator (GIL held, ydh.2 config)

| n_threads      | sims/sec        | vs sequential |
|----------------|----------------:|--------------:|
| 0 (sequential) | 24,798 ± 652    | 1.00×         |
| 1              | 18,471 ± 208    | 0.74×         |
| 2              | 20,752 ± 112    | 0.84×         |
| 4              | 17,952 ± 579    | 0.72×         |
| 8              | 17,574 ± 426    | 0.71×         |

Threading is a **regression** here. Expected: the Python evaluator holds
the GIL for every leaf — all worker threads serialize on the GIL while
the Odin-side descent/expand/backup work is dwarfed by per-leaf Python
overhead. The threaded path adds tree-mutex overhead, virtual-loss
contention, and worker-pool sync; none of that is amortized when the
real work is GIL-bound.

### GIL-releasing evaluator (`time.sleep(200us)` per leaf)

400 sims/move × 16 moves × 2 trials (smaller config — sequential at
1.0× would take 11+ min/cell otherwise).

| n_threads      | sims/sec        | vs sequential |
|----------------|----------------:|--------------:|
| 0 (sequential) | 2,949 ± 2       | 1.00×         |
| 1              | 2,842 ± 217     | 0.96×         |
| 2              | 6,380 ± 28      | 2.16×         |
| 4              | 11,533 ± 389    | 3.91×         |
| 8              | 12,740 ± 585    | **4.32×**     |

This is the **upper bound**: when the per-leaf Python work *yields* the
GIL, threading the leaves on the Odin side overlaps cleanly. n=1 sits at
0.96× (overhead is negligible when eval cost dominates); n=2/n=4 are
near-linear; n=8 sees diminishing returns from tree-mutex contention.

## Use case

Threading helps under exactly the conditions where the in-process Odin
batched path doesn't (yet) apply:

- The evaluator is a single-leaf Python call that drops into a C
  extension (numpy ops, torch inference, an HTTP-RPC client) which
  releases the GIL.
- Workloads where batching adds unacceptable latency or where the
  evaluator API isn't batched-friendly.

For batched NN-eval workloads, `run_simulations_batched` remains the
right tool (it amortizes per-batch Python overhead, see
`experiments/2026-05-16_13-30-ydh.3-batched-sweep/`).

For pure-Python evaluators, **stay sequential** — the threaded path
loses 25-30%.

## Reproduce

```bash
# Pure-Python (worst case)
PYTHONPATH=python <python> experiments/2026-05-16_16-52-i5d-threaded-mcts/bench.py \
  --trials 3 --threads 1 2 4 8

# GIL-releasing (upper bound)
PYTHONPATH=python <python> experiments/2026-05-16_16-52-i5d-threaded-mcts/bench.py \
  --num-sims 400 --num-moves 16 --trials 2 --threads 1 2 4 8 --sleep-us 200
```
