# e11 follow-up: Python ctypes batched sweep vs in-process Odin (post-ydh.6)

## Setup

Same workload as `bench.odin` / `results.md`, driven from Python through
the `MCTSTree.run_simulations_batched` ctypes shim. Miniwini single-thread,
mcts-odin v0.4.0, autogodin post-ydh.6 (`zkq` + `373` + `5km`).

## Raw — Python ctypes path (sims/sec, mean ± 95% CI, n=3)

| latency | batch=1     | batch=8     | batch=32    | batch=128   |
|---------|-------------|-------------|-------------|-------------|
| 0us     | 38,289 ± 127 | 44,268 ± 401 | 38,487 ± 297 | **52,260 ± 462** |
| 100us   | 5,075 ± 24   | 24,463 ± 106 | 34,704 ± 313 | **49,574 ± 860** |
| 1ms     | 831 ± 11     | 5,747 ± 501  | 16,400 ± 288 | **36,492 ± 120** |

## Comparison vs in-process Odin (same host, same run)

| latency | batch | in-process | Python ctypes | FFI tax |
|---------|-------|-----------:|--------------:|--------:|
| 0us     | 1     |  62,482    |  38,289       | 38.7%   |
| 0us     | 8     |  73,155    |  44,268       | 39.5%   |
| 0us     | 32    |  60,503    |  38,487       | 36.4%   |
| 0us     | 128   |  87,281    |  52,260       | 40.1%   |
| 100us   | 1     |   5,313    |   5,075       | 4.5%    |
| 100us   | 8     |  28,673    |  24,463       | 14.7%   |
| 100us   | 32    |  45,429    |  34,704       | 23.6%   |
| 100us   | 128   |  78,889    |  49,574       | 37.2%   |
| 1ms     | 1     |     828    |     831       | -0.4%   |
| 1ms     | 8     |   5,765    |   5,747       | 0.3%    |
| 1ms     | 32    |  18,994    |  16,400       | 13.7%   |
| 1ms     | 128   |  47,284    |  36,492       | 22.8%   |

## Reading

- **Low latency (0us)**: FFI tax 36-40%. Now larger than pre-ydh.6 (was
  17-33%) because the Odin work got 2.7× faster but the Python callback
  per-batch overhead (function dispatch, dict allocation, ctypes marshaling)
  didn't change — its relative share grew.
- **Mid latency (100us, CPU-NN proxy)**: FFI tax 4.5-37%. At batch=1 the
  sleep absorbs everything (4.5%); at batch=128 the per-batch Python work
  is the dominant non-MCTS cost (37%).
- **High latency (1ms, slow-NN proxy)**: FFI tax 0-23%. At batch=1 and
  batch=8 the per-leaf sleep so dominates that FFI overhead is invisible.
  At batch=128 we still pay 23% — Python dict allocation for 128 policies
  isn't free.
- **For an NN-eval integration**: the Python ctypes shim costs ~23% at
  batch=128 with a 1ms forward pass. At 5ms+ forward (more realistic for a
  non-tiny net on CPU) the tax drops to single digits; on GPU with a real
  net the NN forward dominates and ctypes is essentially free.

## Followups

- `cz9` (P2 open): replace per-leaf Python-dict result with flat-array
  C-ABI. ~10% throughput at batch=128 if it ever matters — the larger
  share of FFI tax post-ydh.6 makes this more attractive than it was.
- `i5d` (P3 open): expose `mcts.run_simulations_threaded`. Worth doing
  once we have a real NN evaluator and want CPU-parallel search to coexist
  with the batched leaf evaluator.
