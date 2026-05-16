# e11 follow-up: Python ctypes batched sweep vs in-process Odin

## Setup

Same config as `bench.odin` / `results.md`, driven from Python through the
new `MCTSTree.run_simulations_batched` ctypes shim added in e11
(commit 5279264). Miniwini, idle, post-FPU vendor v0.3.0
(commit 1d2dd32; threaded.odin landed too but isn't exercised here).

## Raw numbers (sims/sec, mean ± 95% CI, n=3)

Python ctypes path:

| latency | batch=1     | batch=8     | batch=32    | batch=128   |
|---------|-------------|-------------|-------------|-------------|
| 0us     | 19,345 ± 101 | 20,445 ± 194 | 20,539 ± 216 | **22,702 ± 71** |
| 100us   | 4,320 ± 24   | 14,824 ± 16  | 18,146 ± 73  | **21,855 ± 122** |
| 1ms     | 755 ± 2      | 4,915 ± 22   | 11,859 ± 82  | **18,307 ± 347** |

## Comparison vs in-process Odin (bench.odin / ydh.3 results.md)

| latency | batch | in-process | Python ctypes | FFI tax |
|---------|-------|------------|---------------|---------|
| 0us     | 1     | 23,268     | 19,345        | 16.9%   |
| 0us     | 8     | 30,804     | 20,445        | 33.6%   |
| 0us     | 32    | 27,810     | 20,539        | 26.1%   |
| 0us     | 128   | 32,132     | 22,702        | 29.3%   |
| 100us   | 1     | 4,499      | 4,320         | 4.0%    |
| 100us   | 8     | 18,421     | 14,824        | 19.5%   |
| 100us   | 32    | 23,802     | 18,146        | 23.8%   |
| 100us   | 128   | 30,504     | 21,855        | 28.4%   |
| 1ms     | 1     | 771        | 755           | 2.1%    |
| 1ms     | 8     | 4,950      | 4,915         | 0.7%    |
| 1ms     | 32    | 13,593     | 11,859        | 12.8%   |
| 1ms     | 128   | 24,065     | 18,307        | 23.9%   |

## Reading

- **Low latency (0us)**: FFI cost is 17-33%. The Python callback overhead
  per batch (function dispatch, dict allocation, ctypes marshaling) is
  the dominant non-MCTS work. Batching helps amortize but the per-call
  cost is still measurable.

- **Mid latency (100us, NN-on-CPU)**: FFI tax bands 4-28%. At batch=1
  the sleep absorbs everything (4.0% tax). At batch=128 we're back to
  Python-call-overhead-dominated (28.4%).

- **High latency (1ms, slow NN forward)**: FFI tax drops to 0.7-24%.
  At batch=1 and batch=8 the sleep so dominates that the FFI overhead
  is invisible. At batch=128 we still pay 24% — Python dict allocation
  for 128 policies isn't free.

- **For an NN-eval integration**: The Python ctypes shim is essentially
  free at batch sizes that the NN naturally wants (typically 32-256).
  At 1ms forward with batch=128 the cost is ~24%, and at 5ms+ forward
  (more realistic for an untiny net on CPU) the tax will be in single
  digits — the NN forward dominates wall time.

## Followups

- `i5d` (filed): expose `mcts.run_simulations_threaded` from v0.3.0.
  Worth doing once we have a real NN evaluator and want CPU-parallel
  search to coexist with the batched leaf evaluator.
- The Python trampoline's dict iteration in
  `_make_batched_trampoline` is the obvious hot spot at high batch
  sizes — could be replaced with NumPy-buffer mode (write to flat
  ndarrays, skip dict construction). Worth ~10% throughput at
  batch=128 if it ever matters.
