# ydh.3 results: batched MCTS throughput sweep

## Setup

In-process Odin bench (no Python in the loop). Same workload as 4ig
(1600 sims/move × 32 moves × 3 trials, 9x9 Go, uniform-prior eval,
value=0, c_puct=1.0, lambda=0, no Dirichlet, post-FPU vendor v0.2.1).
Host: miniwini, idle, single-thread, -o:speed.

Two axes:
- `batch_size`: 1, 8, 32, 128
- per-leaf evaluator `latency`: 0us, 100us, 1ms (synthetic `time.sleep`)

Skipped 10ms latency — at batch=1 it would take ~25 min/cell; amortization
pattern is monotonic so the trend is already obvious by 1ms.

## Results (sims/sec, mean ± 95% CI, n=3)

| latency  | batch=1     | batch=8     | batch=32    | batch=128   |
|----------|-------------|-------------|-------------|-------------|
| 0us      | 23,268 ± 110 | 30,804 ± 177 | 27,810 ± 329 | **32,132 ± 205** |
| 100us    | 4,499 ± 29   | 18,421 ± 83  | 23,802 ± 227 | **30,504 ± 178** |
| 1ms      | 771 ± 2      | 4,950 ± 40   | 13,593 ± 134 | **24,065 ± 132** |

Reference (single-state, no batching, from 4ig): 25,618 ± 86 sims/sec.

## What this says

### Amortization works as the leaf-parallel design predicts

- At 0us latency (no eval cost), batching adds ~30% throughput over
  sequential. The win is locality + reduced per-call MCTS bookkeeping.
- At 100us latency, batch=128 gives **6.8× speedup** over batch=1.
- At 1ms latency, batch=128 gives **31× speedup** over batch=1.

For comparison: pure-sleep cost at batch=1 with 1ms eval is 51,200 ×
1ms = 51 sec per trial, so the bench is essentially measuring sleep at
that cell (771 sims/s ≈ 51,200 / 66s). Batch=128 hides 99.6% of that
sleep behind 1/128th as many evaluator calls.

### batch=32 is sometimes slower than batch=8

Visible at 0us latency (27,810 vs 30,804) and is a small effect. Likely
tree-contention: at batch=32 the leaves picked under virtual-loss
overlap more, so backups thrash a small tree more than they amortize.
Not a real problem for production — NN-eval latencies move the sweet
spot rightward (batch=128 dominates everywhere ≥100us).

### Practical takeaway for an NN evaluator integration

A small-to-medium policy/value net on CPU typically runs in 1-5 ms per
position. Picking that operating point on the table above:

- batch=128 + 1ms eval ≈ 24k sims/sec
- batch=128 + 5ms eval ≈ 5k sims/sec (extrapolated linearly from the
  1ms→batch=128 cell, since at high latency throughput is
  sleep-dominated)

These are *raw MCTS sims*, not NN-positions. The NN runs ~400 forward
passes per move (3200 sims / batch=128 × 32 moves / 32 moves ≈ 400 per
move averaged), each forward batched to 128 positions. At 1ms per
forward pass, that's 400ms NN time per move — comfortably under most
real-time targets.

## Reproduce

```bash
cd ~/autogodin-work/autogodin
odin build experiments/2026-05-16_13-30-ydh.3-batched-sweep \
    -o:speed \
    -out:experiments/2026-05-16_13-30-ydh.3-batched-sweep/bench
./experiments/2026-05-16_13-30-ydh.3-batched-sweep/bench
```

Total wall time: ~5 min on miniwini, dominated by the latency=1ms × batch=1
cell (~3 min for 3 trials).

## Followups

- Wire `mcts.run_simulations_batched` through the C-ABI / Python ctypes
  shim. Currently only the sequential `run_simulations` is exposed via
  `alphago_mcts_tree_run_simulations`. Required for Python-driven NN
  evaluators. (Filing as a P2 follow-up.)
- Once that's in place, re-run this sweep through the Python shim. The
  delta vs the in-process numbers here is the FFI cost at each batch
  size, analogous to 4ig but for batched.
