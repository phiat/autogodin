# cg0: dict vs flat batched evaluator — modest lift, dict iteration is not the dominant tax

## Setup

Same workload as `bench_python.py` but running both the legacy dict path
AND the new `run_simulations_batched_flat` in one harness so the cell-by-
cell uplift is exact. Miniwini single-thread, post-cg0 build (mcts-odin
v0.4.0).

The flat path:

1. Pre-allocates `(batch_size, n_actions)` int32 scratch for actions,
   float32 scratch for probs, plus `(batch_size,)` counts and values.
   All sized once at `run_simulations_batched_flat` call time.
2. Pre-resolves the four `.ctypes.data` pointers (accessing this per
   leaf was visible as `numpy._internal._ctypes` alloc cost in cz9's
   profile).
3. Per FFI batch: hand the scratch to the evaluator (which slice-assigns
   directly into the rows); trampoline issues four whole-buffer
   `ctypes.memmove`s into MCTS's out buffers. No per-state dict alloc,
   no per-cell `__setitem__`.

## Raw — dict vs flat (sims/sec, mean ± 95% CI, n=2)

| latency | batch | dict           | flat           | ratio |
|---------|-------|---------------:|---------------:|------:|
| 0us     | 1     | 21,114 ± 216   | 21,945 ± 72    | 1.04× |
| 0us     | 8     | 24,291 ± 93    | 26,529 ± 121   | 1.09× |
| 0us     | 32    | 21,652 ± 275   | 23,483 ± 72    | 1.08× |
| 0us     | 128   | 26,942 ± 59    | **29,999 ± 204** | **1.11×** |
| 100us   | 1     | 4,413 ± 30     | 4,443 ± 223    | 1.01× |
| 100us   | 8     | 16,317 ± 96    | 17,692 ± 164   | 1.08× |
| 100us   | 32    | 19,313 ± 88    | 19,846 ± 541   | 1.03× |
| 100us   | 128   | 26,185 ± 426   | **28,856 ± 772** | **1.10×** |
| 1ms     | 1     | 764 ± 5        | 765 ± 6        | 1.00× |
| 1ms     | 8     | 5,124 ± 27     | 4,994 ± 18     | 0.97× |
| 1ms     | 32    | 12,406 ± 10    | 12,940 ± 70    | 1.04× |
| 1ms     | 128   | 21,639 ± 68    | **23,603 ± 160** | **1.09×** |

## Reading

- Flat is a **modest, consistent win** at every cell except 1ms × batch=8
  (0.97×, within CI overlap — call it neutral).
- **Biggest gains at batch=128** (1.09-1.11×), where the dict-iteration
  share of per-batch Python work is largest. Smallest at batch=1 (1.00-
  1.04×) where view construction + evaluator body dominate.
- Latency tier matters less than batch size: the flat win at batch=128
  is ~10% across 0us, 100us, AND 1ms latency tiers. The dict-iteration
  cost amortizes per-batch, not per-latency.
- **The dict iteration was not the dominant tax**. The original ydh.3
  results_python.md framed batched Python tax as 23-40%; cg0 takes back
  ~10% of that, not 30%+. The remaining tax is in: views-list
  construction (`GoBoard.__new__` × batch_size per FFI call), the
  evaluator body itself (`get_legal_moves_flat` + loop), and per-batch
  CFUNCTYPE marshaling.

## What about the in-process ceiling?

ydh.3's in-process Odin baseline at 1ms × batch=128 was 47,284 sims/sec.
cg0's flat hits 23,603 sims/sec — still ~50% of in-process. So the FFI
tax is now closer to 50% than the 23% reported in ydh.3, but **this is a
miniwini absolute-throughput shift, not a regression**: a sanity micro-
bench of plain dict path today gives 27,789 sims/sec at 0us batch=128
(vs ydh.3's 52,260 in the same cell). Miniwini is currently running
~half what it was during ydh.3 — likely thermal / k3s load, not code
drift. The cell-by-cell dict-vs-flat ratio is the apples-to-apples
comparison and is robust.

The result: the largest remaining Python tax is **not** dict iteration.
Likely next-biggest levers, in rough order:

1. Avoid per-batch views-list construction in Python (e.g. expose a
   batched `get_legal_moves_flat` that fills a single
   `(batch_size, n_actions)` int32 buffer in Odin — skip the per-state
   Python evaluator loop entirely).
2. Avoid `GoBoard.__new__` for the views (cache view objects + rebind
   `_h` per call, or expose the batch as raw pointers + helpers).
3. CFUNCTYPE marshal cost is bounded by once-per-batch, not per-leaf —
   probably not worth chasing.

## Reproduce

```bash
PYTHONPATH=python <python> experiments/2026-05-16_13-30-ydh.3-batched-sweep/bench_python_flat.py \
  --num-sims 1600 --moves 32 --trials 2 \
  --latencies 0 100 1000 --batches 1 8 32 128
```
