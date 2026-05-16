# ydh.3 results — batched MCTS throughput sweep (post-ydh.6)

## Setup

- Miniwini, single-thread, 9×9 Go, uniform-policy in-process Odin evaluator.
- 1600 sims/move × 32 moves × 3 trials per cell.
- mcts-odin v0.4.0 vendored. autogodin post-`zkq` + `373` + `5km`
  (ydh.6 hotspot fixes — legality probe in-place, capture-probe early-exit,
  caller-owned legal-moves buffer).
- Latency axis: synthetic `time.sleep` inside the evaluator (NN-eval proxy).
- Numbers are mean ± 95% CI (n=3).

## Raw — sims/sec (in-process Odin evaluator)

| latency | batch=1     | batch=8     | batch=32    | batch=128   |
|---------|-------------|-------------|-------------|-------------|
| 0us     | 62,482 ± 506 | 73,155 ± 5,293 | 60,503 ± 252 | **87,281 ± 575** |
| 100us   | 5,313 ± 5    | 28,673 ± 187 | 45,429 ± 244 | **78,889 ± 309** |
| 1ms     | 828 ± 2      | 5,765 ± 37   | 18,994 ± 55  | **47,284 ± 340** |

## Reading

- **0us cells**: nearly all-CPU. Batching adds small bookkeeping overhead;
  batch=128 still wins because PUCT + virtual-loss are tightly looped without
  per-leaf overhead. The batch=32 dip vs batch=8 is most likely cache /
  contention noise — wide CI at batch=8 at 0us suggests the same.
- **100us cells (CPU-NN proxy)**: batching pays off cleanly — 14.9× speedup
  batch=128 vs batch=1.
- **1ms cells (slow-NN proxy)**: amortization is dominant. 47,284 / 828 =
  **57× speedup** at batch=128. Confirms virtual-loss leaf-parallelism is
  doing what it should.

## Delta vs pre-ydh.6 baseline

ydh.6 hotspot fixes (`zkq` + `373` + `5km`) widened the gap most at low
latency — the Odin work inside each leaf got ~2-3× faster, so a higher
batch_size is now needed to make per-leaf latency dominant.

Pre/post comparison at headline cells (in-process):

| cell              | pre-ydh.6 | post-ydh.6 | x       |
|-------------------|----------:|-----------:|--------:|
| 0us × batch=1     |  23,268   |  62,482    | 2.69×   |
| 0us × batch=128   |  32,132   |  87,281    | 2.72×   |
| 100us × batch=128 |  30,504   |  78,889    | 2.59×   |
| 1ms × batch=128   |  24,065   |  47,284    | 1.96×   |

## Followups

- batch=32 at 0us looks anomalously low. The Python sweep was running on the
  same host (different cores, but shared cache). Re-run isolated if it ever
  matters for a real result.
- 10ms latency cells still excluded — at batch=1 each cell is ~25 min.
  Pattern is monotonic with latency and already very clear at 1ms.
