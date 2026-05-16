# 441: CPU-only NN-eval A/B — batched works; threaded doesn't

## What landed

Wraps a randomly-initialized `SizeInvariantGoResNet(channels=32,
n_blocks=4, value_hidden=32)` — 76k params, the same architecture
ydh.5 actually trained — as a sequential / batched (dict + flat) /
threaded evaluator and benches each path. 9×9 Go, 800 sims/move ×
16 moves × 2 trials, miniwini, post-cg0 build.

This is the throughput half of the 441 deliverable. The 100-game
Odin-vs-C++ strength A/B under informative priors is filed separately
as `autogodin-7v8` — it reuses this NN evaluator wrapper.

## Results (sims/sec, mean ± 95% CI, n=2)

**Sequential** (torch threads = 8, default; one leaf per simulation):

| path               | sims/sec  | vs seq dict |
|--------------------|----------:|------------:|
| seq dict eval      | 1,294 ± 17 | 1.00×       |
| seq flat eval (cz9)| 1,270 ± 15 | 0.98×       |

Sequential dict ≈ flat. With a real NN forward dominating per-leaf
cost (~0.7 ms), the dict-iteration share of work is too small to
recover from the flat path. cz9's 10.9% sequential lift was on a
uniform-prior bench where the evaluator body cost ~0.

**Batched** (leaf-parallel + virtual loss; torch threads = 8):

| path                       | sims/sec    | vs seq | vs in-batch dict |
|----------------------------|------------:|-------:|-----------------:|
| batched dict bs=8          | 3,448 ± 54  | 2.7×   | —                |
| batched dict bs=32         | 5,921 ± 97  | 4.6×   | —                |
| batched dict bs=128        | 7,503 ± 67  | 5.8×   | 1.00×            |
| batched flat bs=32 (cg0)   | 6,010 ± 139 | 4.6×   | 1.02× (vs bs=32 dict) |
| batched flat bs=128 (cg0)  | **7,888 ± 162** | **6.1×** | **1.05× (vs bs=128 dict)** |

**Batching is the win**: 5.8-6.1× at bs=128 vs sequential, on a real
torch forward. cg0's flat path adds another 2-5% on top — consistent
with cg0's miniwini sweep (1.09-1.11× there).

**Threaded** (root-parallel; torch threads forced to 1):

| path                              | sims/sec    | vs torch=1 baseline |
|-----------------------------------|------------:|--------------------:|
| seq dict baseline (torch=1)       | 1,408 ± 7   | 1.00×               |
| threaded n=1                      | 1,343 ± 9   | 0.95×               |
| threaded n=2                      | 588 ± 12    | **0.42× — regression** |
| threaded n=4                      | 105 ± 0     | **0.07× — collapse** |
| threaded n=8                      | 68 ± 4      | **0.05× — collapse** |

**The threaded path fails with a real Python NN evaluator** — not a
mild regression, a catastrophic one. This is the opposite of i5d's
sleep-evaluator finding (which showed 4.3× at n=8).

### Why threading collapses with torch but not with sleep

The i5d threaded bench showed 0.96× at n=1 and 4.3× at n=8 with
`time.sleep(200µs)` per leaf. Real torch CPU forward at the same
~200-700µs latency regresses *catastrophically*. Two compounding
effects:

1. **Python-side eval body holds the GIL**. The trampoline acquires
   the GIL on entry; the evaluator does `board.to_numpy()`, builds an
   int64 plane, builds an output dict iterating over `legal` actions.
   For 81 actions × ~half the per-leaf time, every MCTS worker
   serializes on the GIL for the non-torch portion of each call.
   `time.sleep` releases the GIL for its entire duration; torch
   releases it only inside the kernel dispatch.

2. **torch CPU kernel dispatch is not contention-free across threads**
   even at `set_num_threads(1)`. Multiple Python threads calling
   `module.forward` concurrently appear to serialize through PyTorch's
   internal dispatcher / allocator, on top of the GIL-held Python work.
   The n=4 / n=8 collapse below n=1 baseline (0.07× / 0.05×) is far
   worse than pure GIL serialization would predict — suggesting active
   thread oversubscription or context-switch thrashing.

### Practical conclusion

For Python-NN evaluators (the realistic Phase 3 path):

- **Use batched**: 5.8× at bs=128 over sequential, real and reproducible.
- **Do NOT use threaded with a Python evaluator**. The `run_simulations_threaded`
  API is for in-process Odin evaluators or for evaluators that release
  the GIL during their *entire* body (rare; even numpy ops hold the GIL
  for the loop-over-batch portion).

The threaded path is not broken — it just has a narrow sweet spot
(i5d's sleep upper bound, or in-process Odin evals). Stating that
clearly in the README under run_simulations_threaded is on the to-do
list.

## What's NOT in this report

- **Strength A/B** (100 games Odin-MCTS vs C++-MCTS at 200 sims/move,
  random-init NN evaluator). Filed as autogodin-7v8 (cheap, CPU-only,
  ~1 hour on miniwini) — the last open correctness gate before public
  release per the wrap-up audit.

## Reproduce

```bash
# miniwini (build per setup_autogo.sh, plus 'uv pip install torch mup rich')
PYTHONPATH="python/odin_backend:python:../autogo/src" \
  .venv/bin/python experiments/2026-05-16_18-05-441-cpu-nn-eval/bench.py \
  --num-sims 800 --moves 16 --trials 2
```
