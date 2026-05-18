# bpoC-rerun-cpp: end-to-end C++ vs Odin backend comparison

**Date:** 2026-05-17 · **Beads:** [autogodin-ycy](bd:autogodin-ycy)
(parent), [autogodin-bz6](bd:autogodin-bz6) (training),
[autogodin-hkb](bd:autogodin-hkb) (gauntlet)

## TL;DR

Same training recipe (bpoC PATH C: 5k random pre-collect + 4× selfplay
→ train iters, 256ch×10b ResNet, 200 sims/move, 500 games/iter)
through two MCTS backends:

| metric | Odin shim (postfix run) | C++ pybind (this run) |
|---|---|---|
| wall-clock (JL L4) | 106 min | 225 min — **2.1× slower** |
| iter4 train_value_acc | **99.34%** | 70.31% |
| iter4 play strength (this gauntlet) | **0 wins** | **21 wins** |

The Odin run's "amazing" value_acc was Goodharted: it trained BEFORE the
[autogodin-6qt](bd:autogodin-6qt) fix (f21a3cd, this session), so its
selfplay games were over-deterministic and the value head fit them
trivially. The C++ run had real Dirichlet entropy
(`std::random_device`) so its selfplay games were genuinely random; the
70% value_acc on harder data is the honest signal. The gauntlet
confirms it: the model trained on honest data **plays better** by an
overwhelming margin.

## Setup

Both runs on JarvisLabs L4 IN2 ($0.44/hr), identical recipe and code
except for the MCTS backend during selfplay-data-generation:

- **Postfix run** (`experiments/2026-05-17_07-40-bpoC-rerun-postfix/`):
  PYTHONPATH includes `python/odin_backend` so `alpha_go_cpp` is
  aliased to `alpha_go_odin`. Ran at HEAD before f21a3cd, so every
  MCTSTree(...) used the buggy `seed=0` default.
- **C++ run** (`experiments/2026-05-17_16-00-bpoC-rerun-cpp/`):
  PYTHONPATH excludes the shim. Pre-flight assertion verified
  `alpha_go_cpp.MCTSTree is not alpha_go_odin.MCTSTree` at startup.

Gauntlet on **miniwini** (CPU-only) with both iter4 ckpts evaluated
through real `alpha_go_cpp` pybind11 (no shim — startup assertion enforced):
- 21 games (stopped early at user request; Wilson CI already conclusive)
- Alternating colors (11 ODIN-Black, 10 ODIN-White)
- 200 sims/move, dirichlet=0, temperature=1.0 for first 10 moves then argmax
- Move cap 200; mean game length 161

## Training wall-clock (head-to-head)

| stage | C++ run | Odin run | ratio |
|---|---:|---:|---:|
| pre_collect (random) | 86s | 85s | 1.0× |
| train iter0 | 922s | 919s | 1.0× ✓ |
| selfplay it0 | 3574s | 1359s | 2.6× |
| train iter1 | 114s | 71s | 1.6× |
| selfplay it1 | 2801s | 1308s | 2.1× |
| train iter2 | 192s | 131s | 1.5× |
| selfplay it2 | 2584s | 1083s | 2.4× |
| train iter3 | 266s | 154s | 1.7× |
| selfplay it3 | 2618s | 1095s | 2.4× |
| train iter4 | 339s | 155s | 2.2× |
| **TOTAL** | **225 min** | **106 min** | **2.13×** |

The end-to-end Odin advantage is larger than the cpp-vs-odin-rebench
micro-bench predicted (which had Odin 0.75-3.35× depending on cell).
The compounding comes from per-leaf-eval overhead in the C++ Python
callback path accumulating over many short selfplay games.

## Training value_acc curve

|     | iter0  | iter1  | iter2  | iter3  | iter4  |
|---  |  ---:  |  ---:  |  ---:  |  ---:  |  ---:  |
| Odin run | 67.07% | 98.91% | 98.79% | 99.11% | 99.34% |
| C++ run  | 67.06% | 75.20% | 71.65% | 70.82% | 70.31% |

iter0 is identical (same random-game pre-collect, same training script).
iter1+ diverges because the selfplay data the value head was trained on
diverges:

- **Odin selfplay games** (pre-f21a3cd): MCTSTree default seed=0 →
  Dirichlet RNG always starts from the same state → games per worker
  thread are byte-identical when policy is peaked (the
  [autogodin-6qt](bd:autogodin-6qt) failure mode), and mostly redundant
  when policy is diffuse. Value head sees a near-constant outcome
  distribution and fits to 99%.
- **C++ selfplay games**: MCTSTree uses `std::random_device{}()` per
  construction (autogo/src/alpha_go/cpp/mcts/mcts.cpp:12) →
  genuinely random Dirichlet noise → diverse games → value head sees a
  real outcome distribution and plateaus at 70% (the realistic ceiling
  for the 9×9 selfplay manifold at this model size).

## Gauntlet results (21 games)

| | wins | losses | win-rate | Wilson 95% CI |
|---|---:|---:|---:|---:|
| Odin-trained iter4 (postfix) | **0** | 21 | 0.0% | **[0.0%, 15.5%]** |
| C++-trained iter4 (this run) | 21 | 0 | 100.0% | [84.5%, 100.0%] |

Binomial null probability (true rate is 0.5): **4.77 × 10⁻⁷**.
Bayes factor vs equal-strength hypothesis: ~2 × 10⁶.

By color: Odin-trained went **0 / 11 as Black**, **0 / 10 as White** —
swept under both colors. No draws.

Mean game length 161 moves, consistent with model resignation
thresholds firing during decisive losing positions.

## Interpretation

The honest C++ run produced a meaningfully stronger model than the
Odin run that had been distorted by [autogodin-6qt](bd:autogodin-6qt).
This validates the f21a3cd fix as load-bearing: the bug had quietly
inflated the postfix run's metrics by a full 29 percentage points of
value_acc while simultaneously making the resulting model weaker in
actual play. **Value_acc was Goodharted; the gauntlet is the only
trustworthy strength signal.**

This also means **all prior bpoC-style training results from before
f21a3cd should be re-evaluated** before being cited. The Odin shim
through the buggy MCTSTree default was producing data that overstated
training success.

## What this DOESN'T show

This experiment compares **two training pipelines**, not the two MCTS
backends in isolation. With the f21a3cd fix in place, an Odin-shim
training run today would produce data with the same entropy as the C++
path (modulo the per-tree-construction overhead in randomized seeds).
The right next step to claim backend-comparison parity at training
time is:

1. **bpoC-rerun-odin-postfix-v2**: same recipe, Odin shim WITH the
   f21a3cd fix applied. Expect value_acc curve to land near the C++
   run's (~70%) and gauntlet to be near 50/50 vs this run's iter4.
   This would seal "training pipeline is backend-agnostic given the fix."
2. **Cleanup**: the over-confident value_acc claim in the postfix
   run's report (and any README citation of "99.34%") should be
   annotated with the 6qt caveat.

## Artifacts

- `postmortem/iter4_best.pt` — C++-trained iter4 (gitignored, 46 MB)
- `postmortem/summary.json` — training timings + per-iter metadata
- `postmortem/logs/` — full 12-file selfplay + train log set
- `postmortem/gauntlet_run.log` — raw miniwini gauntlet log
- `gauntlet_results.json` — structured 21-game record + Wilson CI
- `gauntlet.py` — gauntlet runner (reusable for future model comparisons)

## Cost

- Training (JL L4 411331, 225 min): ~$1.65
- Gauntlet (miniwini, free): $0
- Earlier autogodin-6qt repro (L4 411323, 15 min): ~$0.10
- **Total session spend on this dimension: ~$1.75**

JL balance after this work: ~$1.95 remaining.
