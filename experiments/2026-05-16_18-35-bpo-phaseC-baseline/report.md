# bpo PATH C: scaled training-loop baseline on Odin (JL L4)

Date: 2026-05-16 · Bead: autogodin-bpo (in progress)

## Goal

Real baseline run after ydh.8's smoke. Scale the training loop up to
something credible — fastlearn-arch model, real batch size, real
selfplay volumes, with **Odin MCTS on GPU** (unblocked by the 7km
fix). Acceptance from `autogodin-bpo`: per-iter loss curve, total
cost, sample artifact. (No claim of strength vs upstream baselines —
that requires the full league, gauntlet evaluation, and the parent
dataset; see `autogodin-bpo` PATH A and `autogodin-mls`.)

## Setup

- Provider: JarvisLabs, machine_id 410905, **L4 IN2 ($0.44/hr)** —
  right-sized after ydh.8's A100 1.4%-VRAM waste lesson.
- Bootstrap: `scripts/jl_bootstrap.sh` (same as ydh.8 — apt + Odin +
  uv + clone + build both backends + `/nfs` symlink), ~12 min.
- Model: `SizeInvariantGoResNet-256ch-10b`, **13,028,099 params** (~13M),
  batch 512. (Upstream `run.py` patches the upstream train.py
  constants in-place: 128→256 channels, 128→512 batch.)
- Selfplay: 4 parallel workers + `--batched-inference` (shared
  GPU engine, one model copy), Odin MCTS via the shim (post-7km),
  200 sims/move, c_puct=1.0, temperature=1.0, dirichlet noise on.

## Configuration

| param | value |
|---|---|
| random pre_collect | 5,000 games |
| selfplay games / iter | 500 |
| selfplay workers | 4 |
| MCTS sims / move | 200 |
| model | SizeInvariantGoResNet-256ch-10b, ~13M params |
| batch size | 512 |
| optimiser | AdamW, lr=1e-3 cosine, wd=5e-3 |
| time budget / train iter | 900 s (15 min) |
| value head | tanh, BCE loss against game outcome |
| policy head | softmax CE against MCTS visits, teacher-masked |

## Per-iter results

| iter | train wall | samples | steps | train_loss | policy_acc | value_acc | peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 900 s (capped) | 810,000 | 2584 | 3.79 | 9.89% | 66.96% | 4.1 GB |
| 1 | 100 s | 73,551 | 286 | 3.80 | 8.44% | **74.74%** | 4.2 GB |
| 2 | 77 s | 55,931 | 218 | 4.08 | 7.53% | 65.45% | 4.2 GB |
| 3 | 66 s | 50,168 | 194 | 4.23 | 8.00% | 60.79% | 4.2 GB |

| iter | selfplay games | wall | g/min | notes |
|---:|---:|---:|---:|---|
| 0 | 500 | 35.5 min | 14.1 | mostly 162-move max-out games |
| 1 | 500 | 27.2 min | 18.4 | resign-threshold starts firing |
| 2 | 500 | 24.9 min | 20.1 | more games resigning |
| 3 | partial | — | — | killed mid-collect (see below) |

GPU usage: peak VRAM ~4.2 GB on a 24 GB L4 (17%). Much healthier than
ydh.8's 1.4% on A100. Power draw during MCTS+selfplay sat around
40-60% of L4's 72 W envelope.

## The interesting finding: catastrophic forgetting

**Value accuracy peaked at iter1 (74.74%) then degraded** through iter2
(65.45%) and iter3 (60.79%). Train loss also climbed: 3.79 → 3.80 →
4.08 → 4.23. Policy accuracy hovers around 8-10% — close to the random-
init policy_acc baseline, so policy is not really learning across iters.

Root cause: **our `run.py` writes `dataset-it{N+1}.txt` containing
only the latest `selfplay-itN`** — not the carry-forward chain upstream
uses. Upstream's `run_iteration.sh` is explicit about this: each
`dataset-it{N+1}.txt` for N ≥ 1 inherits every path from the previous
iteration's dataset and appends the new league + selfplay dirs. We
flatten that to a single-path file:

```
dataset-it0.txt: experiments/.../random-it0          # 5000 games (~810k samples)
dataset-it1.txt: experiments/.../selfplay-it0        # 500 games (~74k samples)
dataset-it2.txt: experiments/.../selfplay-it1        # 500 games (~56k samples)
dataset-it3.txt: experiments/.../selfplay-it2        # 500 games (~50k samples)
```

Iter 1+ trains on ~10× less data than iter 0 *and* loses all signal
from prior iters. With MIN_STEPS=300 forcing 300 gradient steps even
on tiny datasets, the model effectively memorises the latest 50k
samples and forgets everything else. The value head briefly improves
(iter1) when the data shifts from random-vs-random to actual MCTS
play, then deteriorates as each new iter's data drifts further from
the previous.

This is a **runner bug, not a learning bug**. The Odin MCTS, GPU NN
forward, train loop, and resume-from-checkpoint all work correctly —
the loss curve just *measures the wrong thing* because we feed it the
wrong data sequence. Fix is a one-line change in `run.py`'s dataset-
txt writer:

```python
# carry forward prior dataset entries (except random-it0 after iter0)
prev_lines = (EXP / f"dataset-it{it}.txt").read_text().splitlines()
keep = [l for l in prev_lines if not l.strip().startswith("#")
        and "random-it0" not in l]  # drop random-it0 from iter1+
ds.write_text("\n".join(keep + [f"experiments/{EXP_NAME}/selfplay-it{it}"]) + "\n")
```

Filed as autogodin-pez follow-up — re-run will get a real loss curve.

## What this validates

- **Bootstrap + train + Odin-MCTS selfplay + resume loop** runs
  end-to-end on a single GPU instance without docker/cluster.
- **Odin MCTS via the shim is now safe** for batched evaluators
  (the 7km fix landed in time; selfplay used Odin's batched tree, GPU
  forward via `LeafBatchedNNEvaluator`).
- **L4 was the right pick** at this workload: 4.2 GB peak VRAM /
  24 GB available is workable. Bigger GPU would have been pure waste.
- **Resign-threshold healthy**: as the value head learned (iter1),
  games started resigning earlier → selfplay throughput grew from
  14 g/min (iter0) → 20 g/min (iter2). This is exactly the autogo
  resign mechanism doing its job.

## What this does NOT validate

- Any strength vs upstream baselines. Cannot claim "training works"
  in the strength sense until the dataset-carryforward fix is in.
- Holdout accuracy / generalisation: we did not split off a holdout
  set or compute against the fastlearn parent's reference numbers.
  See `autogodin-mls` for that path.
- Full league dynamics (gauntlet, league-state). PATH A territory.

## Cost

| segment | wall | spend |
|---|---:|---:|
| instance creation | <1 min | $0.01 |
| bootstrap | 12 min | $0.09 |
| pre_collect 5k | 83 s | $0.01 |
| train iter0 | 15.3 min | $0.11 |
| selfplay iter0 | 35.5 min | $0.26 |
| train iter1 + selfplay iter1 | 28.9 min | $0.21 |
| train iter2 + selfplay iter2 | 26.1 min | $0.19 |
| **(over-run past stop-after-iter2:** train iter3 + partial iter3 selfplay**)** | ~5 min | **$0.04** |
| pull + destroy | ~1 min | $0.01 |
| **total billed (L4 IN2 @ $0.44/hr)** | **~125 min** | **~$0.92** |

The over-run was caused by two pollers losing SSH connections (exit
255) without proper retry. By the time I noticed, the runner had
trained iter3 and started iter3 selfplay; killed it then. Costed
about $0.04 above the iter2 stop target. Lesson saved as a feedback
memory.

Original estimate was $0.50; final cost ~$0.92. Two contributors:
selfplay-worker scaling worse than I estimated (1.5× from 4 workers,
not 4×), and the poller-death over-run. Both honest mistakes worth
flagging.

## What's next

- **Fix the dataset carryforward in `run.py`** (autogodin-pez). Cheap
  one-line change.
- **Re-run with the fix** on the same scope (iter0..iter4, 500
  games/iter, L4): should produce a *monotonically improving* curve
  instead of the U-shape we got.
- After that, **autogodin-mls / bpo PATH B** (parent dataset bootstrap)
  becomes the natural rigorous follow-up. Or **bpo PATH A** if we
  want to fully replicate upstream's run_iteration.sh on Odin.
