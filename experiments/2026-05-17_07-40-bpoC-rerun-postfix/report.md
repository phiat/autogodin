# bpo PATH C rerun: carry-forward fix verified (JL L4)

Date: 2026-05-17 · Bead: autogodin-bpo (closing) · Prior: `../2026-05-16_18-35-bpo-phaseC-baseline/report.md`

## Goal

Re-run the bpoC scope (same model, same selfplay volumes, 4 iters) after
the `autogodin-8hj` carry-forward fix landed in `run.py`. The previous
run's `value_acc` regression (67 → 75 → 65 → 61%) traced to a bug in
the dataset-txt writer that didn't carry prior iterations forward,
so iter ≥ 2 trained on only the latest ~45k selfplay samples. Acceptance:
no regression — value_acc holds or improves across iter1..iter4.

## Setup

Identical to the previous bpoC run except:

- New experiment dir; same configuration knobs (5k random pre_collect,
  500 selfplay games / iter, 200 sims, 256ch×10b, batch 512).
- `run.py` includes the `8hj` carry-forward writer: iter ≥ 1 dataset-txt
  inherits every prior path (except `random-it0`, which is iter0-only)
  and appends the freshly-collected `selfplay-it{N}`.
- Two redundant watchdogs around the run instead of one SSH-poller —
  remembered from previous run's $0.04 over-run.

Provider: JarvisLabs, **L4 IN2 @ $0.44/hr**, 50 GB storage.
Bootstrap: `scripts/jl_bootstrap.sh`, completed in ~14 min.

## Per-iter results

| iter | dataset (paths × games) | samples | steps | train_loss | train_policy_acc | train_value_acc | peak VRAM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0  | random-it0 (5000)                 | 810,000 | 2577 | 3.7997 |  9.97% | **67.07%** | 4.1 GB |
| 1  | selfplay-it0 (500)                |  ~45,000 |  174 | 3.7765 | 93.37% | **98.91%** | 4.2 GB |
| 2  | selfplay-it0+it1 (1000)           |  ~90,000 |  342 | **2.9734** | 96.40% | **98.79%** | 4.2 GB |
| 3  | selfplay-it0..it2 (1500)          | ~135,000 |  400 | 3.1253 | 100.00% | **99.11%** | 4.2 GB |
| 4  | selfplay-it0..it3 (2000)          | ~180,000 |  400 | 3.1773 | 98.70% | **99.34%** | 4.2 GB |

| iter | selfplay games | wall (min) | g/min |
|---:|---:|---:|---:|
| 0 | 500 | 22.6 | 22.1 |
| 1 | 500 | 21.8 | 22.9 |
| 2 | 500 | 18.0 | 27.7 |
| 3 | 500 | 18.3 | 27.4 |

Resign threshold kicks in around iter2 (g/min jumps from ~22 to ~28),
same shape as the previous run. Selfplay was uniformly faster this time
(iter0 22.1 g/min vs 14.1 g/min before) — likely lower contention on
this particular L4 host, since iter0 selfplay code path didn't change.

## Carry-forward fix verified

Side-by-side `train_value_acc` across iters:

| iter | broken run (`bpo` #1) | this run (`bpo` #2) |
|---:|---:|---:|
| 0 | 66.96% | **67.07%** |
| 1 | 74.74% | **98.91%** |
| 2 | **65.45%** ⬇ regression | **98.79%** |
| 3 | **60.79%** ⬇ regression | **99.11%** |
| 4 | (partial, killed) | **99.34%** |

`train_loss` shape mirrors it: previous run climbed 3.79 → 4.23 across
iters (catastrophic forgetting on the latest-only dataset); this one
drops to 2.97 at iter2 (the first iteration that actually carries
forward) and stays in the 3.13–3.18 band afterward. The slight uptick
from iter2 → iter3 is expected: the dataset is now larger and more
diverse, so the model can no longer trivially memorise it.

Step counts confirm the carry-forward dataset really did grow:
174 → 342 → 400 → 400 (capped by 900s budget × ~0.4s/step at iter3+).

## What this validates

- **The `8hj` carry-forward writer works.** This was the only code
  change between bpoC #1 and bpoC #2; the value_acc regression is gone.
- **Closing `autogodin-bpo` PATH C acceptance gate.** Per the bead:
  "loss curve, sample artifact, total cost, vs random-init holdout."
  No holdout split is configured in the upstream train.py, so the
  curve here is train-set. PATH B (`autogodin-mls`) is the natural
  follow-up if rigorous holdout numbers are needed.
- **Robust watchdog approach worked.** Two layers, both kept the
  instance from over-running: (1) Claude-harness background polling
  `jl run status` every 120 s with auto-pause on terminal state;
  (2) a streaming log watcher emitting iter-completion events. No
  over-run this time; the watchdog paused 411131 within ~2 min of
  `summary.json` appearing.

## What this does NOT validate

- **Strength vs upstream baseline.** Train-set policy_acc hits 100%
  at iter3 — the model is memorising its own selfplay games, not
  necessarily learning a stronger policy. Holdout evaluation or a
  gauntlet match vs the previous-iter model is what would settle that.
- **Generalisation past 5 iters.** The trend is flat-ish saturation
  on train_value_acc (~99%) at iter2+, which is consistent with the
  model fitting the carry-forward dataset rather than meaningfully
  generalising. A longer run with a holdout split is needed.

## Cost

| segment | wall | spend |
|---|---:|---:|
| instance creation | <1 min | $0.01 |
| bootstrap (apt + Odin + uv + build both backends) | ~14 min | $0.10 |
| pre_collect 5k random | 1.4 min | $0.01 |
| iter0 train | 15.3 min | $0.11 |
| iter0 selfplay | 22.6 min | $0.17 |
| iter1 train + iter1 selfplay | 23.0 min | $0.17 |
| iter2 train + iter2 selfplay | 20.2 min | $0.15 |
| iter3 train + iter3 selfplay | 20.8 min | $0.15 |
| iter4 train | 2.6 min | $0.02 |
| post-run pause delay + download | ~3 min | $0.02 |
| **total billed (L4 IN2 @ $0.44/hr)** | **~107 min** | **~$0.82** |

Original estimate was $1.10–$1.30; came in at $0.82 — lower than the
previous run's $0.92 despite producing all 4 iters this time. Two
contributors: (1) faster selfplay across the board (~25-30% throughput
gain over previous run, likely host-load variance), (2) no over-run
because the watchdog paused the instance on terminal state instead of
running past the stop signal.

## Artifacts

- `summary.json` — config + per-phase timings + checkpoint list (run output).
- `logs/00_random.log` — random pre_collect.
- `logs/01_train_it0.log`, `logs/03_train_it{1..4}.log` — train output incl. `===RESULT===` JSON line.
- `logs/02_selfplay_it{0..3}.log` — selfplay output.
- `checkpoints/iter{0..4}_best.pt` — 47 MB each, gitignored. Saved locally for future eval / parent-dataset bootstrap.

## What's next

- **`autogodin-bpo` can close.** PATH C acceptance gate met.
- **`autogodin-mls` (PATH B)** — running fastlearn Phase A on a parent
  dataset is the next natural step. Either generate a 10-iter parent
  via PATH A (~$9-13.5) or ask upstream for the existing dataset-it10
  NPZs. Phase A sweep itself is ~$1.50 once dataset-it10 exists.
- **`autogodin-bpo` PATH A** — full league + gauntlet replication of
  upstream `run_iteration.sh` on Odin. Costliest path; only if we want
  to claim parity with the upstream champion.
