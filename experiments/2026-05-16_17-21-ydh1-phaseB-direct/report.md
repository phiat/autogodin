# ydh.8 Phase B (direct): training loop on JL GPU

Date: 2026-05-16 · Bead: autogodin-ydh.8 (closed)

## Goal

Demonstrate the autogo training loop runs end-to-end on a rented JL GPU
with the Odin port present in the source tree. Acceptance: iter0 +
iter1 checkpoints exist on a GPU instance; report has loss curves,
timings, sample game, cost. (See autogodin-ydh.8 description.)

This is **a smoke test of the pipeline**, not a baseline of any
meaningful skill. See [[autogodin-bpo]] for the real-baseline plan.

## What we ran

A single-instance, docker-less variant of upstream's
`run_iteration.sh 0 1`:

  1. pre_collect 500 random×random games (CPU on instance)
  2. train iter0 from random data (GPU)
  3. collect 200 selfplay games with iter0 NN + C++ MCTS @ 200 sims/move
  4. train iter1 from iter0 selfplay (GPU, resume from iter0)

Files: `run.py` (driver) + `train.py` (copied from
`autogo/experiments/2026-04-26_22-32-train-fromscratch/`) +
`_collect_iter0.py` (runtime agent registration).

## Setup

- Provider: JarvisLabs, machine_id 410887, A100-PCIE-40GB IN2
- Bootstrap: `scripts/jl_bootstrap.sh` — apt deps + uv + Odin
  (`mise install odin@dev-2026-05`) + clone autogodin + setup_autogo
  + build alpha_go_odin + build alpha_go_cpp wheel + `/nfs` symlink to
  `~/nfs-local`. ~12 min wall.
- Model (train.py default): SizeInvariantGoResNet 128ch × 10b,
  **2,966,531 params** (~3M, not the 18M MuP variant from upstream's
  fastlearn).
- Run command:
  ```sh
  PYTHONPATH="$PWD/python:autogo/src" GAME_DATA_DIR=$HOME/nfs-local/game_data_root \
    autogo/.venv/bin/python -u experiments/2026-05-16_17-21-ydh1-phaseB-direct/run.py
  ```

## Results

| step | wall | output |
|---|---:|---|
| 1 — pre_collect 500 random games | 12.6 s | 500 npz |
| 2 — train iter0 (1264 steps, 2 epochs) | 33 s | iter0_best.pt (11.9 MB) |
| 3 — collect 200 selfplay games (C++ MCTS) | 1289.7 s | 200 npz |
| 4 — train iter1 (452 steps, 2 epochs, resume) | 13 s | iter1_best.pt (11.9 MB) |

Train-loop metrics (from `logs/02_train_it0.log`, `04_train_it1.log`):

| iter | train_loss | policy_acc | value_acc | peak VRAM |
|---|---:|---:|---:|---:|
| 0 | 3.86 | 8.68% | 66.7% | 548 MB |
| 1 | 3.84 | 8.79% | 73.0% | 560 MB |

The value head moved 66.7 → 73.0% — *some* signal made it through. The
policy head moved 8.68 → 8.79%, which is noise on 200 selfplay games.
Don't read meaningful Go skill into either number — these are smoke
metrics, not training results.

## Cost

| segment | wall | spend |
|---|---:|---:|
| instance creation + ssh handshake | ~1 min | $0.01 |
| bootstrap (apt + Odin + builds + venv) | ~12 min | $0.18 |
| steps 1+2 (first try, succeeded) | ~1 min | $0.02 |
| step 3 (first try, failed at shim) | ~10 sec | $0.001 |
| steps 3+4 (re-run after fix) | ~22 min | $0.33 |
| artifact pull + destroy | ~30 sec | $0.01 |
| **total billed (A100 IN2 @ $0.89/hr)** | **~37 min** | **~$0.55** |

## Things that went wrong on the way

Three failures during the run, two of which were avoidable:

1. **`mise install odin@nightly` 404** — there's no `nightly` tag in
   `odin-lang/Odin` releases. Should have checked `mise ls` before
   writing the bootstrap; fixed by hard-coding `dev-2026-05` (the
   version we use locally). 1 min lost.
2. **`clang` missing** — autogodin's bootstrap apt list was written
   from memory and missed `clang`, which Odin's default linker driver
   requires. Cross-checking `autogo/.devcontainer/Dockerfile` upfront
   would have caught it. 2 min lost.
3. **Odin shim batched-API mismatch** (autogodin-7km) — the
   `python/odin_backend/alpha_go_cpp.py` shim aliases
   `alpha_go_cpp.MCTSTree` to `alpha_go_odin.MCTSTree`, but their
   `run_simulations_batched` evaluator-return signatures *differ*
   (C++: `list[(p,v)]`; Odin: `(list[p], list[v])`). `CppMCTSAgent +
   LeafBatchedNNEvaluator` were written for the C++ shape only. Going
   through the shim crashed in the Odin trampoline. Switched step 3 to
   the real C++ wheel; the shim bug is filed as a separate bead.

## GPU sizing — we paid for an idle silicon truck

Peak VRAM use was **548 MB on a 40 GB card — 1.4% utilization**. Power
draw stayed around 45 W of 250 W (18%). The GPU was doing some work
during forward passes but mostly sat waiting on single-process MCTS
tree traversal.

Honest read: on this workload (small model, batch 128, one selfplay
worker), an **L4 IN2 at $0.44/hr** would have produced the same
wall-clock at half the price (~$0.28 total). The A100 was the wrong
GPU for this run; right-sizing is now documented in [[autogodin-bpo]]
and a feedback memory.

## What this validates

Despite the rough edges, the pipeline genuinely ran end-to-end on a
rented GPU using the autogodin tree:

- `scripts/jl_bootstrap.sh` provisions a JL PyTorch instance for
  autogodin from cold start in ~12 min.
- `experiments/2026-05-16_17-21-ydh1-phaseB-direct/run.py` is a
  single-node, docker-less replacement for `run_iteration.sh 0 1` that
  works without cluster.toml or GHCR auth.
- Both Odin (`alpha_go_odin`) and C++ (`alpha_go_cpp`) backends build
  cleanly on a fresh Ubuntu PyTorch image; the contract-parity test
  from autogodin-27v presumably still holds (not re-run here).
- `train.py` from upstream autogo runs unmodified once `/nfs` is
  symlinked and `GAME_DATA_DIR` is set.

## What this does NOT validate

- Any claim about Go playing strength. 500 random + 200 selfplay
  isn't enough data for the 3M-param model to learn anything; treat
  the policy_acc numbers as noise.
- Odin MCTS on GPU. Step 3 ran with C++ MCTS due to autogodin-7km.
  Demonstrating Odin MCTS through autogo's NN-agent path needs that
  bug fixed (or a non-shim direct agent class).
- The full cluster/league flow from `run_iteration.sh` — we ran a
  simplified single-node variant.

## Next

See [[autogodin-bpo]] for the real-baseline plan. The natural sequel
is **PATH C**: scale this same script up to fastlearn-equivalent arch
(256ch × 14b), batch 512, 8-16 parallel selfplay workers, 5+
iterations — on an **L4** unless we're explicitly throughput-bound.
