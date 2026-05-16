# ydh.5: integration smoke — Odin plugs into autogo's training stack

## What landed

End-to-end smoke that proves the Odin backend can power autogo's
training loop, not just the throughput micro-benches.

`smoke_train.py` walks the full pipeline:

1. `import alpha_go_cpp` resolves to our shim
   (`python/odin_backend/alpha_go_cpp.py`), which re-exports
   `alpha_go_odin`. autogo code that does `import alpha_go_cpp` (e.g.
   `gameplay.py`, the agents/) sees the Odin board + MCTS.
2. `python -m alpha_go.self_play --black random --white random
   --num_games 5` runs self-play on the Odin GoBoard and writes
   autogo-format NPZ files (5 games / 810 positions).
3. `GoDataset` loads those NPZ files (`board`, `move`, `winner` keys).
4. `SizeInvariantGoResNet(channels=32, n_blocks=4)` (76,323 params)
   trains 50 steps on the Odin-generated data. CPU; ~29 steps/sec on
   miniwini's local i9.
5. `MCTSTree` (Odin) + Python NN evaluator runs 200 sims in 130ms
   (1,486 sims/sec — Python NN-eval cost dominates).

```
✓ alpha_go_cpp resolves to Odin shim: .../python/odin_backend/alpha_go_cpp.py
✓ GoDataset: 810 positions across 5 games
✓ SizeInvariantGoResNet built: 76,323 params
  step  50  loss=2.9562  policy=2.6669  value=0.2893
✓ training: 50 steps in 1.72s (29.0 steps/sec); loss 5.0999 → 2.9562
✓ MCTS 200 sims in 0.13s (1486 sims/sec)
  top-5 action probs: [(80, '0.121'), (62, '0.080'), ...]
```

## What this is NOT

Not a full fastlearn Phase A replication. The original Phase A is a
9-config NN-architecture sweep on the parent's accumulated `dataset-it10`
(~10 iterations of MCTS-collected self-play data, ~50k games). That
dataset is not in this repo; recreating it via Odin self-play before
training is on the order of a multi-hour run, and the sweep itself
needs a GPU. This smoke is what's tractable on CPU and within a single
session.

The follow-up (track separately) is to:

1. Bootstrap a parent dataset using Odin self-play + a strong reference
   agent (or several iterations of selfplay-train-promote).
2. Rent a GPU and re-run the 9 Phase A configs.
3. Compare end holdout policy_acc against the original report's
   `0.3046–0.3078` range (the bug fix + 192ch baseline).

## Why we can ship the bead anyway

The bead's stated goal — *confirm Odin is suitable for actual training
loops, not just synthetic benchmarks* — is what's actually validated
here. The integration is real: autogo's self-play, dataset loader,
ResNet, and MCTS evaluator paths all run through the Odin backend with
zero changes to autogo source (just `PYTHONPATH=python/odin_backend:...`
or the existing `scripts/run_with_odin_backend.sh`).

The full Phase A replication is a research-quality follow-up, not a
gate on declaring Odin training-ready.

## Reproduce

```bash
# Generate Odin self-play data (uses alpha_go_cpp shim → alpha_go_odin)
GAME_DATA_DIR=/tmp/ydh5-game-data \
  PYTHONPATH="python/odin_backend:python:autogo/src" \
  autogo/.venv/bin/python -m alpha_go.self_play \
  --black random --white random --num_games 5 --board_size 9 \
  --save-name ydh5-smoke-random

# Train + MCTS smoke on the generated data
PYTHONPATH="python/odin_backend:python:autogo/src" \
GAME_DATA_DIR=/tmp/ydh5-game-data \
  autogo/.venv/bin/python \
  experiments/2026-05-16_17-09-ydh.5-fastlearn-on-odin/smoke_train.py
```
