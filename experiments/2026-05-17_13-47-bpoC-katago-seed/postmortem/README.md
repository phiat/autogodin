# bpoC-katago-seed: killed at iter0+selfplay boundary

**Status: aborted.** See [autogodin-6qt](bd:autogodin-6qt) for the bug.

## What happened

- iter0 trained cleanly from 9,931 KataGo SGFs (15 min on L4)
- iter0 ckpt saved with strong metrics:
  - train_policy_acc **45.08%** (vs random-seed baseline 30%) ✓
  - train_value_acc 59.97% (vs baseline 67% — KataGo games are more competitive)
- iter0 selfplay collapsed: **391/391 games byte-identical**
  - all 24 moves, same first 6 `[(0,0),(3,4),(4,6),(3,2),(3,6),(7,6)]`
  - all White-wins by double_pass at score W+7.5
  - first-move histogram across first 100 games: `(0,0)` × 100
- Killed at ~6 min into selfplay-it0 to avoid burning $0.50 producing
  useless data. Total burn: ~$0.20.

## Why this is surprising

Both bpoC-rerun-postfix and this run use the same `CppMCTSAgent` config
(`temperature=1.0, add_noise=True, leaf_batch_size=64`) with 4 workers.

bpoC-rerun-postfix produced diverse selfplay games and trained to
value_acc 67% → 99% across iters — clear evidence the selfplay variance
was real there.

The ONLY difference here is iter0 training data: KataGo SGFs vs
random-vs-random games. The KataGo-trained iter0 has a much
more-peaked policy distribution (45% top-1 acc vs 30%), and that
peakiness appears to be enough to break selfplay diversity.

## Hypothesis (testable)

When iter0's policy has one move at >99%, Dirichlet root noise + temp=1.0
sampling produce a distribution that's still effectively 1-hot. Net
effect: every selfplay game starts with the same first move, and MCTS
expansion converges to the same tree across all 4 workers (workers may
also share a seed pattern).

## Artifacts in this dir

- `iter0_best.pt` — 46 MB checkpoint (NOT committed — gitignored)
- `2026*-game*.npz` — 20 sample selfplay NPZs (byte-identical to each other)
- `01_train_it0.log` — full iter0 training log
- `02_selfplay_it0.log` — selfplay log showing W+7.5 (24 moves, double_pass) for every game

## Next steps (tracked in autogodin-6qt)

1. **Local smoke**: reproduce with 2 games + 50 sims on local CPU
2. **Inspect Odin shim**: does `python/odin_backend/__init__.py` actually wire
   `add_noise` into `alpha_go_odin.MCTSTree`, or silently drop it (12g.4 pattern)?
3. **Print effective seeds**: instrument `alpha_go.self_play` to log per-worker
   `seed` and `np.random.get_state()` after warmup to prove workers differ
4. **Sampling check**: when network policy is >99% on one move, what's the
   actual sample distribution from CppMCTSAgent with temp=1.0?
5. After fix: retry bpoC-katago-seed (the iter0 ckpt here can be reused;
   just resume from `iter0_best.pt` and skip the iter0-train step)

The good news is the converter + KataGo data pipeline works perfectly; the
bug is downstream in selfplay diversity.
