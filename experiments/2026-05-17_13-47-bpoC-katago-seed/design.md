# bpoC-katago-seed: bpoC PATH C bootstrapped from KataGo SGFs

## Question

How much does the source of iter0 training data matter for bpoC PATH C?

The bpoC-rerun-postfix baseline trained iter0 from 5k random-vs-random games.
Its value_acc curve was 67% → 99% over iter0 → iter1, suggesting iter0 mostly
just learns "moves that don't break the board" and the real signal arrives
when iter0's network plays itself at iter1.

Hypothesis: seeding iter0 from **strong** game data (KataGo selfplay) should
give iter1 selfplay a much better-informed prior, and the iter4 model should
be measurably stronger head-to-head against bpoC-rerun-postfix iter4.

## Setup

Identical to bpoC-rerun-postfix EXCEPT:

| step | baseline | here |
|---|---|---|
| iter0 dataset | 5000 random×random NPZs | **9,931 KataGo 9×9 NPZs** |
| pre_collect step | yes (~5 min CPU) | **skipped** |
| iter1..4 selfplay+train | unchanged | unchanged |
| model arch | 256ch × 10b | unchanged |
| MCTS sims/move | 200 | unchanged |
| selfplay games/iter | 500 | unchanged |

The KataGo NPZs come from `game_data/9x9/katago-9x9-sample/` — converted from
50 daily archives of katagoarchive.org/kata1/traininggames via
`tools/sgf_to_npz.py` (494,842 SGFs scanned → 9,931 9×9 games kept after
filtering for SZ=9, no handicap, parseable result).

Komi distribution is varied (6.5 most common, range 4.5..10.5) — modern 9×9
komi is mostly in this range, so we accept the spread as realistic.

## Policy target trick

KataGo NPZs don't carry `mcts_visits` arrays (only the played move). autogo's
`GoDataset` (dataset.py:214-223) detects this and falls back to a
label-smoothed one-hot policy on the played move (eps=0.1). So the SGF move
acts as the policy target, smoothed.

That's not as rich a target as MCTS visit counts, but it should still beat
the random-bootstrap iter0 policy, which has essentially no signal.

## Expected outcome

- iter0 trains to higher value_acc than baseline (KataGo games end decisively)
- iter0 policy_acc is non-trivial (won't be flat like random-bootstrap)
- iter4 plays meaningfully better than bpoC-rerun-postfix iter4 in head-to-head

## Non-goals

- We do NOT measure absolute strength vs KataGo. The point is internal
  comparison: katago-seed iter4 vs random-seed iter4.
- We do NOT train multiple architectures or do a FLOPs sweep. That's 12g.4
  territory (still has the deterministic-bug to fix before re-running).

## Cost estimate

- bpoC-rerun-postfix wall-clock on L4: ~75 min @ $0.44/hr = **~$0.55**
- Skipping pre-collect saves ~5 min CPU → similar or slightly cheaper
- One-time KataGo NPZ upload: 117 MB tar, ~2 min over JL rsync
