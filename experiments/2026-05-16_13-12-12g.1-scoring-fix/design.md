# Design: Tromp-Taylor scoring bug, life/death fix

**Bead:** `autogodin-12g.1`
**Date:** 2026-05-16
**Status:** Proposal, prototype landed alongside.

## Problem

autogodin's GoBoard.score() implements Tromp-Taylor (TT) area scoring:
**every stone on the board counts as alive**, and an empty region is awarded
to a color iff that region touches only that color's stones. The C++ and
Odin implementations match (parity-fingerprinted), so the algorithm is the
same on both sides — the issue is the *algorithm itself*, not the port.

TT's correctness depends on the game being *fully resolved* before the
final position is scored: dead stones must already be captured. In
self-play, this requires the policy to keep playing until every dead
group has been captured, which a learning policy often won't do — it
passes too early.

The upstream README notes the trained model "misjudges life/death due to
being trained on Tromp-Taylor scoring rules." The mechanism:

```
. . . . . . . . .
. X X X X X X X .       Black wall, alive
. X O O O O O X .
. X O . . . O X .       White group: closed shape but only 1 eye —
. X O . . . O X .       under perfect play, Black plays inside and
. X O . . . O X .       captures the lot. Real outcome: White's 12
. X O O O O O X .       stones are dead.
. X X X X X X X .
. . . . . . . . .       TT score after Black passes here:
                          - Black stones: 24, territory (outside): 32
                          - White stones: 12, "territory" (inside): 9
                                                  (single-color region)
                        => Net: Black +35 (after komi)
                        Reality: Black should win by ~60+
                        (the 12 dead White stones plus inner 9 = +21
                        more for Black if dead stones are accounted).
```

Worse: the value head learns from these mis-scored positions, so it
internalizes a confused picture of what dying-but-still-on-board means.
The same pathology in many positions makes endgames wobbly.

## Options

### A — "Cleanup playout" before scoring (KataGo-style)

When self-play hits 2 consecutive passes, keep running a search-driven
cleanup that plays moves into both players' territories until no
captures happen and no further moves are productive. Then score with TT
on the cleaned board.

**Pro:** Doesn't change scoring semantics. Cleanup uses the same
machinery (MCTS + policy + value) we already have.

**Con:** Needs a stopping rule that's neither too eager (truncates real
games early) nor too lazy (runs forever on quiet positions). Doubles or
triples self-play wall time on positions that need cleanup.

### B — Benson's algorithm (unconditional life) + dead-stone removal

Run Benson's algorithm on the final position: each group is "unconditionally
alive" iff it has ≥2 "vital regions" (each empty region adjacent to ALL
stones in the group, and every empty point in the region is adjacent to
the group). Groups failing Benson are marked dead and removed before TT.

**Pro:** Deterministic, no NN, no search. O(N²) where N = board cells.
Provably correct (groups marked alive ARE alive; the algorithm is
conservative — true-alive groups may not pass Benson, but those marked
dead really are dead).

**Con:** Conservative — many actually-alive groups don't have *unconditional*
life (e.g., simple seki). Removing them would be wrong. Need a fallback
for the "Benson-says-not-alive but might be alive" case.

### C — Eye-counting heuristic + dead-stone removal

For each group, count "eye-like points": single empty cells where ALL
on-board neighbors are friendly. Groups with ≥2 such points are *likely*
alive. The rest are removed before TT.

**Pro:** Trivially cheap. Catches the README's pathological case (1-eye
groups) and similar. Implements in ~40 lines of Python.

**Con:** Not perfect. False positives (long-life groups with 1 big eye
that's actually 2 effective eyes). False negatives (groups with 2 eyes
that the heuristic doesn't recognize because of stones inside the eye
points). Documented and bounded by the test set.

### D — Small NN life/death classifier

Train a small CNN to label each on-board stone as alive/dead. Run it
post-game.

**Pro:** Captures sekis, complex life/death, partial-territory shapes.

**Con:** Another model dependency. Requires a labeled life/death dataset.
Out of scope for "fix the scoring bug" — this is its own research arc.

## Recommendation

**Ship option C first** (eye-counting heuristic + TT) **with optional
fallback to option B for cases C is silent on.** That is:

1. Use eye-counting as the primary life/death classifier:
   - Groups with ≥2 eye-like points → alive
   - Groups with 0 or 1 eye-like points → dead → remove before TT
2. The cleanup is purely a *scoring-time* transform — the GoBoard itself
   isn't mutated during self-play. (This way old training data scored
   under TT is still valid, just incomplete.)
3. Optional later: bolt on Benson's algorithm to recover edge cases the
   eye heuristic misses. Skip option A unless the strength gain from C
   alone isn't enough.

### Why C over A

Option A (cleanup playout) sounds more principled but it has a
non-trivial implementation cost — you need a cleanup-search subroutine,
a stopping rule, and to plumb it into the training pipeline. Option C is
~40 lines of pure Python and gets the README's pathology right today.
Once we have data showing C is the bottleneck, we revisit.

### Why not D

D is the right answer for production strength but it's a separate
research arc. The point of 12g.1 is "fix the scoring bug" — i.e. make
the training signal sound, not optimize endgame.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Eye heuristic mis-classifies real positions | Medium | Test set of ~100 labeled positions; track FN/FP rate, accept <5% |
| Removing "alive" stones inflates one side's score | Low (we mark alive on ≥2 eyes) | Test set covers 2-eye groups; manual audit on misses |
| Training drift — old data scored under TT, new under TT-with-cleanup | Medium | Keep TT as a fallback flag (`score_with_cleanup=True`); regenerate self-play data with new scorer; don't mix old + new in one training run |
| Eye-counting cost dominates self-play | Very low | O(N) where N = 81 (9x9). Microseconds per call. |

## Implementation: where does the fix live?

**Self-play scoring is in Python.** The Odin/C++ `score()` procs match the
TT contract and are the *runtime* scorer — used inside MCTS terminal-value
backups. Changing them affects MCTS search; that's out of scope for 12g.1
(MCTS searches positions that may not be "finished", so cleanup-style
scoring there would be wrong).

The fix should sit at the *training* boundary:
- `score_with_cleanup(board) -> float` lives in Python (`alpha_go/` upstream
  or a small helper in autogodin's python/parity/).
- Self-play loop: when a game ends, call `score_with_cleanup` to produce
  the training label, then `score` (the runtime TT) keeps its current
  meaning during MCTS.

This keeps Odin/C++ surgery to zero and concentrates the fix in the
training data pipeline.

## Acceptance criteria mapping

| Criterion | Status |
|---|---|
| design.md exists with chosen approach + risk assessment | This file |
| Python prototype scoring 100 endgame positions correctly | `prototype.py` + `test_positions.py`, run with `python prototype.py --self-test` |

## Prototype validation

See `prototype.py`. Synthetic test set in `test_positions.py` covers:

- 25 1-eye groups (must be classified dead)
- 25 2-eye groups (must be classified alive)
- 25 mixed positions with both kinds
- 25 corner/edge groups with eye-like points that are *not* real eyes
  (e.g., diagonal-only protection)

Each position is checked against TT-with-cleanup score and the
ground-truth label. Acceptance: ≥95% correct on the test set; document
failure modes.

## Out of scope

- Seki detection (very rare on 9x9, not in this acceptance set)
- Bent-four-in-the-corner and other death-shape pattern recognition
- Cleanup playout (option A) — folder name + bead reserved for future
  follow-up if the eye heuristic underperforms
- A real life/death NN (option D) — separate research arc
- Integrating the new scorer into upstream autogo's self-play loop — that's
  upstream's call; we ship the Python prototype as a drop-in, and they
  can adopt or not
