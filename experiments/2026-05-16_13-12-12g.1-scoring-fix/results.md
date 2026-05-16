# Results: 12g.1 Tromp-Taylor scoring fix (prototype)

## Run

```bash
cd experiments/2026-05-16_13-12-12g.1-scoring-fix
python prototype.py --self-test
```

## Output

```
Self-test: 100/100 positions correct (100.0%)

Cleanup changed score on 51/100 positions.
  delta (|cleanup - TT|): min=75.0 mean=78.6 max=81.0
  e.g. one_eye_isolated_0_0_pad5: TT=-88.5, cleanup=-7.5, delta=+81.0
  e.g. one_eye_isolated_0_4_pad6: TT=-88.5, cleanup=-7.5, delta=+81.0
  e.g. one_eye_isolated_4_0_pad7: TT=-88.5, cleanup=-7.5, delta=+81.0
```

## What this shows

- The eye-counting heuristic correctly classifies the synthetic set:
  2-eye structures stay alive (no removals); 1-eye structures are
  removed before TT.
- On the 51 fixtures with dead groups, the **pure-TT score is wrong by
  75-81 points**. This isn't subtle — TT can flip the winner entirely
  because it credits the dead group with both its stones AND the
  encircled eye as territory.
- The cleanup-aware score is the right answer for those positions.

## What's *not* shown

- Real-game endgames (no actual self-play data here). The synthetic set
  is designed to test the heuristic's two failure modes (false positives,
  false negatives) on unambiguous shapes; it does not stress-test sekis,
  false eyes, or complex life/death races. Those go in the `slg` /
  v0.2 follow-ups described in design.md.
- Performance: the prototype is plain Python, ~ms per board. Not a
  bottleneck for self-play (which spends seconds per game in MCTS), but
  not yet ported to Odin. If/when adopted upstream, a port lives in
  python/parity/ or directly in upstream autogo.

## Acceptance criteria status

| Criterion | Status |
|---|---|
| design.md with chosen approach + risk assessment | ✓ `design.md` |
| Python prototype scores 100 endgame positions correctly | ✓ 100/100 |

## Suggested next steps (out of scope for 12g.1)

1. **Drop the cleanup into autogo's self-play scoring.** The right place
   is upstream's training-data generation; ship the prototype as a PR
   to ericjang/autogo or as a wrapper in autogodin's python/parity/.
2. **Quantify training effect.** Train a model with and without the
   cleanup-aware scorer; compare endgame strength on a fixed test set.
   This is the real win — design.md hypothesizes it; data lives in a
   separate experiment.
3. **Bolt on Benson's algorithm** (option B from design.md) if the eye
   heuristic shows misses on real-game endgames.
4. **Port to Odin** only after the Python prototype is validated
   end-to-end in the training loop — premature porting eats time.
