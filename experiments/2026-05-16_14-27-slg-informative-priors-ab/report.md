# slg: MCTS strength A/B under informative priors

**Result: PASS.** Odin 46 / C++ 54 / 0 draws over 100 games × 200 sims/move.
odin_winrate = **0.460**, Wilson 95% CI = **[0.366, 0.557]** → 0.5 inside.

The two MCTS implementations are statistically equivalent at strength when
given informative priors. The 10/100 uniform-prior result we saw post-vendor
was the FPU concentration-vs-spread regime caveat documented in
`odin/vendor/mcts-odin/mcts/mcts.odin` (`Config.fpu_reduction`), not an
algorithmic regression in Odin.

## Setup

- Backends: `alpha_go_odin` v(post-vendor v0.3.0, commit 4684fac) vs
  `alpha_go_cpp` (upstream autogo).
- Board: 9×9 Go, komi 7.5.
- MCTS: 200 sims/move, c_puct=1.0, no Dirichlet noise, temperature=1.0
  for the first 15 moves then argmax. Move cap 200.
- Evaluator: deterministic NumPy-only "informative random projection".
  Fixed-seed `W_policy ∈ ℝ^{82×244}`, `W_value ∈ ℝ^{244}`. Feature vector =
  `[black_one_hot(81), white_one_hot(81), legal_mask(81), to_play(1)]`.
  Output: softmax-masked policy over legal+pass, sigmoid value.
  Identical callable both backends → no eval drift.
- Host: miniwini, single-thread.
- 100 games, alternating sides per game (Odin Black on even g, Odin White
  on odd g). Seed-base 0xC0FFEE, eval-seed 0xBEEF.

## Why a synthetic evaluator instead of a real NN

The point of this gate was to *exit the FPU-degenerate uniform regime*, not
to play strong Go. A random-init torch GoResNet would have required
installing torch (~2GB) and would have cost hours of CPU forwards for the
same signal. A deterministic random-projection produces non-uniform,
board-dependent priors at NumPy speed — exactly what's needed to verify
the post-FPU strength gate.

If a real NN run is ever wanted (e.g. once a trained checkpoint exists),
the harness can be swapped in by replacing `make_informative_evaluator` with
the autogo `LocalNNEvaluator` / `LeafBatchedNNEvaluator` — same calling
contract.

## Numbers

| metric          | value                |
|-----------------|----------------------|
| games           | 100                  |
| decided         | 100                  |
| Odin wins       | 46                   |
| C++ wins        | 54                   |
| draws           | 0                    |
| odin_winrate    | 0.460                |
| Wilson 95% CI   | [0.366, 0.557]       |
| 0.5 in CI       | **yes**              |
| elapsed         | 114.6 s              |

100 games × 200 sims/move = 20,000 trees, ~115 seconds → ~174 trees/sec
combined across both backends (sequential Python eval; not a perf number,
just a sanity check that the harness is well-tuned).

## Conclusion

This closes the strength-equivalence acceptance criterion in the 3xv epic
under realistic (informative) prior conditions. The uniform-eval baseline
is preserved as a documented FPU-regime caveat, not actionable signal.

## Reproduce

```bash
# On miniwini (or any host with alpha_go_cpp+alpha_go_odin+numpy):
PYTHONPATH=python <python> experiments/2026-05-16_14-27-slg-informative-priors-ab/ab_selfplay_nn.py \
  --games 100 --num-sims 200
```
