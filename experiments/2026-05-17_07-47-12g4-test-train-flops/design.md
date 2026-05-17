# 12g.4: test-time vs train-time compute trade-off

Date: 2026-05-17 · Bead: autogodin-12g.4 · Parent: autogodin-12g (research epic)

## Question

At a fixed total inference-FLOP budget per move, does compute go further
spent on **more MCTS sims with a smaller value/policy net**, or **fewer
sims with a bigger net**?

This is the "P ≈ NP" framing from the README applied empirically: can
function approximation replace search?

## Methodology

For each of K candidate (model_size, mcts_sims) pairs that match a chosen
per-move FLOP budget, train a model to a comparable level of supervision
and play a round-robin gauntlet among them. Compute Bayeselo ratings (or
just pairwise winrate matrices) and compare.

Three budgets are run so the curve isn't a single point: see if "small
model + many sims" wins consistently, or only at some compute level, or
loses at all levels.

### Matched-FLOP budgets

Per-move FLOP estimates use the standard `2 × params × tokens_per_pass`
heuristic; for our 9×9 ResNets, tokens_per_pass ≈ 81 (one per cell).

| pair name | model | params | FLOPs / forward | sims / move | FLOPs / move |
|---|---|---:|---:|---:|---:|
| **budget-S** at ~5 GFLOPs/move |
| S-small | 128ch × 10b ("3M") | ~3.0 M | ~486 M | 10  | 4.9 G |
| S-mid   | 192ch × 12b        | ~7.5 M | ~1.21 G | 4  | 4.9 G |
| S-big   | 256ch × 14b ("18M")| ~18 M  | ~2.92 G | 2  | 5.8 G |
| **budget-M** at ~50 GFLOPs/move |
| M-small | 128ch × 10b | ~3.0 M | ~486 M  | 100 | 48.6 G |
| M-mid   | 192ch × 12b | ~7.5 M | ~1.21 G | 40  | 48.4 G |
| M-big   | 256ch × 14b | ~18 M  | ~2.92 G | 17  | 49.6 G |
| **budget-L** at ~200 GFLOPs/move |
| L-small | 128ch × 10b | ~3.0 M | ~486 M  | 400 | 194 G |
| L-mid   | 192ch × 12b | ~7.5 M | ~1.21 G | 165 | 199 G |
| L-big   | 256ch × 14b | ~18 M  | ~2.92 G | 70  | 204 G |

Pure-argmax (sims=1) at the L budget would require a ~600M-param model.
Not realistic to train at 9×9 in this experiment; instead the smallest
`sims` in each row is the closest argmax-like operating point.

Numbers via `flops.py`. The (192ch × 12b) "mid" config is interpolated —
not in upstream `MODEL_CONFIGS`, added in this experiment's `models.py`.

### Training protocol

Each (channels × blocks) config trains from scratch on the same dataset
to control for data:

- Dataset: bpoC `selfplay-it0..it3` (the ~180k carry-forward dataset
  from `experiments/2026-05-17_07-40-bpoC-rerun-postfix`). Same data
  for all three model sizes.
- Hyperparams: AdamW, lr=1e-3 cosine, batch 512, time_budget 900 s
  per model (matches bpoC `train.py`).
- Train each model once; reuse the checkpoint across all 3 sim counts
  for that model. So `S-small`, `M-small`, `L-small` all use the same
  trained `128ch×10b` checkpoint with different MCTS sims at eval.

Total training compute: 3 model sizes × 900 s = 45 min on L4.

### Gauntlet

For each FLOP budget, play a round-robin among the 3 (model, sims) cells
at that budget, plus a fixed reference panel:

- `random` (uniform policy, no NN)
- bpoC iter0 with 200 sims (random-init baseline)
- bpoC iter4 with 200 sims (carry-forward champion)

So per budget: 3 candidates + 3 reference = 6 agents → 15 unique pairings
× 30 games (15 as black, 15 as white) × 2 sides = ~450 games per budget.

Three budgets → ~1,350 evaluation games. At bpoC's observed 22-28 g/min
on L4 (mostly tree-walk-bound at these sim counts), ~50-60 minutes per
budget = ~3 hours of eval.

### Cost estimate (L4 IN2 @ $0.44/hr)

| segment | wall | spend |
|---|---:|---:|
| bootstrap + dataset stage | 15 min | $0.11 |
| train 3 model sizes | 45 min | $0.33 |
| eval gauntlet × 3 budgets | 3 h | $1.32 |
| download artifacts + destroy | 5 min | $0.04 |
| **estimated total** | **~4.1 h** | **~$1.80** |

(Could halve with shorter time_budget per train and fewer games per
pairing if results look clean early.)

## Reference panel rationale

The bpoC iter0 / iter4 anchors test whether differences within the
(model_size, sims) sweep are large vs. an external scale. If all three
budget-L cells beat iter4 by ~ identical margins, the test is
inconclusive at that budget; if one clearly outscores the others vs the
external panel, that's the signal.

Random is a sanity floor: every trained model should beat random with
margin > random's own noise floor (Wilson 95% CI at 30 games per side
is ±15-18%, so we need ≥ 70% winrate vs random for a "clearly beats
random" claim).

## Outputs

- `flops.py` — calculator: (config, sims) → FLOPs/move.
- `models.py` — registers the missing `192ch × 12b` config.
- `run.py` — orchestrator (mirrors bpoC's pattern): train → eval.
- `report.md` — final numbers + per-budget pairwise winrate matrix.
- `figures/` — ELO bars, FLOPs-vs-winrate scatter.

## Non-goals

- **Strength claim against upstream baselines.** Same caveat as bpoC.
  This is an *internal* sweep: which of these arrangements of OUR
  compute works best. Not an absolute claim.
- **MuP transfer.** Models trained from scratch per size, no MuP base
  shared. Closer to "naive" model scaling.
- **Mid-sim refinement.** We don't sweep sims densely; just 3 cells per
  budget. If results suggest a clear winner, follow-up sweeps land in
  separate experiments.

## What this validates / doesn't

Validates: a small but real empirical answer about where to spend the
next FLOP in this codebase's regime.

Does NOT validate: anything about scaling LAWS — the spread of model
sizes is one decade (3M → 18M), too narrow for power-law fits. See
`autogodin-12g.3` for the scaling-laws sweep.
