# Design: mcts-odin integration

**Bead:** `autogodin-ci2`
**Date:** 2026-05-16
**Status:** Proposal, awaiting sign-off.

## TL;DR

Vendor `mcts-odin`'s `mcts/` directory under `odin/vendor/mcts-odin/` at a
pinned commit. Replace `odin/alpha_go/mcts.odin` with a thin adapter
(`go_adapter.odin`) plus `import "../vendor/mcts-odin/mcts"`. Keep
`exports.odin`'s C-ABI symbol surface stable so the Python ctypes shim and
upstream `alpha_go_cpp` callers don't break.

Estimated lift on ydh.2: 7,927 → ~10–12k sims/sec (current cap is Python
FFI marshalling). Estimated cost: 1 work-session.

## Why now

mcts-odin (sibling repo) landed 7 perf levers since extraction that target
exactly the hot spots autogodin's `mcts.odin` is dominated by:

- Packed slot arrays (replaces our `map[int]int` / `map[int]f32` per node)
- Linear-space priors (no `math.exp` in the PUCT inner loop)
- Branchless PUCT scan
- Per-tree scratch arena (vs our `free_all(temp_allocator)`)
- Eval scratch hoisting onto `Tree`
- Subtree reuse across moves (`reuse_root(action)`)
- Node-pool capacity reserve

Bench: mcts-odin **13,602 sims/sec** (in-process Odin evaluator) vs
autogodin **7,927 sims/sec** (Python ctypes evaluator). Not strictly
apples-to-apples — the FFI shape differs — but the gap is structural, and
cherry-picking individual wins into autogodin's map-based `mcts.odin` would
amount to rewriting the MCTS layer to match mcts-odin's design anyway.

mcts-odin's [`docs/EMBEDDING.md`](../../mcts-odin/docs/EMBEDDING.md) already
documents the embed contract and explicitly names autogodin as the worked
C-ABI example.

## Options considered

### A — Vendor `mcts/` as Odin dep (**recommended**)

```
autogodin/
  odin/
    alpha_go/
      go_game.odin          (unchanged — board only)
      go_adapter.odin       NEW: Game vtable wrapping GoBoard
      exports.odin          REWIRED: drives mcts.Tree instead of MCTSTree
      mcts.odin             DELETED
    vendor/
      mcts-odin/
        mcts/               COPY of mcts-odin/mcts/, pinned commit hash
        LICENSE
        VERSION             one-liner: commit hash + sync date
```

**Pro**

- Self-contained build (no sibling-repo dependency at clone time).
- Inherits every mcts-odin perf win automatically when we sync VERSION.
- Smallest behavioral risk: keeps autogodin's GoBoard (parity-validated)
  + Python FFI surface (Python shim unchanged).
- Pinned commit means mcts-odin churn doesn't surprise us.

**Con**

- Manual sync workflow needed (refresh the `mcts/` copy + bump VERSION).
- ~600 LOC added to repo as vendored code (offset by ~770 LOC deleted
  from `mcts.odin`).

**Effort:** Adapter ~100 LOC. exports.odin rewire ~200 LOC delta. Test +
parity + bench: half-day.

### B — Cherry-pick perf changes into autogodin's `mcts.odin`

The wins are predicated on the packed-slot design. Porting them means
rewriting `mcts.odin` to use packed slots anyway. After that, future
mcts-odin work doesn't flow back here without re-porting each change.
**Worst of both worlds.** Rejected.

### C — Full migrate to `mcts/` + `games/go/`

Drop autogodin's `go_game.odin` too; depend on mcts-odin's `games/go/`.
Smaller autogodin surface, but `games/go/` is scoped as a *demo*, not a
stable API surface (per mcts-odin's `bd memories`: "Includes 3 demo
games"). Risk of breakage when mcts-odin evolves its games/ layout.

**Rejected** — keep our own parity-validated Go board, take only the
generic MCTS package as a dep.

## Migration plan (option A)

1. **Snapshot mcts-odin.** Record current commit; copy `mcts-odin/mcts/`
   tree → `autogodin/odin/vendor/mcts-odin/mcts/`. Write
   `vendor/mcts-odin/VERSION` with commit hash + sync date + LICENSE.

2. **Write `go_adapter.odin`.** Game vtable mapping autogodin's
   `GoBoard` + `do_move`/`undo_move`/`CaptureRecord`/`MoveDelta` to
   mcts-odin's `Game` struct. Action-space convention: MCTS sees actions
   in `[0, size*size]` with the pass-action at `size*size` (mcts-odin's
   convention); the adapter translates to autogodin's `PASS_ACTION = -1`
   internally. Modeled on `mcts-odin/games/go/game.odin`.

3. **Rewire `exports.odin`.** Replace every `ag.MCTSTree` reference with
   `mcts.Tree`; preserve the `alphago_*` C-ABI symbol names exactly so
   `python/alpha_go_odin/__init__.py` keeps loading without changes.
   Notable signature deltas to absorb in the C-ABI wrapper:
   - mcts-odin uses an *output-count* return from the evaluator
     trampoline; autogodin's current trampoline returns via a
     pre-allocated dict. Adapt at the trampoline boundary, keep the
     Python contract identical.
   - `select_action` / `get_action_probabilities` semantics match.

4. **Delete `odin/alpha_go/mcts.odin`.** ~770 LOC gone.

5. **Update `odin/tests/mcts_test.odin`** to drive the new API. Existing
   tests assert visit counts and tree-size — these should hold semantically
   even though internal layout differs.

6. **Verify gates:**
   - `just test` — all 37 Odin tests green
   - `just parity` — fingerprint unchanged (board-level, MCTS-agnostic)
   - `random_games_dual.py` cross-language — still `109bd08a…`
   - `experiments/.../bench.py --backend odin` — sims/sec ↑
   - 100-game A/B re-run — Wilson CI still brackets 0.5

7. **Update `README.md`** with new throughput table and the vendored-dep
   note.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Adapter has a perspective-sign bug | Low (mcts-odin's games/go is a worked example) | The 100-game A/B catches it |
| mcts-odin's Game contract assumptions don't match GoBoard (e.g., `is_terminal` cached at create_node time) | Medium | Adapter validates by re-running parity + A/B |
| Pinned commit drifts; we miss perf upstream | Low | Schedule a quarterly sync; bd memory already reminds to check |
| Python ctypes trampoline shape changes (eval returns count vs dict) breaks autogo's downstream code | Low | Trampoline boundary is internal; Python-facing API stays a dict callable |

## What we keep, what we lose, what we gain

**Keep:**
- `python/alpha_go_odin/` ctypes shim — public API unchanged
- `odin/alpha_go/go_game.odin` — parity-validated board, with `do_move`/`undo_move`
- `odin/alpha_go/exports.odin` — symbol surface unchanged from Python's view
- Parity fingerprint `109bd08a…`
- Ability to run ydh.2 bench / random_games_dual / ab_selfplay against the cpp backend

**Lose:**
- `odin/alpha_go/mcts.odin` (~770 LOC; the inferior implementation)
- Map-based per-node storage (replaced with packed slots)

**Gain:**
- Subtree reuse across moves (immediate gain for self-play / training loops)
- Linear-space priors, branchless PUCT, scratch-arena discipline
- Future mcts-odin perf work flows in via VERSION bumps
- One less codebase to maintain in autogodin

## What this does NOT cover

- Updating `alpha_go_cpp` (upstream): out of scope; that's a sibling project.
- Apples-to-apples bench (`autogodin-xv8`): comes after this lands.
- mcts-odin upstream changes (`mcts-odin-w39.4`): mcts-odin tracks autogodin
  perf changes; this design is the opposite direction (autogodin tracks
  mcts-odin perf via VERSION pins).

## Sign-off checklist

- [ ] User approves option A
- [ ] Pinned commit hash chosen (current HEAD of mcts-odin: `32d10b7`)
- [ ] Vendor layout sign-off (`odin/vendor/mcts-odin/mcts/`)
- [ ] OK to delete `odin/alpha_go/mcts.odin` once the new path is green
