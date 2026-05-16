# Code Review — Post-Foundation Refactor (2026-05-16)

Reviewer: code-reviewer agent. Scope: `odin/alpha_go/go_game.odin`, `mcts.odin`, `exports.odin`, `python/alpha_go_odin/__init__.py`.

Context: foundation refactor just landed (commit f8f0308). ydh.2 throughput
went 2.9k → 7.9k sims/sec (~93% of C++ 8.5k baseline). Peak RSS now 62 MB
flat across 32 moves (was 2.5 GB and growing).

## CRITICAL

### 1. ~~Wrong-allocator `delete` on all `group`/`libs`/`stack` temp-arena slices~~ — FALSE POSITIVE

**Status:** Closed `autogodin-dsi` as not-a-bug after verification.

The reviewer claimed `delete([dynamic])` without an explicit allocator routes
to `context.allocator`. That's wrong. Odin's
`runtime.delete_dynamic_array(array)` calls
`mem_free_with_size(..., array.allocator, ...)` — the *stored* allocator from
make-time. So `delete(stack)`, `delete(group)`, `delete(libs)` correctly free
to `context.temp_allocator` when the dynamic was made with it.

(Slices like `visited`/`lib_visited` *do* need the explicit allocator since
slices have no stored allocator field — the code does this right.)

Verified by running `just test` under `-debug` (Odin memory tracker enabled):
37/37 tests pass with zero leak or wrong-allocator diagnostics.

### 2. `run_simulations` double-evaluates the root — wipes Dirichlet noise

**File:** `odin/alpha_go/mcts.odin`
**Lines:** 423–427 vs 349–373
**Confidence:** 85

`run_simulations` pre-evaluates root and fills `logP_A`, then optionally adds
Dirichlet noise. The first `perform_playout` call then sees `N == 0` and
re-evaluates root, overwriting the noised priors with raw network output.

Net effects:
- One wasted evaluator call per `run_simulations` invocation (measurable with
  an NN evaluator).
- With `dirichlet_alpha > 0`, the noise is silently erased before the first
  simulation can use it.

`run_simulations_batched` avoids this with a `len(logP_A) == 0` guard. Apply
the same guard in `perform_playout`'s `N == 0` branch (or remove the explicit
root pre-eval and let `perform_playout` handle it).

## IMPORTANT

### 3. `eval_state_storage` pointer-stability relies on unasserted capacity

**File:** `odin/alpha_go/mcts.odin`
**Lines:** 510, 554–555
**Confidence:** 82

`eval_states` holds pointers into `eval_state_storage`'s backing array, which
is pre-reserved to `target` entries so the array won't reallocate. Today the
gather loop appends ≤ 1 entry per iteration with exactly `target` iterations,
so the invariant holds. But it isn't asserted anywhere — a future refactor
that adds a second append site would silently dangle every `eval_states`
pointer.

**Fix:** add `assert(len(eval_state_storage) < target, ...)` before each
append. Or switch to a fixed `[]GoBoard` of length `target` indexed by a
counter.

### 4. `compute_puct_scores` allocates a `map[int]f32` per PUCT step

**File:** `odin/alpha_go/mcts.odin`
**Lines:** 158, 174–175
**Confidence:** 88

`compute_puct_scores` returns a heap-allocated map; `select_action_puct`
deletes it immediately. This runs once per descent step × num_simulations =
thousands of map allocations per `run_simulations` call.

**Fix:** inline the argmax directly into `select_action_puct` (no map
needed — we only want the best action). Estimated 3–8% throughput recovery,
closing more of the gap to C++ baseline.

### 5. `path` allocated on heap while all batch-local data is on temp arena

**File:** `odin/alpha_go/mcts.odin`
**Lines:** 506, 496
**Confidence:** 80

`path := make([dynamic]int, 0, 8)` uses `context.allocator` (heap) while
everything else in the batch (`pending`, `eval_state_storage`, `eval_states`,
`deltas`) uses `context.temp_allocator`. Cleanup is correct but mixes
lifetimes implicitly.

**Fix:** make `path` temp-allocator-allocated too, drop the explicit
`delete(p.path)` cleanup, and let the per-batch `free_all(temp_allocator)`
reclaim it.

### 6. MCTS does not legality-check actions before `do_move`

**File:** `odin/alpha_go/mcts.odin` (perform_playout, batched gather)
+ `odin/alpha_go/go_game.odin` (do_move docs)
**Confidence:** 82

`do_move` is documented as "does NOT check legality". `perform_playout`
selects an action from `logP_A` and calls `do_move` directly. With a
well-behaved evaluator (NN trained on legal moves; the test uses
`get_legal_moves_flat`), suicide and other illegal moves never enter
`logP_A`. But the code has no safety net — a misbehaving evaluator silently
corrupts the tree.

**Fix:** either gate `do_move` on `is_legal_flat` (cost: one legality check
per descent step) or document loudly that evaluators MUST zero illegal
actions. For NN evaluators this is implicit via softmax over a mask; the
test evaluator already does it. Risk is medium-low but worth a comment.

### 7. FFI config positional C-ABI binding is fragile

**File:** `python/alpha_go_odin/__init__.py:273–283` + `odin/alpha_go/exports.odin`
**Confidence:** 80

The MCTSConfig setter binds 6 floats + 1 int positionally. Argument-order
mistakes after the foundation refactor would silently swap `temperature` and
`rollout_temperature` with no compile or runtime error. Consider a
struct-based config copy or at least a unit test that verifies each field
round-trips through Python → Odin → readback.

## Summary

The foundation refactor is solid. The two CRITICAL items are inexpensive to
fix and one of them (Dirichlet erasure) is a silent correctness bug for any
training run that uses Dirichlet noise. Items 3–7 are tightening/perf wins.

Estimated combined throughput impact: bug #4 alone is worth 3–8%. The rest
are correctness/robustness rather than perf.
