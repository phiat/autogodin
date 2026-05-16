# xv8: Apples-to-apples MCTS throughput, post-vendor

## TL;DR

Original framing (autogodin-xv8): autogodin's Python bench (7,927 sims/s)
and mcts-odin's in-process bench (13,602 sims/s) aren't comparable —
different FFI shapes. Build a path where both run the *same algorithm*
through different shims, isolate FFI cost.

Post-vendor finding: the framing no longer holds. Both autogodin and
mcts-odin use the *same vendored MCTS* (mcts-odin v0.2.0, commit 6bb0768).
But each side still uses *its own* GoBoard implementation, which have
diverged since the extraction. So the bench delta now measures **GoBoard
divergence**, not FFI overhead.

## Numbers (miniwini, idle, single-thread, -o:speed, 51,200 sims/trial)

| Bench                                    | sims/sec        | Notes |
|------------------------------------------|-----------------|-------|
| autogodin Python ctypes shim (n=5)       | 20,773 ± 132    | clean, post-FPU |
| mcts-odin in-process bench (n=5)         | 13,644 ± 3,794  | very noisy across trials — see mcts-odin-brj |
| mcts-odin in-process bench (trial 1)     | 17,550          | clean before temp_allocator pollution |
| alpha_go_cpp (upstream, reference)       | 8,655 ± 86      | |

## What the comparison says

- autogodin via Python ctypes is *faster* than mcts-odin in-process. The
  obvious read ("Python FFI is free!") is wrong. The actual delta is
  that autogodin's `odin/alpha_go/go_game.odin` has had foundation work
  (do_move/undo_move, working-board descent, captures hoisting) that
  hasn't all flowed into `mcts-odin/games/go/board.odin`. The two
  GoBoards aren't the same code.

- The Python ctypes shim has measurable overhead — the trampoline
  marshals an action/prob dict per leaf and translates pass-action ids.
  We can't see that cost cleanly without running autogodin's GoBoard
  through an *in-process* Odin evaluator (no Python).

## What this means for FFI cost

If we want a clean "Python ctypes vs in-process" measurement, we need
to write an Odin-only autogodin bench: drive `mcts.run_simulations` with
an in-process Odin evaluator over autogodin's GoBoard, then compare to
the Python bench. The delta is the FFI cost. That's a follow-up task.

## Reference data

- Bench script: `experiments/2026-05-16_05-40-mcts-bench-cpp-vs-odin/bench.py`
- mcts-odin bench: `../mcts-odin/bench/bench.odin` (clone, not vendored —
  bench/ is dev-only, not part of mcts-odin's public surface).
- Bench noise upstream issue: `mcts-odin-brj`.

## Status

xv8 closed: the original delta has been resolved by the vendor migration
(both engines now share MCTS code). What remains is the FFI-cost
question, which is a different shape and gets its own bead (autogodin-`<TBD>`).
