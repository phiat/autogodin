# cz9: dict vs flat Python evaluator — flat wins +10.9% on miniwini

## Setup

ydh.2-style bench: 9×9 Go, 1600 sims/move × 32 moves × 3 trials. Uniform
prior + value=0 evaluator. Single-threaded ctypes path through
`alpha_go_odin`. Post-ydh.6 build (zkq + 373 + 5km landed).

Two evaluator shapes compared:

- **`dict`**: legacy `EvaluatorFn`. Evaluator builds `dict[int, float]`
  + value; trampoline iterates dict items and writes into MCTS's ctypes
  out buffers via per-item `__setitem__`.
- **`flat`**: new `FlatEvaluatorFn`. Trampoline holds a pair of
  numpy scratch ndarrays sized to `n_actions`; evaluator writes (action,
  prob) prefixes into them and returns `(count, value)`. Trampoline
  memmoves the prefix into MCTS's ctypes out buffers, using ctypes data
  pointers resolved ONCE per trampoline (not per leaf).

## Result

| host           | dict (sims/sec) | flat (sims/sec) | lift   |
|----------------|----------------:|----------------:|-------:|
| local i9       | 46,794 ± 431    | 48,709 ± 1,383  | +4.1%  |
| **miniwini**   | 49,201 ± 841    | **54,541 ± 616** | **+10.9%** |

On miniwini the flat path puts the alpha_go_odin ctypes shim at
**6.26× alpha_go_cpp** (was 5.51× via the dict path).

## What didn't work (and why)

Two earlier flat designs were *slower* than the dict path on this bench:

1. **Evaluator-fills-numpy-view-in-place over the ctypes out buffer**:
   created `np.ctypeslib.as_array(out_actions, shape=(max_n,))` per leaf.
   That construction is non-trivial; profile showed the wrap dominated.
2. **Evaluator returns dense float32 policy; trampoline does fancy index
   + memmove**: the per-leaf `.ctypes.data` access on the temporary
   `actions`/`probs` ndarrays built a fresh `numpy._internal._ctypes`
   object every time (25,616 `_internal.__init__` calls visible in
   cProfile). The ctypes-view alloc cost exceeded the saved dict overhead.

The winning design (current implementation) avoids both:
- Scratch ndarrays are allocated ONCE per `run_simulations_flat` call
  (not per leaf).
- Their `.ctypes.data` pointers are resolved ONCE and captured in a
  closure (not per leaf).
- Per leaf: evaluator does two slice-assigns into the scratch arrays;
  trampoline does two `ctypes.memmove` calls with the pre-resolved
  pointers.

## Pre-NN-integration caveat

This bench's evaluator is so cheap (assigning a single uniform scalar)
that the trampoline is a measurable fraction of per-leaf cost. A real NN
forward pass (~0.1-5ms) makes the trampoline a much smaller fraction —
the dict overhead becomes irrelevant. The flat path matters more for the
NN-eval transition when:

- The NN naturally produces a dense float32 policy (no dict to convert
  back from).
- An inner-loop training/inference call needs predictable per-leaf cost.

For *batched* NN-eval workloads the picture is different again — there
the per-batch Python work dominates (and `cz9` doesn't address it; see
follow-up at the bottom of `ydh.3`'s `results_python.md`).

## Reproduce

```bash
PYTHONPATH=python <python> experiments/2026-05-16_16-29-cz9-flat-evaluator/bench.py
```
