# 4ig: in-process Odin evaluator bench (Python ctypes FFI cost)

## TL;DR

Python ctypes round-trip per leaf evaluator call costs **18.9%** vs an
in-process Odin evaluator. Real NN-eval workloads (torch in Python) won't
notice — the eval cost dominates the FFI cost by orders of magnitude.
Pure-CPU bench harnesses see it clearly.

## Setup

Same config as `experiments/2026-05-16_05-40-mcts-bench-cpp-vs-odin/bench.py`:

| Knob | Value |
|---|---|
| Board | 9×9, komi 7.5 |
| Sims/move × moves/trial | 1600 × 32 = 51,200 |
| Trials | 5, plus 1 warmup |
| MCTS | c_puct=1.0, lambda=0, dirichlet off, max_depth=100, temperature=1.0 |
| Evaluator | Uniform over legal + pass, value=0 |
| Backend | post-FPU vendored mcts-odin (v0.2.0 / 6bb0768) |
| Host | miniwini (idle, load ~0.5), Odin -o:speed |

The only difference: the in-process bench runs `uniform_evaluator` as a
plain Odin proc; the Python bench routes the same logic through the C-ABI
trampoline → Python `EvaluatorFn` (`policy: dict[int, float], value`).

## Results

```
warmup 0: 2.025s, 51200 sims, 25,281 sims/s
trial 0: 2.002s, 51200 sims, 25,570 sims/s
trial 1: 2.002s, 51200 sims, 25,574 sims/s
trial 2: 1.983s, 51200 sims, 25,813 sims/s
trial 3: 2.001s, 51200 sims, 25,591 sims/s
trial 4: 2.004s, 51200 sims, 25,545 sims/s

in-process odin: 25,618 ± 86 sims/sec (95% CI, n=5)
```

| Backend                                  | sims/sec        | vs C++  |
|------------------------------------------|-----------------|---------|
| autogodin in-process Odin evaluator      | **25,618 ± 86**  | 2.96×   |
| autogodin Python ctypes (uniform eval)   | 20,773 ± 132    | 2.40×   |
| alpha_go_cpp (upstream reference)        | 8,655 ± 86      | 1.00×   |

**FFI cost = (25,618 − 20,773) / 25,618 = 18.9% throughput tax.**

## What this means

- For *bench-style* workloads (uniform/fast-to-compute evaluator),
  the ~19% Python ctypes tax is real signal. If we ever need to squeeze
  perf from a CPU-only self-play loop with a cheap policy, an Odin-native
  evaluator path is worth ~20%.

- For *NN-eval* workloads, the evaluator cost is dominated by the model
  forward pass (microseconds to milliseconds per leaf, depending on net
  + batch size). A 19% overhead on the trampoline is a tiny fraction of
  that — the Python shim is effectively free in that regime.

- Headline number for the README post-vendor: 2.40× C++ on the workload
  the upstream bench measures (Python evaluator). The 2.96× in-process
  number is the *ceiling* — what we'd see with an Odin-native evaluator.

## Reproduce

```bash
# On miniwini (or any host with the autogodin checkout):
cd ~/autogodin-work/autogodin
odin build experiments/2026-05-16_12-50-4ig-inprocess-bench \
    -o:speed \
    -out:experiments/2026-05-16_12-50-4ig-inprocess-bench/bench
./experiments/2026-05-16_12-50-4ig-inprocess-bench/bench
```

The bench is hermetic — no Python, no `.so`, no FFI. Just the autogodin
`alpha_go` Odin package + the vendored mcts package.
