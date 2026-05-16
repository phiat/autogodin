#!/usr/bin/env python3
"""cz9: dict vs flat (numpy-view) Python evaluator.

Same workload as ydh.2 bench.py:
  9x9 Go, 1600 sims/move × 32 moves, uniform-policy evaluator, c_puct=1.0,
  no Dirichlet, no PCR, max_depth=100. Single-threaded.

Two paths:
  - "dict":  evaluator returns dict[int, float] + value; trampoline iterates
             dict items and copies into ctypes buffers (old path).
  - "flat":  evaluator returns a dense float32 policy of length
             size*size + 1 (index = action id, pass = size*size); trampoline
             does numpy fancy indexing + memmove into ctypes buffers. No
             dict alloc, no per-element Python writes.

Reports sims/sec for each + the speedup. The dense-policy shape matches what
an NN forward pass naturally produces; this bench's synthetic uniform
evaluator approximates the cost surface (alloc + fancy index + memmove).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import alpha_go_odin as ao

SIZE = 9
KOMI = 7.5


def make_dict_evaluator(pass_action: int):
    def evaluator(board):
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1
        p = 1.0 / n
        out = {a: p for a in legal}
        out[pass_action] = p
        return out, 0.0
    return evaluator


def make_flat_evaluator(pass_id: int):
    """In-place flat evaluator: writes (action, prob) prefixes into the
    caller-owned scratch arrays. Returns (count, value)."""
    def evaluator(board, scratch_a, scratch_p):
        legal = board.get_legal_moves_flat()
        n_legal = len(legal)
        scratch_a[:n_legal] = legal
        scratch_a[n_legal] = pass_id
        p = np.float32(1.0 / (n_legal + 1))
        scratch_p[: n_legal + 1] = p
        return n_legal + 1, 0.0
    return evaluator


def run_trial(path: str, num_sims: int, num_moves: int) -> tuple[int, float]:
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = 1.0
    cfg.lambda_ = 0.0
    cfg.max_depth = 100

    board = ao.GoBoard(SIZE, KOMI)
    dict_ev = make_dict_evaluator(ao.PASS_ACTION) if path == "dict" else None
    flat_ev = make_flat_evaluator(SIZE * SIZE) if path == "flat" else None  # pass id = size*size in flat path

    total_sims = 0
    t0 = time.perf_counter()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        tree = ao.MCTSTree(board, cfg, seed=0)
        if path == "dict":
            tree.run_simulations(num_sims, dict_ev)
        else:
            tree.run_simulations_flat(num_sims, flat_ev)
        action = tree.select_action(0.0)
        if action == ao.PASS_ACTION:
            board.pass_move()
        else:
            if not board.play_flat(action):
                board.pass_move()
        total_sims += num_sims
    return total_sims, time.perf_counter() - t0


def bench(path: str, trials: int, num_sims: int, num_moves: int) -> tuple[float, float]:
    # Warmup
    _ = run_trial(path, num_sims, num_moves)
    rates = []
    for _ in range(trials):
        sims, dt = run_trial(path, num_sims, num_moves)
        rates.append(sims / dt)
    m = statistics.mean(rates)
    sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    return m, ci


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-sims", type=int, default=1600)
    p.add_argument("--num-moves", type=int, default=32)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    print(f"=== cz9: dict vs flat evaluator ===")
    print(f"config: {args.num_sims} sims/move x {args.num_moves} moves/trial = "
          f"{args.num_sims * args.num_moves:,} sims/trial; {args.trials} trials")
    print()

    for path in ("dict", "flat"):
        m, ci = bench(path, args.trials, args.num_sims, args.num_moves)
        print(f"  {path:>4}: {m:8,.0f} +- {ci:6,.0f} sims/sec  (95% CI, n={args.trials})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
