#!/usr/bin/env python3
"""i5d: thread-scaling bench for run_simulations_threaded.

Sweep n_threads ∈ {1, 2, 4, 8} against sequential run_simulations.
9x9 Go, 1600 sims/move × 32 moves × 3 trials. Uniform-policy Python
evaluator (worst case for threading — GIL serializes the evaluator
body; only Odin-side descent/expand/backup parallelize).
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


def make_uniform_evaluator(pass_action: int):
    def evaluator(board):
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1
        p = 1.0 / n
        out = {a: p for a in legal}
        out[pass_action] = p
        return out, 0.0
    return evaluator


def make_sleep_evaluator(pass_action: int, sleep_us: int):
    """GIL-releasing evaluator: time.sleep releases the GIL, so leaves
    on different threads can overlap. Simulates an NN-eval forward pass
    that calls into a C extension that releases the GIL.
    """
    s = sleep_us / 1_000_000.0
    def evaluator(board):
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1
        p = 1.0 / n
        out = {a: p for a in legal}
        out[pass_action] = p
        time.sleep(s)
        return out, 0.0
    return evaluator


def run_trial(n_threads: int, num_sims: int, num_moves: int, ev_factory=None) -> tuple[int, float]:
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = 1.0
    cfg.lambda_ = 0.0
    cfg.max_depth = 100
    board = ao.GoBoard(SIZE, KOMI)
    ev = (ev_factory or make_uniform_evaluator)(ao.PASS_ACTION)
    total_sims = 0
    t0 = time.perf_counter()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        tree = ao.MCTSTree(board, cfg, seed=0)
        if n_threads <= 0:
            tree.run_simulations(num_sims, ev)
        else:
            tree.run_simulations_threaded(num_sims, n_threads, ev)
        action = tree.select_action(0.0)
        if action == ao.PASS_ACTION:
            board.pass_move()
        elif not board.play_flat(action):
            board.pass_move()
        total_sims += num_sims
    return total_sims, time.perf_counter() - t0


def bench(n_threads: int, trials: int, num_sims: int, num_moves: int, ev_factory=None) -> tuple[float, float]:
    _ = run_trial(n_threads, num_sims, num_moves, ev_factory)  # warmup
    rates = []
    for _ in range(trials):
        sims, dt = run_trial(n_threads, num_sims, num_moves, ev_factory)
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
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--sleep-us", type=int, default=0,
                   help="If >0, evaluator does time.sleep(sleep_us) per leaf "
                        "(GIL-releasing). Shows the upper bound where Odin-side "
                        "descent/expand/backup can actually overlap.")
    args = p.parse_args()

    print(f"=== i5d: thread-scaling sweep ===")
    print(f"config: {args.num_sims} sims/move x {args.num_moves} moves/trial = "
          f"{args.num_sims * args.num_moves:,} sims/trial; {args.trials} trials/cell")
    if args.sleep_us > 0:
        ev_factory = lambda pa: make_sleep_evaluator(pa, args.sleep_us)
        print(f"evaluator: uniform + time.sleep({args.sleep_us}us) per leaf "
              f"(GIL released during sleep)")
    else:
        ev_factory = None
        print(f"evaluator: uniform Python (GIL held throughout)")
    print()

    # Sequential baseline
    m_seq, ci_seq = bench(0, args.trials, args.num_sims, args.num_moves, ev_factory)
    print(f"  sequential:  {m_seq:8,.0f} +- {ci_seq:6,.0f} sims/sec   (baseline)")

    for n in args.threads:
        m, ci = bench(n, args.trials, args.num_sims, args.num_moves, ev_factory)
        speedup = m / m_seq if m_seq > 0 else 0
        print(f"  threaded n={n}: {m:8,.0f} +- {ci:6,.0f} sims/sec   "
              f"{speedup:.2f}x vs sequential")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
