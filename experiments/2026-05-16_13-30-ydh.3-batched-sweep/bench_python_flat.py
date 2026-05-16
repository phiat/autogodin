#!/usr/bin/env python3
"""cg0: same batched sweep as bench_python.py but driven by the flat
(no-dict) batched evaluator.

Runs both the legacy dict path AND the flat path in the same harness so
the cell-by-cell uplift is exact. Reports sims/sec and (flat/dict) ratio.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from statistics import mean, stdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import alpha_go_odin as ao

SIZE = 9
KOMI = 7.5
N_ACTIONS = SIZE * SIZE + 1


def make_dict_evaluator(pass_action: int, latency_us: float):
    latency_s = latency_us / 1e6 if latency_us > 0 else 0.0
    def evaluator(boards):
        if latency_s > 0:
            time.sleep(latency_s)
        policies, values = [], []
        for board in boards:
            legal = board.get_legal_moves_flat()
            n = len(legal) + 1
            p = 1.0 / n
            out = {a: p for a in legal}
            out[pass_action] = p
            policies.append(out)
            values.append(0.0)
        return policies, values
    return evaluator


def make_flat_evaluator(pass_action: int, latency_us: float):
    """Writes legal+pass action ids and uniform priors directly into the
    scratch ndarrays. No per-state dict alloc."""
    latency_s = latency_us / 1e6 if latency_us > 0 else 0.0
    def evaluator(views, out_actions, out_probs, out_counts, out_values):
        if latency_s > 0:
            time.sleep(latency_s)
        for i, view in enumerate(views):
            legal = view.get_legal_moves_flat()
            n = len(legal)
            out_actions[i, :n] = legal
            out_actions[i, n] = pass_action
            out_probs[i, :n+1] = 1.0 / (n + 1)
            out_counts[i] = n + 1
            out_values[i] = 0.0
    return evaluator


def run_trial(mode: str, batch_size: int, latency_us: float,
              num_sims: int, moves: int, seed: int):
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = 100

    board = ao.GoBoard(SIZE, KOMI)
    if mode == "dict":
        ev = make_dict_evaluator(ao.PASS_ACTION, latency_us)
    else:
        ev = make_flat_evaluator(ao.PASS_ACTION, latency_us)

    total_sims = 0
    t0 = time.perf_counter()
    for move in range(moves):
        if board.is_game_over():
            break
        tree = ao.MCTSTree(board, cfg, seed=seed * 1000 + move)
        if mode == "dict":
            tree.run_simulations_batched(num_sims, batch_size, ev)
        else:
            tree.run_simulations_batched_flat(num_sims, batch_size, ev)
        action = tree.select_action(0.0)
        if action == ao.PASS_ACTION:
            board.pass_move()
        else:
            ok = board.play_flat(action)
            if not ok:
                board.pass_move()
        total_sims += num_sims
    dt = time.perf_counter() - t0
    return total_sims, dt


def bench_cell(mode: str, batch_size: int, latency_us: float,
               num_sims: int, moves: int, trials: int):
    _ = run_trial(mode, batch_size, latency_us, num_sims, moves, seed=42)
    rates = []
    for i in range(trials):
        sims, dt = run_trial(mode, batch_size, latency_us, num_sims, moves,
                             seed=100 + i)
        rates.append(sims / dt)
    m = mean(rates)
    sd = stdev(rates) if len(rates) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    return m, ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-sims", type=int, default=1600)
    p.add_argument("--moves", type=int, default=32)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--latencies", type=int, nargs="+", default=[0, 100, 1000],
                   help="evaluator latency in microseconds")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    args = p.parse_args()

    print(f"=== cg0: dict vs flat batched evaluator sweep ===")
    print(f"config: {args.num_sims} sims/move x {args.moves} moves x "
          f"{args.trials} trials per cell")
    print()
    header = (f"{'latency':>10} {'batch':>6}   "
              f"{'dict (sims/s)':>15} {'flat (sims/s)':>15} {'ratio':>7}")
    print(header)
    print("-" * len(header))
    for latency_us in args.latencies:
        for bs in args.batches:
            md, cid = bench_cell("dict", bs, latency_us, args.num_sims,
                                 args.moves, args.trials)
            mf, cif = bench_cell("flat", bs, latency_us, args.num_sims,
                                 args.moves, args.trials)
            ratio = mf / md if md > 0 else 0.0
            print(f"{latency_us:>8}us {bs:>6}   "
                  f"{md:>10,.0f} +-{cid:>3.0f} {mf:>10,.0f} +-{cif:>3.0f} "
                  f"{ratio:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
