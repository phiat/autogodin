#!/usr/bin/env python3
"""ydh.3 follow-up: same batched sweep as bench.odin but driven from Python.

Compares the in-process Odin numbers (bench.odin / results.md) against the
Python ctypes batched path. The delta at each batch_size × latency cell is
the FFI cost — relevant for sizing real NN evaluator integrations.

Run with the same .venv-cpponly that runs the other autogodin benches.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import alpha_go_odin as ao


SIZE = 9
KOMI = 7.5


def make_batched_uniform_evaluator(pass_action: int, latency_us: float):
    """Returns a batched evaluator with optional per-call synthetic latency."""
    latency_s = latency_us / 1e6 if latency_us > 0 else 0.0

    def evaluator(boards):
        if latency_s > 0:
            time.sleep(latency_s)
        policies = []
        values = []
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


def run_trial(batch_size: int, latency_us: float, num_sims: int, moves: int, seed: int):
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = 100

    board = ao.GoBoard(SIZE, KOMI)
    ev = make_batched_uniform_evaluator(ao.PASS_ACTION, latency_us)

    total_sims = 0
    t0 = time.perf_counter()
    for move in range(moves):
        if board.is_game_over():
            break
        tree = ao.MCTSTree(board, cfg, seed=seed * 1000 + move)
        tree.run_simulations_batched(num_sims, batch_size, ev)
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


def bench_cell(batch_size: int, latency_us: float, num_sims: int, moves: int, trials: int):
    # Warmup
    _ = run_trial(batch_size, latency_us, num_sims, moves, seed=42)
    rates = []
    for i in range(trials):
        sims, dt = run_trial(batch_size, latency_us, num_sims, moves, seed=100 + i)
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
                   help="Per-leaf evaluator latencies in microseconds")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    args = p.parse_args()

    print("autogodin e11 — Python ctypes batched MCTS throughput sweep")
    print(f"9x9 Go, uniform policy, Python ctypes path through alpha_go_odin")
    print(f"config: {args.num_sims} sims/move x {args.moves} moves/trial = "
          f"{args.num_sims * args.moves:,} sims/trial; {args.trials} trials/cell")
    print()
    print("batch_size x per-leaf evaluator latency:")
    print()

    for lat in args.latencies:
        for bs in args.batches:
            m, ci = bench_cell(bs, lat, args.num_sims, args.moves, args.trials)
            print(f"batch={bs}\tlatency={lat}us\t| {m:.0f} +- {ci:.0f} sims/s")
        print()


if __name__ == "__main__":
    main()
