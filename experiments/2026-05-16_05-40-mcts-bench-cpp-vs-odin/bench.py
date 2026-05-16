#!/usr/bin/env python3
"""Single-thread MCTS throughput micro-bench: C++ vs Odin (NN-free).

Holds everything fixed except the backend module:

    9x9 empty start, komi 7.5
    1600 simulations / move, 32 moves / trial
    deterministic uniform-policy evaluator (no NN, no GPU)
    MCTSConfig: c_puct=1.0, no dirichlet, no PCR, lambda=0, max_depth=100

The evaluator returns the uniform legal-action policy + value=0.0 for every
leaf, so MCTS visits are policy-blind UCT, and total work per backend is
deterministic given the action sequence. Both backends consume the same
callback signature, so the only thing that varies is the C side of the FFI
boundary: alpha_go_cpp (pybind11) vs alpha_go_odin (ctypes).

Usage:
    python bench.py --backend odin --trials 5 --out data.csv
    python bench.py --backend cpp  --trials 5 --out data.csv  # appends
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

SIZE = 9
KOMI = 7.5
NUM_SIMS = 1600
NUM_MOVES = 32
C_PUCT = 1.0
MAX_DEPTH = 100


def make_uniform_evaluator(pass_action: int):
    def evaluator(board) -> tuple[dict[int, float], float]:
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1  # +1 for pass
        p = 1.0 / n
        out = {a: p for a in legal}
        out[pass_action] = p
        return out, 0.0
    return evaluator


def run_trial(ag, num_sims: int, num_moves: int) -> tuple[int, float]:
    """One trial: play `num_moves` moves of self-play with `num_sims` sims each.

    Returns (total_simulations, elapsed_seconds).
    """
    board = ag.GoBoard(SIZE, KOMI)
    cfg = ag.MCTSConfig()
    cfg.c_puct = C_PUCT
    cfg.dirichlet_weight = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.temperature = 1.0
    cfg.lambda_ = 0.0
    cfg.max_depth = MAX_DEPTH

    evaluator = make_uniform_evaluator(ag.PASS_ACTION)

    total_sims = 0
    t0 = time.perf_counter()
    for move_idx in range(num_moves):
        if board.is_game_over():
            break
        tree = ag.MCTSTree(board, cfg)
        tree.run_simulations(num_sims, evaluator)
        total_sims += num_sims
        action = tree.select_action(0.0)  # greedy (argmax visit count)
        if action == ag.PASS_ACTION:
            board.pass_move()
        else:
            ok = board.play_flat(action)
            if not ok:
                board.pass_move()
    elapsed = time.perf_counter() - t0
    return total_sims, elapsed


def mean_and_ci95(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, 0.0)
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    # Half-width of 95% CI: t_{0.975, n-1} ~ 2 for small n; use 1.96 for normal-approx.
    return m, 1.96 * sd / math.sqrt(len(xs))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["odin", "cpp"], required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed trials before measurement (JIT/cache warmup).")
    p.add_argument("--num-sims", type=int, default=NUM_SIMS)
    p.add_argument("--num-moves", type=int, default=NUM_MOVES)
    p.add_argument("--out", required=True, help="CSV to append rows to.")
    args = p.parse_args()

    ag = importlib.import_module(
        "alpha_go_odin" if args.backend == "odin" else "alpha_go_cpp"
    )

    print(f"=== backend={args.backend} sims={args.num_sims} moves={args.num_moves} "
          f"warmup={args.warmup} trials={args.trials} ===", flush=True)

    for w in range(args.warmup):
        n, dt = run_trial(ag, args.num_sims, args.num_moves)
        print(f"  warmup {w}: {n} sims in {dt:.3f}s ({n/dt:,.0f}/s)", flush=True)

    sims_per_sec: list[float] = []
    rows: list[dict] = []
    for t in range(args.trials):
        n, dt = run_trial(ag, args.num_sims, args.num_moves)
        sps = n / dt
        sims_per_sec.append(sps)
        rows.append({
            "backend": args.backend, "trial": t,
            "total_sims": n, "elapsed_sec": round(dt, 6),
            "sims_per_sec": round(sps, 2),
            "num_sims": args.num_sims, "num_moves": args.num_moves,
            "size": SIZE,
        })
        print(f"  trial {t}: {n} sims in {dt:.3f}s ({sps:,.0f}/s)", flush=True)

    m, ci = mean_and_ci95(sims_per_sec)
    print(f"  {args.backend}: {m:,.0f} ± {ci:,.0f} sims/sec (95% CI, n={len(sims_per_sec)})",
          flush=True)

    fieldnames = ["backend", "trial", "total_sims", "elapsed_sec",
                  "sims_per_sec", "num_sims", "num_moves", "size"]
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
