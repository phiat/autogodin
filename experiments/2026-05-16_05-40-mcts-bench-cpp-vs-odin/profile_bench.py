#!/usr/bin/env python3
"""Profiling ablation: pin down where the 232 µs/sim gap lives.

Three eval modes, same MCTS code path, same 1600 sims x 32 moves:

    'full'     — current uniform-policy eval (re-builds dict every call)
    'cached'   — pre-built dict for the empty 9x9 board; returns SAME dict
                  every call. Strips per-call Python work; keeps the FFI hop.
    'precomp'  — alongside 'cached', but also skips board.get_legal_moves_flat().

Comparing full vs cached vs precomp tells us how much of the per-sim cost is
inside the Python eval callback vs inside the .so itself.

Usage:
    python profile_bench.py --backend odin   --mode full
    python profile_bench.py --backend odin   --mode cached
    python profile_bench.py --backend odin   --mode precomp
    python profile_bench.py --backend cpp    --mode full     # etc.

Writes one row per run to data_ablation.csv.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

SIZE = 9
KOMI = 7.5
NUM_SIMS = 1600
NUM_MOVES = 32


def build_evaluators(ag):
    n_actions = SIZE * SIZE + 1  # 82
    p = 1.0 / n_actions
    cached_dict = {i: p for i in range(n_actions - 1)}
    cached_dict[ag.PASS_ACTION] = p
    cached_value = 0.0

    def full(board):
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1
        pp = 1.0 / n
        out = {a: pp for a in legal}
        out[ag.PASS_ACTION] = pp
        return out, 0.0

    def cached(board):
        # Skip dict construction; skip the per-call uniform recomputation;
        # still call get_legal_moves_flat() to keep the C side honest about
        # legality (returning impossible actions could corrupt MCTS).
        board.get_legal_moves_flat()
        return cached_dict, cached_value

    def precomp(board):
        # Strip the legality check too. UCT will pick illegal nodes;
        # that's fine for *timing* only — DO NOT use the resulting tree.
        return cached_dict, cached_value

    return {"full": full, "cached": cached, "precomp": precomp}


def run_trial(ag, evaluator, num_sims, num_moves):
    board = ag.GoBoard(SIZE, KOMI)
    cfg = ag.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_weight = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.lambda_ = 0.0
    cfg.max_depth = 100
    cfg.temperature = 1.0

    total_sims = 0
    t0 = time.perf_counter()
    for _ in range(num_moves):
        if board.is_game_over():
            break
        tree = ag.MCTSTree(board, cfg)
        tree.run_simulations(num_sims, evaluator)
        total_sims += num_sims
        a = tree.select_action(0.0)
        if a == ag.PASS_ACTION:
            board.pass_move()
        else:
            ok = board.play_flat(a)
            if not ok:
                board.pass_move()
    return total_sims, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["odin", "cpp"], required=True)
    p.add_argument("--mode", choices=["full", "cached", "precomp"], required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default=os.path.join(HERE, "data_ablation.csv"))
    args = p.parse_args()

    ag = importlib.import_module(
        "alpha_go_odin" if args.backend == "odin" else "alpha_go_cpp"
    )
    evals = build_evaluators(ag)
    evaluator = evals[args.mode]

    print(f"=== backend={args.backend} mode={args.mode} sims={NUM_SIMS} moves={NUM_MOVES} ===",
          flush=True)
    for w in range(args.warmup):
        n, dt = run_trial(ag, evaluator, NUM_SIMS, NUM_MOVES)
        print(f"  warmup {w}: {n/dt:,.0f}/s", flush=True)

    rows = []
    for t in range(args.trials):
        n, dt = run_trial(ag, evaluator, NUM_SIMS, NUM_MOVES)
        sps = n / dt
        rows.append({"backend": args.backend, "mode": args.mode, "trial": t,
                     "total_sims": n, "elapsed_sec": round(dt, 6),
                     "sims_per_sec": round(sps, 2)})
        print(f"  trial {t}: {sps:,.0f}/s ({dt:.2f}s)", flush=True)

    fieldnames = ["backend", "mode", "trial", "total_sims", "elapsed_sec", "sims_per_sec"]
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
