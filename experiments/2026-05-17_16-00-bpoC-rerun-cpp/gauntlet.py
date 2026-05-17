#!/usr/bin/env python3
"""Gauntlet: bpoC-rerun-postfix iter4 (Odin-trained) vs bpoC-rerun-cpp iter4
(C++-trained), 100 games, Wilson 95% CI.

Both ckpts are the same architecture (SizeInvariantGoResNet 256ch × 10b)
trained with the byte-identical bpoC PATH C recipe; the ONLY difference is
the MCTS backend used during the selfplay-data-generation step in training.

The two ckpts produce DIFFERENT value_acc curves:
  Odin-trained iter4: train_value_acc 99.34%
  C++-trained  iter4: train_value_acc 70.31%

The Odin one looks "stronger" by value_acc but that's a training-data
artifact: bpoC-rerun-postfix ran BEFORE the autogodin-6qt seed-default fix
(f21a3cd), so its selfplay games were over-deterministic and the value
head fit them too easily. The C++ run had genuinely random Dirichlet
noise (std::random_device) so its data was harder to fit.

This gauntlet asks the only question that matters: WHICH MODEL ACTUALLY
PLAYS BETTER? Both via the same eval-time MCTS (upstream alpha_go_cpp
pybind11, no shim) so any strength delta is purely model-quality, not
backend-throughput.

Recipe (mirrors 7v8 nn-strength-ab eval methodology):
  - 100 games, alternating colors (50 with model-A=Black, 50 with =White)
  - 200 sims/move (matches training)
  - dirichlet_alpha = 0  (no exploration noise at eval)
  - temperature = 1.0 for the first 10 moves, then 0.0 (argmax)
  - max 200 moves/game
  - Game outcomes scored via GoBoard.get_winner() at end of play

Run after both ckpts exist locally:

  PYTHONPATH="$PWD/python:autogo/src" \\
    autogo/.venv-cpponly/bin/python \\
    experiments/2026-05-17_16-00-bpoC-rerun-cpp/gauntlet.py

DO NOT include python/odin_backend on PYTHONPATH — eval must use real C++.
The script asserts this at startup.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

import alpha_go_cpp
import alpha_go_odin
# Catch shim leak BEFORE we burn any compute.
assert alpha_go_cpp.MCTSTree is not alpha_go_odin.MCTSTree, (
    "SHIM LEAK — alpha_go_cpp is aliased to Odin; remove python/odin_backend "
    "from PYTHONPATH for the gauntlet."
)

from alpha_go.agents.nn_mcts import LeafBatchedNNEvaluator

ODIN_CKPT = REPO_ROOT / "experiments" / "2026-05-17_07-40-bpoC-rerun-postfix" / "checkpoints" / "iter4_best.pt"
CPP_CKPT  = HERE / "postmortem" / "iter4_best.pt"
SIZE = 9
KOMI = 7.5


def make_evaluator(ckpt: Path) -> LeafBatchedNNEvaluator:
    assert ckpt.exists(), f"missing {ckpt}"
    return LeafBatchedNNEvaluator(str(ckpt), SIZE, "256x10")


def make_config(c_puct: float, temperature: float, max_depth: int):
    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = temperature
    cfg.max_depth = max_depth
    return cfg


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def play_one_game(*, ev_a, ev_b, a_plays_black: bool,
                  num_sims: int, c_puct: float, max_depth: int,
                  temperature: float, sample_until_move: int,
                  move_cap: int) -> tuple[int, int]:
    """Returns (winner_int 1=B|2=W|0=draw, num_moves)."""
    board = alpha_go_cpp.GoBoard(SIZE, KOMI)
    cfg = make_config(c_puct, temperature, max_depth)

    move = 0
    while move < move_cap:
        if board.is_game_over():
            break
        is_black = (move % 2 == 0)
        if (is_black and a_plays_black) or (not is_black and not a_plays_black):
            ev = ev_a
        else:
            ev = ev_b
        tree = alpha_go_cpp.MCTSTree(board, cfg)
        tree.run_simulations_batched(num_sims, 64, ev.batch_evaluate)
        t_pick = temperature if move < sample_until_move else 0.0
        action = tree.select_action(t_pick)
        if action == alpha_go_cpp.PASS_ACTION:
            board.pass_move()
        else:
            ok = board.play_flat(action)
            if not ok:
                board.pass_move()
        move += 1

    winner = board.get_winner()
    return int(winner), move


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--num-sims", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-until-move", type=int, default=10)
    p.add_argument("--move-cap", type=int, default=200)
    p.add_argument("--save-csv", type=Path, default=HERE / "gauntlet_results.csv")
    p.add_argument("--save-json", type=Path, default=HERE / "gauntlet_results.json")
    args = p.parse_args()

    print(f"== bpoC iter4 gauntlet: Odin-trained vs C++-trained ==")
    print(f"  ODIN ckpt: {ODIN_CKPT}")
    print(f"  CPP  ckpt: {CPP_CKPT}")
    print(f"  games={args.games} sims/move={args.num_sims} "
          f"temp={args.temperature} sample_until_move={args.sample_until_move}")
    print(f"  c_puct={args.c_puct} max_depth={args.max_depth} move_cap={args.move_cap}")
    print(f"  eval-backend: alpha_go_cpp = {alpha_go_cpp.__file__}")

    ev_odin = make_evaluator(ODIN_CKPT)
    ev_cpp  = make_evaluator(CPP_CKPT)

    rows = []
    odin_wins = cpp_wins = draws = 0
    t0 = time.monotonic()

    for i in range(args.games):
        # ODIN-trained model is "A". Alternate colors per pair: even=A_Black, odd=A_White.
        a_plays_black = (i % 2 == 0)
        winner, n_moves = play_one_game(
            ev_a=ev_odin, ev_b=ev_cpp, a_plays_black=a_plays_black,
            num_sims=args.num_sims, c_puct=args.c_puct, max_depth=args.max_depth,
            temperature=args.temperature, sample_until_move=args.sample_until_move,
            move_cap=args.move_cap,
        )
        winner_is_black = (winner == 1)
        winner_is_white = (winner == 2)
        if winner == 0:
            draws += 1
            who = "DRAW"
        elif (a_plays_black and winner_is_black) or (not a_plays_black and winner_is_white):
            odin_wins += 1
            who = "ODIN"
        else:
            cpp_wins += 1
            who = "CPP"

        rows.append({
            "game": i, "a_color": "B" if a_plays_black else "W",
            "winner_int": winner, "winner": who, "moves": n_moves,
        })

        dt = time.monotonic() - t0
        gpm = (i + 1) / dt * 60 if dt > 0 else 0.0
        p, lo, hi = wilson(odin_wins, odin_wins + cpp_wins)
        print(f"  Game {i+1:3d}/{args.games}: A={'B' if a_plays_black else 'W'} "
              f"winner={who} moves={n_moves}  "
              f"ODIN {odin_wins} - {cpp_wins} CPP  "
              f"(draws={draws}, odin_p={p:.3f} CI[{lo:.3f},{hi:.3f}], "
              f"{gpm:.1f} gpm)", flush=True)

    # Final result
    decided = odin_wins + cpp_wins
    p, lo, hi = wilson(odin_wins, decided)
    print()
    print("=" * 60)
    print(f"FINAL: ODIN-trained {odin_wins} - CPP-trained {cpp_wins} (draws {draws})")
    print(f"  ODIN win rate: {p*100:.1f}%  Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
    if lo <= 0.5 <= hi:
        print("  CI brackets 0.5 → strength-equivalent at 95% confidence.")
    elif p > 0.5:
        print("  ODIN-trained ckpt is significantly STRONGER (CI above 0.5).")
    else:
        print("  CPP-trained ckpt is significantly STRONGER (CI below 0.5).")

    args.save_json.write_text(json.dumps({
        "config": vars(args) | {"games": args.games},
        "odin_wins": odin_wins, "cpp_wins": cpp_wins, "draws": draws,
        "odin_winrate": p, "wilson_lo": lo, "wilson_hi": hi,
        "rows": rows,
    }, indent=2, default=str))
    import csv
    with args.save_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "a_color", "winner_int", "winner", "moves"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.save_json} and {args.save_csv}")


if __name__ == "__main__":
    main()
