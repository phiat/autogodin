#!/usr/bin/env python3
"""slg: MCTS strength A/B between Odin and C++ under *informative* priors.

The uniform-prior A/B (experiments/2026-05-16_11-58-mcts-ab-postvendor/) put
both backends in the FPU concentration-vs-spread regime documented in
odin/vendor/mcts-odin/mcts/mcts.odin Config.fpu_reduction — under uniform priors
at low sim budgets, Odin's correct FPU-driven spread can lose to C++'s
accidental concentration. That's not a real strength bug, but it leaves the
strength gate uncovered.

This harness fixes that by replacing uniform with a *deterministic synthetic*
informative-prior evaluator: a seeded random projection of (board, legality,
to_play) → (action_logits, value). Same callable both backends, so any
divergence in win-rate is real MCTS-algorithm divergence (not eval drift).

Why not a real torch NN: a random-init GoResNet on CPU would be ~hours of
wall time and need torch installed; the point of this gate is to exit the
FPU-degenerate uniform regime, not to play strong Go. Random projection
gives non-uniform, board-dependent priors deterministically.

Pass: Wilson 95% CI brackets 0.5 → no real algorithm divergence; uniform-prior
regression was the documented FPU caveat, not a regression.
Fail: CI tight away from 0.5 → real bug; chase it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import alpha_go_cpp as ac
import alpha_go_odin as ao


SIZE = 9
KOMI = 7.5
N_CELLS = SIZE * SIZE
N_ACTIONS = N_CELLS + 1  # last index = pass
PASS_INDEX = N_CELLS


def make_informative_evaluator(pass_action: int, seed: int = 0xBEEF) -> Callable:
    """Deterministic, board-dependent prior generator.

    Builds (W_policy, W_value) once with a fixed seed, then for each board:
        feats = concat(board_one_hot_black, board_one_hot_white, legal_mask, [to_play])  # 244-d
        policy_logits = W_policy @ feats                                                 # 82
        value = sigmoid(W_value @ feats)
    Policy is masked to legal actions (plus PASS) and softmax-normalized.

    Identical output for identical inputs → both backends see the same eval
    even though the matmul is done in Python.
    """
    rng = np.random.default_rng(seed)
    feat_dim = 3 * N_CELLS + 1  # black-stones (81) + white-stones (81) + legal_mask (81) + to_play (1)
    W_policy = rng.normal(0.0, 0.5, size=(N_ACTIONS, feat_dim)).astype(np.float32)
    W_value = rng.normal(0.0, 0.3, size=(feat_dim,)).astype(np.float32)

    def evaluator(board) -> tuple[dict[int, float], float]:
        b = board.to_numpy().astype(np.float32).ravel()   # 81 cells, values in {0,1,2}
        black_oh = (b == 1.0).astype(np.float32)
        white_oh = (b == 2.0).astype(np.float32)
        legal_list = board.get_legal_moves_flat()
        legal_mask = np.zeros(N_CELLS, dtype=np.float32)
        for i in legal_list:
            legal_mask[i] = 1.0
        to_play = float(board.to_play())
        feats = np.concatenate([black_oh, white_oh, legal_mask, np.array([to_play], dtype=np.float32)])

        logits = W_policy @ feats  # (82,)
        # Mask: illegal moves get -inf; pass always legal.
        mask = np.full(N_ACTIONS, -1e9, dtype=np.float32)
        for i in legal_list:
            mask[i] = 0.0
        mask[PASS_INDEX] = 0.0
        masked = logits + mask
        masked -= masked.max()
        exp = np.exp(masked)
        probs = exp / exp.sum()

        policy: dict[int, float] = {i: float(probs[i]) for i in legal_list}
        policy[pass_action] = float(probs[PASS_INDEX])

        v_logit = float(W_value @ feats)
        value = 1.0 / (1.0 + math.exp(-v_logit))
        return policy, value

    return evaluator


def make_config(backend, c_puct: float, max_depth: int, temperature: float) -> object:
    cfg = backend.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = temperature
    cfg.max_depth = max_depth
    return cfg


def play_one_game(
    *,
    black_backend, white_backend,
    num_sims: int,
    c_puct: float,
    max_depth: int,
    temperature: float,
    sample_until_move: int,
    move_cap: int,
    seed: int,
    eval_seed: int,
) -> dict:
    odin_board = ao.GoBoard(SIZE, KOMI)
    cpp_board  = ac.GoBoard(SIZE, KOMI)

    cfg_odin = make_config(ao, c_puct, max_depth, temperature)
    cfg_cpp  = make_config(ac, c_puct, max_depth, temperature)
    ev_odin = make_informative_evaluator(ao.PASS_ACTION, seed=eval_seed)
    ev_cpp  = make_informative_evaluator(ac.PASS_ACTION, seed=eval_seed)

    move = 0
    while move < move_cap:
        if odin_board.is_game_over():
            break

        is_black = (move % 2 == 0)
        backend = black_backend if is_black else white_backend
        if backend is ao:
            tree = ao.MCTSTree(odin_board, cfg_odin, seed=seed * 1000 + move)
            tree.run_simulations(num_sims, ev_odin)
        else:
            tree = ac.MCTSTree(cpp_board, cfg_cpp)
            tree.run_simulations(num_sims, ev_cpp)

        t_pick = temperature if move < sample_until_move else 0.0
        action = tree.select_action(t_pick)

        if action == ao.PASS_ACTION:
            odin_board.pass_move()
            cpp_board.pass_move()
        else:
            ok_o = odin_board.play_flat(action)
            ok_c = cpp_board.play_flat(action)
            if not (ok_o and ok_c):
                odin_board.pass_move()
                cpp_board.pass_move()
        move += 1

    winner = odin_board.get_winner()
    cpp_winner = cpp_board.get_winner()
    return {
        "winner": int(winner),
        "cpp_winner": int(cpp_winner),
        "num_moves": move,
        "final_score_odin": float(odin_board.score()) if hasattr(odin_board, "score") else None,
    }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - margin, center + margin)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--num-sims", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-until-move", type=int, default=15)
    p.add_argument("--move-cap", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=0xC0FFEE)
    p.add_argument("--eval-seed", type=int, default=0xBEEF,
                   help="Seed for the W_policy / W_value random projection; fixed for the whole run.")
    p.add_argument("--out-csv", default=os.path.join(HERE, "results.csv"))
    p.add_argument("--out-summary", default=os.path.join(HERE, "summary.json"))
    args = p.parse_args()

    print(f"=== slg: Odin vs C++ MCTS strength A/B under informative priors "
          f"(games={args.games} sims/move={args.num_sims} temp={args.temperature} "
          f"sample_until={args.sample_until_move} eval_seed={args.eval_seed:#x}) ===", flush=True)

    rows = []
    odin_wins = cpp_wins = draws = 0
    t0 = time.perf_counter()

    for g in range(args.games):
        odin_is_black = (g % 2 == 0)
        black, white = (ao, ac) if odin_is_black else (ac, ao)

        seed = args.seed_base + g
        gt0 = time.perf_counter()
        result = play_one_game(
            black_backend=black, white_backend=white,
            num_sims=args.num_sims, c_puct=args.c_puct, max_depth=args.max_depth,
            temperature=args.temperature, sample_until_move=args.sample_until_move,
            move_cap=args.move_cap, seed=seed, eval_seed=args.eval_seed,
        )
        dt = time.perf_counter() - gt0

        winner = result["winner"]
        if winner == 0:
            outcome = "draw"; draws += 1
        elif (winner == ao.BLACK) == odin_is_black:
            outcome = "odin"; odin_wins += 1
        else:
            outcome = "cpp"; cpp_wins += 1

        rows.append({
            "game": g, "seed": seed, "odin_is_black": int(odin_is_black),
            "winner_color": winner, "outcome": outcome,
            "num_moves": result["num_moves"], "elapsed_sec": round(dt, 3),
            "cpp_winner_matches": int(result["cpp_winner"] == winner),
        })
        decided = odin_wins + cpp_wins
        rate_str = f"{odin_wins/decided:.3f}" if decided > 0 else "n/a"
        print(f"  game {g:3d}: {outcome:5s}  moves={result['num_moves']:3d}  "
              f"{dt:.1f}s  running odin_winrate={rate_str} "
              f"(O{odin_wins} C{cpp_wins} D{draws})", flush=True)

    elapsed = time.perf_counter() - t0
    decided = odin_wins + cpp_wins
    odin_winrate = (odin_wins / decided) if decided else 0.0
    lo, hi = wilson_ci(odin_wins, decided)

    summary = {
        "games": args.games, "decided": decided,
        "odin_wins": odin_wins, "cpp_wins": cpp_wins, "draws": draws,
        "odin_winrate": round(odin_winrate, 4),
        "wilson95_ci": [round(lo, 4), round(hi, 4)],
        "bracket_50": bool(lo <= 0.5 <= hi),
        "config": {
            "num_sims": args.num_sims, "c_puct": args.c_puct,
            "max_depth": args.max_depth, "temperature": args.temperature,
            "sample_until_move": args.sample_until_move,
            "move_cap": args.move_cap, "seed_base": args.seed_base,
            "eval_seed": args.eval_seed,
            "size": SIZE, "komi": KOMI,
            "evaluator": "informative-random-projection",
        },
        "elapsed_sec": round(elapsed, 2),
    }

    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: O={odin_wins} C={cpp_wins} D={draws}  "
          f"odin_winrate={odin_winrate:.3f}  "
          f"Wilson95% = [{lo:.3f}, {hi:.3f}]  "
          f"elapsed={elapsed:.1f}s", flush=True)
    print(f"50% in CI: {lo <= 0.5 <= hi}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
