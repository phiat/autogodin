#!/usr/bin/env python3
"""MCTS strength A/B: Odin vs C++ at equal sim budgets.

Both backends expose the same alphago_* OO API surface. Each game:
  - one backend plays Black, the other White (alternated per game_idx)
  - shared uniform-policy evaluator on both sides
  - each move: side-to-move's backend builds an MCTSTree against its current
    board view, runs N sims, samples an action, applies it to BOTH backends'
    boards (keeping them in sync)
  - game ends on 2x consecutive pass (handled by the boards themselves)

What this catches: subtle MCTS semantic bugs (perspective sign, exploration
constant flip, virtual-loss accounting) that pass unit-test parity but
produce systematically weaker play. If Odin and C++ implement the same
algorithm, win-rate over many games should overlap 50%.

Reads no NN — uniform-policy evaluator means MCTS itself is the only thing
exerting strength. 200 sims/move is plenty for a 9x9 board to discriminate.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import alpha_go_cpp as ac
import alpha_go_odin as ao


SIZE = 9
KOMI = 7.5


def make_uniform_evaluator(pass_action: int) -> Callable:
    """policy(legal) + PASS, all uniform; value = 0.5 (neutral)."""
    def evaluator(board) -> tuple[dict[int, float], float]:
        legal = board.get_legal_moves_flat()
        n = len(legal) + 1
        p = 1.0 / n
        out = {a: p for a in legal}
        out[pass_action] = p
        return out, 0.5
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
) -> dict:
    """Run a single game. Returns {winner, num_moves, hash_path}."""
    odin_board = ao.GoBoard(SIZE, KOMI)
    cpp_board  = ac.GoBoard(SIZE, KOMI)

    cfg_odin = make_config(ao, c_puct, max_depth, temperature)
    cfg_cpp  = make_config(ac, c_puct, max_depth, temperature)
    ev_odin = make_uniform_evaluator(ao.PASS_ACTION)
    ev_cpp  = make_uniform_evaluator(ac.PASS_ACTION)

    rng = random.Random(seed)

    move = 0
    while move < move_cap:
        # End on consecutive-pass (either board agrees; they're mirrors).
        if odin_board.is_game_over():
            break

        # Whose turn? Black = move%2 == 0.
        is_black = (move % 2 == 0)
        backend = black_backend if is_black else white_backend
        if backend is ao:
            tree = ao.MCTSTree(odin_board, cfg_odin, seed=seed * 1000 + move)
            tree.run_simulations(num_sims, ev_odin)
        else:
            # alpha_go_cpp.MCTSTree has no per-tree seed (rng is global to the C++
            # side). Game-level variation still comes from the per-move tree state
            # + categorical sampling at the configured temperature.
            tree = ac.MCTSTree(cpp_board, cfg_cpp)
            tree.run_simulations(num_sims, ev_cpp)

        # First `sample_until_move` moves: sample from visit distribution at the
        # configured temperature. Beyond: argmax (greedy).
        t_pick = temperature if move < sample_until_move else 0.0
        action = tree.select_action(t_pick)

        # Apply to BOTH boards to keep state mirrored.
        if action == ao.PASS_ACTION:  # both PASS_ACTION values are -1
            odin_board.pass_move()
            cpp_board.pass_move()
        else:
            ok_o = odin_board.play_flat(action)
            ok_c = cpp_board.play_flat(action)
            if not (ok_o and ok_c):
                # Either side judged the action illegal — score this as a pass
                # so the game can end gracefully. Should be rare with a sound MCTS.
                odin_board.pass_move()
                cpp_board.pass_move()

        move += 1

    # Both boards should agree on winner; use Odin's view.
    winner = odin_board.get_winner()  # BLACK/WHITE/EMPTY (= 1/2/0)
    cpp_winner = cpp_board.get_winner()

    return {
        "winner": int(winner),
        "cpp_winner": int(cpp_winner),  # sanity: should equal winner
        "num_moves": move,
        "final_score_odin": float(odin_board.score()) if hasattr(odin_board, "score") else None,
    }


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — better than normal-approx for small n."""
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
    p.add_argument("--sample-until-move", type=int, default=15,
                   help="Sample from visit distribution for first N moves; argmax after.")
    p.add_argument("--move-cap", type=int, default=200, help="Hard ceiling on moves per game.")
    p.add_argument("--seed-base", type=int, default=0xC0FFEE)
    p.add_argument("--out-csv", default=os.path.join(HERE, "results.csv"))
    p.add_argument("--out-summary", default=os.path.join(HERE, "summary.json"))
    args = p.parse_args()

    print(f"=== Odin vs C++ MCTS strength A/B "
          f"(games={args.games} sims/move={args.num_sims} temp={args.temperature} "
          f"sample_until={args.sample_until_move}) ===", flush=True)

    rows = []
    odin_wins = cpp_wins = draws = 0
    t0 = time.perf_counter()

    for g in range(args.games):
        # Alternate: even game -> Odin Black, odd -> Odin White.
        odin_is_black = (g % 2 == 0)
        black, white = (ao, ac) if odin_is_black else (ac, ao)

        seed = args.seed_base + g
        gt0 = time.perf_counter()
        result = play_one_game(
            black_backend=black, white_backend=white,
            num_sims=args.num_sims, c_puct=args.c_puct, max_depth=args.max_depth,
            temperature=args.temperature, sample_until_move=args.sample_until_move,
            move_cap=args.move_cap, seed=seed,
        )
        dt = time.perf_counter() - gt0

        winner = result["winner"]  # 1=BLACK 2=WHITE 0=draw
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
        "config": {
            "num_sims": args.num_sims, "c_puct": args.c_puct,
            "max_depth": args.max_depth, "temperature": args.temperature,
            "sample_until_move": args.sample_until_move,
            "move_cap": args.move_cap, "seed_base": args.seed_base,
            "size": SIZE, "komi": KOMI,
        },
        "elapsed_sec": round(elapsed, 2),
    }

    fields = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: O={odin_wins} C={cpp_wins} D={draws}  "
          f"odin_winrate={odin_winrate:.3f}  "
          f"Wilson95% = [{lo:.3f}, {hi:.3f}]  "
          f"elapsed={elapsed:.1f}s",
          flush=True)
    print(f"50% in CI: {lo <= 0.5 <= hi}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
