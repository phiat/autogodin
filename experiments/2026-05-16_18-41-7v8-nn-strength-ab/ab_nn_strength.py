#!/usr/bin/env python3
"""7v8: MCTS strength A/B between Odin and C++ under a real NN evaluator.

slg used a deterministic random projection as informative-prior surrogate.
That cleared the FPU-degenerate uniform regime, but the bead asks for the
final correctness gate: a real (random-init) GoResNet evaluator, the same
one we'd plug into Phase 3.

Same harness shape as slg: 100 games at 200 sims/move, alternating colors,
identical seeds, same evaluator closure for both backends (so any
divergence is an MCTS-algorithm divergence, not eval drift).

Pass: Wilson 95% CI brackets 0.5 → Odin port is parity-complete under
realistic priors. Phase 1+2 sealed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Callable

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import alpha_go_cpp as ac
import alpha_go_odin as ao
from alpha_go.model import SizeInvariantGoResNet

SIZE = 9
KOMI = 7.5
N_CELLS = SIZE * SIZE
N_ACTIONS = N_CELLS + 1
PASS_INDEX = N_CELLS


def make_net(seed: int):
    """Random-init SizeInvariantGoResNet (32ch x 4b, 76k params) — same
    architecture ydh.5 and 441 used. Deterministic given seed."""
    torch.manual_seed(seed)
    net = SizeInvariantGoResNet(channels=32, n_blocks=4, value_hidden=32)
    net.eval()
    return net


def make_nn_evaluator(net, pass_action: int) -> Callable:
    """Backend-agnostic single-leaf NN evaluator. Identical output for
    identical board state → both backends see the same eval."""
    def evaluator(board) -> tuple[dict[int, float], float]:
        raw = board.to_numpy()
        cur = board.to_play()
        opp = ao.BLACK if cur == ao.WHITE else ao.WHITE
        b = np.zeros_like(raw, dtype=np.int64)
        b[raw == cur] = 1
        b[raw == opp] = 2
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(b).unsqueeze(0),
                         torch.ones(1, SIZE, SIZE))
        probs = torch.softmax(pl[0], dim=-1).numpy()
        legal = board.get_legal_moves_flat()
        out = {a: float(probs[a]) for a in legal}
        out[pass_action] = float(probs[PASS_INDEX])
        return out, float(torch.tanh(vl[0]).item())
    return evaluator


def make_config(backend, c_puct: float, max_depth: int, temperature: float):
    cfg = backend.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = temperature
    cfg.max_depth = max_depth
    return cfg


def play_one_game(*, net, black_backend, white_backend,
                  num_sims: int, c_puct: float, max_depth: int,
                  temperature: float, sample_until_move: int,
                  move_cap: int, seed: int) -> dict:
    odin_board = ao.GoBoard(SIZE, KOMI)
    cpp_board  = ac.GoBoard(SIZE, KOMI)

    cfg_odin = make_config(ao, c_puct, max_depth, temperature)
    cfg_cpp  = make_config(ac, c_puct, max_depth, temperature)
    ev_odin = make_nn_evaluator(net, ao.PASS_ACTION)
    ev_cpp  = make_nn_evaluator(net, ac.PASS_ACTION)

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
    p.add_argument("--net-seed", type=int, default=0,
                   help="torch.manual_seed for the random-init GoResNet")
    p.add_argument("--out-csv", default=os.path.join(HERE, "results.csv"))
    p.add_argument("--out-summary", default=os.path.join(HERE, "summary.json"))
    args = p.parse_args()

    print(f"=== 7v8: Odin vs C++ MCTS strength A/B, NN evaluator "
          f"(games={args.games} sims/move={args.num_sims} temp={args.temperature} "
          f"sample_until={args.sample_until_move} net_seed={args.net_seed}) ===",
          flush=True)

    net = make_net(args.net_seed)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"net: SizeInvariantGoResNet(32ch x 4b), {n_params:,} params", flush=True)

    rows = []
    odin_wins = cpp_wins = draws = 0
    t0 = time.perf_counter()

    for g in range(args.games):
        odin_is_black = (g % 2 == 0)
        black, white = (ao, ac) if odin_is_black else (ac, ao)

        seed = args.seed_base + g
        gt0 = time.perf_counter()
        result = play_one_game(
            net=net,
            black_backend=black, white_backend=white,
            num_sims=args.num_sims, c_puct=args.c_puct,
            max_depth=args.max_depth, temperature=args.temperature,
            sample_until_move=args.sample_until_move,
            move_cap=args.move_cap, seed=seed,
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
            "net_seed": args.net_seed,
            "size": SIZE, "komi": KOMI,
            "evaluator": "SizeInvariantGoResNet(32ch x 4b), random-init",
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
