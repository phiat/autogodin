#!/usr/bin/env python3
"""L4 repro for autogodin-6qt — does the bug live in the BATCHED path?

Runs 4 games each with leaf_batch_size in {0, 64}. With batched=64 and a
peaked-policy ckpt we expect the 391-identical-games failure mode.

Run on L4 after jl_bootstrap.sh and after iter0_best.pt is uploaded
to /tmp/iter0_best.pt:

  PYTHONPATH=python/odin_backend:python:autogo/src \\
    GAME_DATA_DIR=/home/nfs-local/game_data_root ALPHAGO_BACKEND=odin \\
    autogo/.venv/bin/python repro_l4.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

os.environ["ALPHAGO_BACKEND"] = "odin"

from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator
from alpha_go.gameplay import play_game

CKPT = Path("/tmp/iter0_best.pt")
assert CKPT.exists()


def make_agent(leaf_batch_size: int):
    ev = LeafBatchedNNEvaluator(str(CKPT), 9, "256x10")
    return CppMCTSAgent(
        evaluator=ev,
        num_simulations=200,   # match the L4 selfplay config
        c_puct=1.0,
        temperature=1.0,
        add_noise=True,
        leaf_batch_size=leaf_batch_size,
    )


def first_n_moves(record, n):
    return [(int(m[0]), int(m[1])) for m in record.moves[:n]]


def run_set(leaf_batch_size: int, n_games: int = 4):
    print(f"\n=== leaf_batch_size={leaf_batch_size} ({n_games} games) ===")
    sigs = []
    for i in range(n_games):
        g = play_game(
            black_agent=make_agent(leaf_batch_size),
            white_agent=make_agent(leaf_batch_size),
            board_size=9,
            seed=42 + i,
            max_moves=80,
            komi=7.5,
        )
        first6 = tuple(first_n_moves(g, 6))
        sigs.append((g.winner, g.num_moves, first6))
        print(f"  game {i} (seed={42+i}): win={g.winner} n={g.num_moves} first6={first6}")
    distinct = len(set(sigs))
    print(f"  distinct game signatures: {distinct}/{n_games}")
    if distinct == 1:
        print(f"  >> DETERMINISTIC — 6qt bug present")
    else:
        print(f"  >> DIVERSE — 6qt bug NOT present in this path")
    return distinct


def main():
    print("Repro autogodin-6qt: does batched MCTS produce identical games?")
    print(f"  ckpt: {CKPT}")

    d0 = run_set(leaf_batch_size=0)
    d64 = run_set(leaf_batch_size=64)

    print()
    print("=" * 60)
    print(f"non-batched (leaf_batch_size=0): {d0} distinct signatures")
    print(f"    batched (leaf_batch_size=64): {d64} distinct signatures")
    print()
    if d0 > 1 and d64 == 1:
        print("VERDICT: 6qt bug LIVES in the batched-inference path.")
        sys.exit(0)
    elif d0 == 1 and d64 == 1:
        print("VERDICT: bug is in BOTH paths (RNG-default mismatch with peaked policy?)")
        sys.exit(0)
    elif d0 > 1 and d64 > 1:
        print("VERDICT: bug does NOT repro on L4 with this ckpt+settings — needs deeper investigation.")
        sys.exit(1)
    else:
        print(f"VERDICT: unexpected combination (d0={d0}, d64={d64}).")
        sys.exit(2)


if __name__ == "__main__":
    main()
