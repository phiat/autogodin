#!/usr/bin/env python3
"""Reproduce autogodin-6qt: deterministic selfplay despite add_noise=True.

Plays 2 games with the saved iter0_best.pt at low sims (50) and prints the
first 5 moves of each. With the bug, both games should be identical.

After the fix, the games should differ.

Run on local CPU:
    PYTHONPATH=python/odin_backend:python:autogo/src \\
      autogo/.venv-cpponly/bin/python \\
      experiments/2026-05-17_13-47-bpoC-katago-seed/postmortem/repro.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the Odin shim — same path as the real selfplay run.
os.environ["ALPHAGO_BACKEND"] = "odin"

from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator
from alpha_go.gameplay import play_game

CKPT = Path(__file__).parent / "iter0_best.pt"
assert CKPT.exists(), f"missing {CKPT} — run bpoC-katago-seed iter0 first"


def make_agent():
    ev = LeafBatchedNNEvaluator(str(CKPT), 9, "256x10")
    return CppMCTSAgent(
        evaluator=ev,
        num_simulations=50,   # low for speed
        c_puct=1.0,
        temperature=1.0,
        add_noise=True,
        # leaf_batch_size=0 forces the non-batched MCTS path, which uses
        # evaluator.evaluate(single_board) -> (dict, float). The batched path
        # has a separate WIP contract issue unrelated to autogodin-6qt.
        leaf_batch_size=0,
    )


def first_n_moves(record, n):
    return [(int(m[0]), int(m[1])) for m in record.moves[:n]]


def main():
    print("=== game 1 (seed=42) ===")
    g1 = play_game(
        black_agent=make_agent(),
        white_agent=make_agent(),
        board_size=9,
        seed=42,
        max_moves=80,
        komi=7.5,
    )
    print(f"  winner={g1.winner} num_moves={g1.num_moves} result={g1.result}")
    print(f"  first 8 moves: {first_n_moves(g1, 8)}")

    print("=== game 2 (seed=43) ===")
    g2 = play_game(
        black_agent=make_agent(),
        white_agent=make_agent(),
        board_size=9,
        seed=43,
        max_moves=80,
        komi=7.5,
    )
    print(f"  winner={g2.winner} num_moves={g2.num_moves} result={g2.result}")
    print(f"  first 8 moves: {first_n_moves(g2, 8)}")

    same = (g1.winner == g2.winner
            and g1.num_moves == g2.num_moves
            and first_n_moves(g1, 8) == first_n_moves(g2, 8))
    print()
    print(f"DETERMINISTIC? {'YES (bug present)' if same else 'NO (bug fixed)'}")
    sys.exit(0 if not same else 1)


if __name__ == "__main__":
    main()
