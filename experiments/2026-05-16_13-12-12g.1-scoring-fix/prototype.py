#!/usr/bin/env python3
"""Tromp-Taylor scoring with eye-counting dead-stone removal.

See design.md for context. The runtime score() in odin/alpha_go and the
upstream alpha_go_cpp use pure TT, which counts dead stones as alive.
This prototype adds a cleanup pass: each group's "eye-like points" are
counted; groups with <2 eyes are removed before TT is applied.

The board representation here is a NumPy int8 grid:
  0 = empty
  1 = black
  2 = white

This matches alpha_go_odin / alpha_go_cpp's wire format.

Self-test:
    python prototype.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterator

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2

# -----------------------------------------------------------------------------
# Board primitives
# -----------------------------------------------------------------------------

def neighbors(idx: int, size: int) -> Iterator[int]:
    r, c = divmod(idx, size)
    if r > 0:        yield idx - size
    if r < size - 1: yield idx + size
    if c > 0:        yield idx - 1
    if c < size - 1: yield idx + 1


def group_of(board: np.ndarray, idx: int) -> set[int]:
    """All connected stones of board[idx]'s color, by BFS."""
    color = int(board[idx])
    if color == EMPTY:
        return set()
    size = int(np.sqrt(len(board)))
    stack = [idx]
    seen = {idx}
    while stack:
        cur = stack.pop()
        for n in neighbors(cur, size):
            if n not in seen and int(board[n]) == color:
                seen.add(n)
                stack.append(n)
    return seen


def is_eye_like(board: np.ndarray, idx: int, color: int) -> bool:
    """An empty point whose every on-board neighbor is `color`.

    This is the classic 'eye-like point' from Brügmann/MoGo. It's an
    approximation: it counts as an eye if surrounded by friendlies, with
    no special handling for diagonal weakness (false eyes). Good enough
    for the README's pathology; we'll quantify the miss rate in tests.
    """
    if int(board[idx]) != EMPTY:
        return False
    size = int(np.sqrt(len(board)))
    n_count = 0
    for n in neighbors(idx, size):
        n_count += 1
        if int(board[n]) != color:
            return False
    # All on-board neighbors are `color`. The "off-board sides are walls"
    # case (corner/edge points) is implicitly fine — corners with 2 friendly
    # neighbors are eye-like (the missing neighbors are off-board walls
    # which favor the friendly).
    return n_count >= 1


def eye_count(board: np.ndarray, group: set[int]) -> int:
    """Number of eye-like points adjacent to any stone in `group`."""
    if not group:
        return 0
    size = int(np.sqrt(len(board)))
    color = int(board[next(iter(group))])
    candidates: set[int] = set()
    for s in group:
        for n in neighbors(s, size):
            if int(board[n]) == EMPTY:
                candidates.add(n)
    return sum(1 for c in candidates if is_eye_like(board, c, color))


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def tt_score(board: np.ndarray, komi: float = 7.5) -> float:
    """Plain Tromp-Taylor area score. Returns black - white."""
    size = int(np.sqrt(len(board)))
    black, white = 0.0, komi
    for i in range(size * size):
        if board[i] == BLACK: black += 1
        elif board[i] == WHITE: white += 1

    visited = np.zeros(size * size, dtype=bool)
    for i in range(size * size):
        if board[i] != EMPTY or visited[i]:
            continue
        territory = []
        borders = set()
        stack = [i]
        visited[i] = True
        while stack:
            cur = stack.pop()
            territory.append(cur)
            for n in neighbors(cur, size):
                if board[n] == EMPTY:
                    if not visited[n]:
                        visited[n] = True
                        stack.append(n)
                else:
                    borders.add(int(board[n]))
        if len(borders) == 1:
            owner = next(iter(borders))
            if owner == BLACK: black += len(territory)
            else:              white += len(territory)
    return black - white


def find_groups(board: np.ndarray) -> list[set[int]]:
    """All groups of like-colored stones."""
    size = int(np.sqrt(len(board)))
    seen = np.zeros(size * size, dtype=bool)
    groups: list[set[int]] = []
    for i in range(size * size):
        if seen[i] or board[i] == EMPTY:
            continue
        g = group_of(board, i)
        for s in g: seen[s] = True
        groups.append(g)
    return groups


def cleanup_and_score(board: np.ndarray, komi: float = 7.5, min_eyes: int = 2) -> tuple[float, set[int]]:
    """TT score after eye-counting dead-stone removal.

    Returns (score, dead_stone_indices).
    """
    work = board.copy()
    dead: set[int] = set()
    for g in find_groups(work):
        if eye_count(work, g) < min_eyes:
            dead.update(g)
    for s in dead:
        work[s] = EMPTY
    return tt_score(work, komi), dead


# -----------------------------------------------------------------------------
# Self-test entry point
# -----------------------------------------------------------------------------

def run_self_test() -> int:
    from test_positions import all_positions

    total, correct = 0, 0
    failures: list[tuple[str, float, float, int, int]] = []
    tt_vs_cleanup_deltas: list[tuple[str, float, float, float]] = []
    for name, board, label_score, expected_dead in all_positions():
        score, dead = cleanup_and_score(board)
        ok = abs(score - label_score) < 0.01 and len(dead) == expected_dead
        total += 1
        if ok:
            correct += 1
        else:
            failures.append((name, score, label_score, len(dead), expected_dead))
        # Independent check: pure TT on the raw board, vs cleanup-aware score.
        # A nonzero delta on dead-group fixtures proves the cleanup matters.
        tt_only = tt_score(board)
        delta = score - tt_only
        tt_vs_cleanup_deltas.append((name, tt_only, score, delta))

    print(f"Self-test: {correct}/{total} positions correct ({100.0 * correct / total:.1f}%)")
    for name, s, ls, d, ed in failures[:10]:
        print(f"  FAIL {name}: score={s:+.1f} (expected {ls:+.1f}), dead={d} (expected {ed})")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    # Summary of how often cleanup actually changes the score vs raw TT.
    nontrivial = [d for d in tt_vs_cleanup_deltas if abs(d[3]) > 0.01]
    print(f"\nCleanup changed score on {len(nontrivial)}/{total} positions.")
    if nontrivial:
        deltas = [abs(d[3]) for d in nontrivial]
        print(f"  delta (|cleanup - TT|): min={min(deltas):.1f} mean={sum(deltas)/len(deltas):.1f} max={max(deltas):.1f}")
        # Show a sample
        for name, tt_only, s, delta in nontrivial[:3]:
            print(f"  e.g. {name}: TT={tt_only:+.1f}, cleanup={s:+.1f}, delta={delta:+.1f}")

    return 0 if correct == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="Run prototype against test_positions.py")
    args = ap.parse_args()
    if args.self_test:
        return run_self_test()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
