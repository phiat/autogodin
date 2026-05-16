#!/usr/bin/env python3
"""Behavior-parity fingerprint that works against BOTH backends.

Unlike random_games.py (which reads our Odin-only `current_hash` accessor),
this harness fingerprints only the externally observable state:

    - action sequence
    - final board contents (sha256 of to_numpy bytes)
    - per-move board content sha (for first-divergence pinpointing on diff)
    - final score, winner, move_count

If the upstream `alpha_go_cpp` (pybind11) and our `alpha_go_odin` (ctypes
shim) implement the same Tromp-Taylor + KataGo-superko semantics, the
fingerprints MUST match. This is the cross-language contract.

USAGE:
    python random_games_dual.py --backend odin   # writes fingerprint
    python random_games_dual.py --backend cpp    # same
    python random_games_dual.py --backend both   # runs both and diffs in-process
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

DEFAULT_GAMES = 10
DEFAULT_MAX_MOVES = 200
DEFAULT_SIZE = 9
DEFAULT_KOMI = 7.5
PASS_PROB = 0.02


def load_backend(name: str):
    if name == "odin":
        return importlib.import_module("alpha_go_odin")
    if name == "cpp":
        return importlib.import_module("alpha_go_cpp")
    raise ValueError(f"unknown backend: {name}")


def board_sha(board) -> str:
    # to_numpy() returns (size, size) int8. Bytes are exactly size*size.
    return hashlib.sha1(board.to_numpy().tobytes()).hexdigest()


def play_game(ag, size: int, komi: float, seed: int, max_moves: int) -> dict:
    rng = random.Random(seed)
    b = ag.GoBoard(size, komi)
    trace = {"seed": seed, "size": size, "komi": komi, "moves": []}

    for i in range(max_moves):
        if b.is_game_over():
            break
        legal = sorted(b.get_legal_moves_flat())
        if rng.random() < PASS_PROB or not legal:
            action = ag.PASS_ACTION
            b.pass_move()
        else:
            action = legal[rng.randrange(len(legal))]
            ok = b.play_flat(action)
            assert ok, f"action {action} rejected after appearing in legal list"
        trace["moves"].append({
            "i": i, "action": action,
            "to_play_after": int(b.to_play()),
            "board_sha": board_sha(b),
        })

    trace["final_board_sha"] = board_sha(b)
    trace["final_score"] = round(float(b.score()), 4)
    trace["winner"] = int(b.get_winner())
    trace["move_count"] = b.move_count()
    return trace


def fingerprint(traces: list[dict]) -> str:
    blob = json.dumps(traces, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def run_one(backend: str, args) -> tuple[str, list[dict]]:
    ag = load_backend(backend)
    traces = [play_game(ag, args.size, args.komi, args.seed + i, args.max_moves)
              for i in range(args.games)]
    return fingerprint(traces), traces


def diff_first(a_traces: list[dict], b_traces: list[dict]) -> str | None:
    """Returns a human-readable first-divergence description, or None if equal."""
    for at, bt in zip(a_traces, b_traces):
        for i, (am, bm) in enumerate(zip(at["moves"], bt["moves"])):
            if am != bm:
                return (f"game seed={at['seed']} move {i}: "
                        f"odin={am}  cpp={bm}")
        # also compare final
        a_final = (at["final_board_sha"], at["final_score"], at["winner"], at["move_count"])
        b_final = (bt["final_board_sha"], bt["final_score"], bt["winner"], bt["move_count"])
        if a_final != b_final:
            return f"game seed={at['seed']} final: odin={a_final} cpp={b_final}"
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["odin", "cpp", "both"], default="both")
    p.add_argument("--games", type=int, default=DEFAULT_GAMES)
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--komi", type=float, default=DEFAULT_KOMI)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args()

    if args.backend in ("odin", "cpp"):
        fp, _ = run_one(args.backend, args)
        print(json.dumps({"backend": args.backend, "fingerprint": fp}, indent=2))
        return 0

    odin_fp, odin_traces = run_one("odin", args)
    cpp_fp, cpp_traces = run_one("cpp", args)
    same = odin_fp == cpp_fp
    out = {
        "odin": odin_fp,
        "cpp":  cpp_fp,
        "match": same,
        "games": args.games, "max_moves": args.max_moves, "size": args.size,
    }
    if not same:
        out["first_diff"] = diff_first(odin_traces, cpp_traces)
    print(json.dumps(out, indent=2))
    return 0 if same else 2


if __name__ == "__main__":
    raise SystemExit(main())
