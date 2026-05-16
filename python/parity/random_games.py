#!/usr/bin/env python3
"""Parity fingerprint harness for the autogodin Odin backend.

Plays N seeded random games and prints a deterministic fingerprint of:
- per-move Zobrist hash after each play / pass / rejection
- captures count delta
- ko_point flag
- final score, winner, move_count

The fingerprint is a SHA-256 of the structured trace. Identical algorithms
(same legality, same scoring, same Zobrist init) MUST yield identical
fingerprints regardless of language. This is the primary parity contract
between the Odin port and the upstream C++ reference.

USAGE:
    python python/parity/random_games.py                # print fingerprint
    python python/parity/random_games.py --emit fixture.json   # write trace
    python python/parity/random_games.py --check fixture.json  # diff vs file

When the C++ side is reachable in this repo (see bd autogodin-3xv.13), wire
a parallel `cpp_random_games.py` that produces the same JSON for diffing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

# Allow running from anywhere; assume repo root contains a python/ dir.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import alpha_go_odin as ag  # noqa: E402

DEFAULT_GAMES = 10
DEFAULT_MAX_MOVES = 200
DEFAULT_SIZE = 9
DEFAULT_KOMI = 7.5
PASS_PROB = 0.02  # passing is rare in random rollouts


def play_random_game(size: int, komi: float, seed: int, max_moves: int) -> dict:
    """One seeded random game. Returns a structured trace."""
    rng = random.Random(seed)
    b = ag.GoBoard(size, komi)
    n = size * size

    trace = {
        "seed": seed, "size": size, "komi": komi,
        "moves": [],
        "hash_seq": [b.current_hash()],
    }

    for move_idx in range(max_moves):
        if b.is_game_over():
            break

        # Random move: with PASS_PROB probability pass, else pick uniformly
        # among legal moves (no pass). If no legal moves, pass.
        legal = b.get_legal_moves_flat()
        action: int
        if rng.random() < PASS_PROB or not legal:
            action = ag.PASS_ACTION
            b.pass_move()
        else:
            # Deterministic: sort legal so iteration order doesn't depend on
            # ctypes buffer alignment / dict iter ordering.
            legal = sorted(legal)
            action = legal[rng.randrange(len(legal))]
            ok = b.play_flat(action)
            assert ok, f"legality bug: action {action} rejected after being in legal list"

        trace["moves"].append({
            "i": move_idx,
            "action": action,
            "to_play_after": b.to_play(),
            "ko_point": b.ko_point(),
            "hash": b.current_hash(),
        })
        trace["hash_seq"].append(b.current_hash())

    trace["final_score"] = round(b.score(), 4)
    trace["winner"] = b.get_winner()
    trace["move_count"] = b.move_count()
    return trace


def fingerprint_traces(traces: list[dict]) -> str:
    blob = json.dumps(traces, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, default=DEFAULT_GAMES)
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--komi", type=float, default=DEFAULT_KOMI)
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="Base seed; per-game seed is base + game_index.")
    p.add_argument("--emit", type=str, default=None,
                   help="Write the trace bundle to this JSON path.")
    p.add_argument("--check", type=str, default=None,
                   help="Verify the produced fingerprint matches the one in this JSON.")
    args = p.parse_args()

    traces = []
    for i in range(args.games):
        traces.append(play_random_game(args.size, args.komi, args.seed + i, args.max_moves))

    fp = fingerprint_traces(traces)
    summary = {
        "fingerprint": fp,
        "config": {
            "games": args.games, "max_moves": args.max_moves,
            "size": args.size, "komi": args.komi, "seed": args.seed,
            "pass_prob": PASS_PROB,
        },
        "summary": [
            {
                "seed": tr["seed"], "moves": tr["move_count"],
                "score": tr["final_score"], "winner": tr["winner"],
                "final_hash": tr["hash_seq"][-1],
            }
            for tr in traces
        ],
    }
    print(json.dumps(summary, indent=2))

    if args.emit:
        with open(args.emit, "w") as f:
            json.dump({"fingerprint": fp, "config": summary["config"], "traces": traces}, f)
        print(f"# wrote trace bundle to {args.emit}", file=sys.stderr)

    if args.check:
        with open(args.check) as f:
            golden = json.load(f)
        if golden.get("fingerprint") != fp:
            print(f"# MISMATCH: got {fp}, expected {golden['fingerprint']}", file=sys.stderr)
            # Find first diff for debugging
            for got_tr, want_tr in zip(traces, golden.get("traces", [])):
                for i, (g, w) in enumerate(zip(got_tr["moves"], want_tr["moves"])):
                    if g != w:
                        print(f"#  game seed={got_tr['seed']} first diff at move {i}",
                              file=sys.stderr)
                        print(f"#    got:    {g}", file=sys.stderr)
                        print(f"#    expect: {w}", file=sys.stderr)
                        return 2
            return 2
        print(f"# OK: fingerprint matches {args.check}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
