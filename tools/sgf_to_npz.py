#!/usr/bin/env python3
"""SGF -> autogo-NPZ converter for KataGo training/rating games.

Filters input SGFs to 9x9, no-handicap, with a parseable result. Replays
moves through alpha_go_cpp.GoBoard so the emitted boards/moves/winner
schema matches what autogo's GoDataset expects.

Input: directory tree of .sgf files (will recurse).
Output: directory of .npz files, one per kept game.

Usage:
    PYTHONPATH=python:autogo/src autogo/.venv-cpponly/bin/python \\
        tools/sgf_to_npz.py <in_dir> <out_dir> [--max N] [--board-size 9]

Smoke (5 games, see they parse + load via GoDataset):
    ... tools/sgf_to_npz.py /tmp/2020-12-08sgfs /tmp/converted-smoke --max 5
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np

import alpha_go_cpp


# SGF property extractor: tolerates whitespace, captures the [..] payload.
# Properties we read: SZ, KM, HA, RU, PB, PW, RE.
_PROP = re.compile(r"([A-Z]{1,2})\[((?:\\\]|[^\]])*)\]")


def parse_sgf_header(text: str) -> dict[str, str]:
    """Return the first occurrence of each requested property."""
    out: dict[str, str] = {}
    wanted = {"SZ", "KM", "HA", "RU", "PB", "PW", "RE"}
    for m in _PROP.finditer(text):
        k = m.group(1)
        if k in wanted and k not in out:
            out[k] = m.group(2)
        if len(out) == len(wanted):
            break
    return out


def parse_sgf_moves(text: str) -> list[tuple[int, tuple[int, int] | None]]:
    """Walk B[..]/W[..] moves in order. Returns [(color, (col,row) or None), ...]
    where color is 1=BLACK, 2=WHITE; None = pass."""
    out: list[tuple[int, tuple[int, int] | None]] = []
    # SGF coords: 'a'..'i' for 9x9. Empty value ([]) = pass. Legacy 19x19 also
    # uses "tt" for pass; not relevant at 9x9 but harmless to map to None.
    move_re = re.compile(r";\s*([BW])\[((?:[a-z]{0,2}))\]")
    for m in move_re.finditer(text):
        color = 1 if m.group(1) == "B" else 2
        coord = m.group(2)
        if coord == "" or coord == "tt":
            out.append((color, None))
        elif len(coord) == 2:
            col = ord(coord[0]) - ord("a")
            row = ord(coord[1]) - ord("a")
            out.append((color, (col, row)))
        else:
            # Malformed, skip the whole move stream
            return []
    return out


def parse_result(re_str: str) -> tuple[int, str] | None:
    """Returns (winner_int, normalized_result_str) or None to skip.

    winner: 1=BLACK, 2=WHITE, 0=draw.
    """
    s = re_str.strip().lower()
    if not s or s in {"void", "?", "0", "draw"}:
        # No result or draw -> represent as draw if explicit, else skip.
        return (0, "Draw") if s == "draw" else None
    if s.startswith("b+"):
        return 1, f"B+{re_str.split('+', 1)[1]}"
    if s.startswith("w+"):
        return 2, f"W+{re_str.split('+', 1)[1]}"
    return None


def replay(moves: list[tuple[int, tuple[int, int] | None]], board_size: int,
            komi: float) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Replay through a GoBoard. Returns (boards_before, moves_array, termination)
    or None if an illegal move was encountered.

    GoBoard tracks to_play internally and auto-alternates after each play/pass,
    so we ignore the SGF-declared color and call play(row,col) / pass_move()
    directly. If the SGF declares a color that conflicts with board.to_play()
    we insert a pass to realign (handles e.g. a missing pass record).
    """
    b = alpha_go_cpp.GoBoard(board_size, komi)
    boards = []
    move_arr = []
    for color, mv in moves:
        if int(b.to_play()) != color:
            # Insert a no-op pass to realign turn parity. Pass is legal anytime
            # and doesn't get its own boards/moves row (it's not in the SGF).
            b.pass_move()
        # Snapshot the board BEFORE this move (autogo's schema: the move at
        # index i was played FROM boards[i]).
        boards.append(b.to_numpy().copy().astype(np.int8))
        if mv is None:
            move_arr.append((-1, -1))
            b.pass_move()
        else:
            col, row = mv
            if not b.is_legal(row, col):
                return None
            b.play(row, col)
            move_arr.append((row, col))
    return (np.stack(boards, axis=0).astype(np.int8),
            np.array(move_arr, dtype=np.int16),
            "natural")


def convert_sgf(path: Path, board_size: int, allow_handicap: bool,
                komi_range: tuple[float, float] | None) -> dict | None:
    """Returns dict ready for np.savez or None if game should be skipped."""
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        text = path.read_bytes().decode("utf-8", errors="ignore")

    hdr = parse_sgf_header(text)
    sz = int(hdr.get("SZ", "0") or 0)
    if sz != board_size:
        return None
    ha = int(hdr.get("HA", "0") or 0)
    if not allow_handicap and ha != 0:
        return None
    try:
        komi = float(hdr.get("KM", "7.5"))
    except ValueError:
        komi = 7.5
    if komi_range is not None and not (komi_range[0] <= komi <= komi_range[1]):
        return None

    re_parsed = parse_result(hdr.get("RE", ""))
    if re_parsed is None:
        return None
    winner, result_str = re_parsed

    moves = parse_sgf_moves(text)
    if not moves:
        return None

    replay_out = replay(moves, board_size, komi)
    if replay_out is None:
        return None
    boards, move_arr, termination = replay_out

    # Approximate termination from the result string.
    if "+R" in result_str or "+Resign" in result_str:
        termination = "resign"
    elif winner == 0:
        termination = "draw"

    return dict(
        boards=boards,
        moves=move_arr,
        winner=np.int8(winner),
        result=str(result_str),
        board_size=np.int16(board_size),
        black_agent=str(hdr.get("PB", "")),
        white_agent=str(hdr.get("PW", "")),
        black_checkpoint_path="",
        white_checkpoint_path="",
        num_moves=np.int32(len(moves)),
        komi=np.float32(komi),
        termination=str(termination),
        code_version="sgf_to_npz/v1",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("in_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--max", type=int, default=0,
                   help="Max games to write (0 = unlimited)")
    p.add_argument("--allow-handicap", action="store_true")
    p.add_argument("--komi-min", type=float, default=None)
    p.add_argument("--komi-max", type=float, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    komi_range = None
    if args.komi_min is not None and args.komi_max is not None:
        komi_range = (args.komi_min, args.komi_max)

    sgfs = sorted(args.in_dir.rglob("*.sgf"))
    if not sgfs:
        print(f"No .sgf files found under {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    stats = {"scanned": 0, "kept": 0, "skip_size": 0, "skip_handicap": 0,
             "skip_komi": 0, "skip_no_result": 0, "skip_no_moves": 0,
             "skip_illegal_move": 0, "skip_other": 0}
    written = 0
    for path in sgfs:
        stats["scanned"] += 1
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            text = path.read_bytes().decode("utf-8", errors="ignore")
        hdr = parse_sgf_header(text)
        sz_str = hdr.get("SZ", "0") or "0"
        if ":" in sz_str:
            # Rectangular board, e.g. "15:14" — not square, skip.
            stats["skip_size"] += 1; continue
        try:
            sz = int(sz_str)
        except ValueError:
            stats["skip_size"] += 1; continue
        if sz != args.board_size:
            stats["skip_size"] += 1; continue
        ha = int(hdr.get("HA", "0") or 0)
        if not args.allow_handicap and ha != 0:
            stats["skip_handicap"] += 1; continue
        try:
            komi = float(hdr.get("KM", "7.5"))
        except ValueError:
            komi = 7.5
        if komi_range and not (komi_range[0] <= komi <= komi_range[1]):
            stats["skip_komi"] += 1; continue
        re_parsed = parse_result(hdr.get("RE", ""))
        if re_parsed is None:
            stats["skip_no_result"] += 1; continue
        moves = parse_sgf_moves(text)
        if not moves:
            stats["skip_no_moves"] += 1; continue
        replay_out = replay(moves, args.board_size, komi)
        if replay_out is None:
            stats["skip_illegal_move"] += 1; continue
        boards, move_arr, termination = replay_out
        winner, result_str = re_parsed
        if "+R" in result_str:
            termination = "resign"
        elif winner == 0:
            termination = "draw"

        out_path = args.out_dir / (path.stem + ".npz")
        np.savez(out_path,
                 boards=boards, moves=move_arr,
                 winner=np.int8(winner),
                 result=str(result_str),
                 board_size=np.int16(args.board_size),
                 black_agent=str(hdr.get("PB", "")),
                 white_agent=str(hdr.get("PW", "")),
                 black_checkpoint_path="",
                 white_checkpoint_path="",
                 num_moves=np.int32(len(moves)),
                 komi=np.float32(komi),
                 termination=str(termination),
                 code_version="sgf_to_npz/v1")
        stats["kept"] += 1
        written += 1
        if not args.quiet and written <= 3:
            print(f"  wrote {out_path.name}  moves={len(moves)} "
                  f"komi={komi} result={result_str}")
        if args.max and written >= args.max:
            break

    print()
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k:>20s}: {v}")
    print(f"Wrote {written} NPZs to {args.out_dir}")


if __name__ == "__main__":
    main()
