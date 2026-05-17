#!/usr/bin/env python3
"""Score the 12g.4 gauntlet: winrate matrix + Wilson CI per (cand, ref) pair.

Reads NPZs under gauntlet_data/gauntlet/<cand>-vs-<ref>/{asB,asW}/, counts
candidate wins, prints a markdown table + writes results.json.

NPZ structure (per game): keys 'winner' (1=black, 2=white, 0=draw),
'black_agent' / 'white_agent' strings, 'termination'.

Run:
    python score.py
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
DATA = EXP / "gauntlet_data" / "gauntlet"

# Order is what appears in tables; budgets first, then references, then random.
CANDIDATES = [
    ("budget-S-small-3M", "S", "small-3M"),
    ("budget-S-mid-7M",   "S", "mid-7M"),
    ("budget-S-big-18M",  "S", "big-18M"),
    ("budget-M-small-3M", "M", "small-3M"),
    ("budget-M-mid-7M",   "M", "mid-7M"),
    ("budget-M-big-18M",  "M", "big-18M"),
    ("budget-L-small-3M", "L", "small-3M"),
    ("budget-L-mid-7M",   "L", "mid-7M"),
    ("budget-L-big-18M",  "L", "big-18M"),
]
REFS = ["random", "ref-bpoC-iter0", "ref-bpoC-iter4"]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (winrate, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def score_match(match_dir: Path, cand_color: str) -> tuple[int, int, int, list[str]]:
    """Walk one match dir; return (cand_wins, draws, total, terminations)."""
    if not match_dir.exists():
        return 0, 0, 0, []
    wins = draws = 0
    terms = []
    total = 0
    for npz in sorted(match_dir.glob("*.npz")):
        try:
            d = np.load(npz, allow_pickle=True)
        except Exception as e:
            print(f"WARN: failed to load {npz.name}: {e}")
            continue
        w = int(d["winner"])
        terms.append(str(d.get("termination", "")))
        if w == 0:
            draws += 1
        elif (w == 1 and cand_color == "B") or (w == 2 and cand_color == "W"):
            wins += 1
        total += 1
    return wins, draws, total, terms


def main():
    results: dict = {}  # results[cand][ref] = {asB: (w,d,n), asW: (w,d,n), agg: ...}
    for cand_name, budget, model in CANDIDATES:
        results[cand_name] = {"_budget": budget, "_model": model}
        for ref in REFS:
            pair_dir = DATA / f"{cand_name}-vs-{ref}"
            wB, dB, nB, tB = score_match(pair_dir / "asB", "B")
            wW, dW, nW, tW = score_match(pair_dir / "asW", "W")
            tot_w = wB + wW
            tot_d = dB + dW
            tot_n = nB + nW
            p, lo, hi = wilson(tot_w, tot_n)
            results[cand_name][ref] = {
                "asB_wins": wB, "asB_total": nB,
                "asW_wins": wW, "asW_total": nW,
                "draws": tot_d,
                "wins": tot_w, "total": tot_n,
                "winrate": p, "wilson_lo": lo, "wilson_hi": hi,
                "asB_terminations": _term_summary(tB),
                "asW_terminations": _term_summary(tW),
            }

    (EXP / "results.json").write_text(json.dumps(results, indent=2))
    _print_tables(results)


def _term_summary(terms: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for t in terms:
        out[t] += 1
    return dict(out)


def _print_tables(results: dict):
    # Header line
    print()
    print("## Winrate matrix (candidate vs reference, 40 games each = 20 as B + 20 as W)")
    print()
    print("| candidate | vs random | vs bpoC iter0 | vs bpoC iter4 |")
    print("|---|---:|---:|---:|")
    for cand_name, *_ in CANDIDATES:
        row = [f"`{cand_name}`"]
        for ref in REFS:
            r = results[cand_name][ref]
            row.append(f"{r['winrate']*100:5.1f}% [{r['wilson_lo']*100:.0f},{r['wilson_hi']*100:.0f}] ({r['wins']}/{r['total']})")
        print("| " + " | ".join(row) + " |")

    print()
    print("## Per-budget summary (mean winrate vs the trained refs only)")
    print()
    print("| budget | model | sims-equiv-FLOPs/move | vs iter0 | vs iter4 |")
    print("|---|---|---|---:|---:|")
    # Reverse-look up sims-per-move from flops.py
    from flops import CANDIDATES as MODELS, matched_pairs, BUDGETS
    budget_target = dict(BUDGETS)
    for cand_name, budget, model in CANDIDATES:
        # find sims for this (budget, model)
        target = budget_target[f"budget-{budget}"]
        sims = None
        for spec, s, _ in matched_pairs(target):
            if spec.name == model:
                sims = s
                break
        r_i0 = results[cand_name]["ref-bpoC-iter0"]
        r_i4 = results[cand_name]["ref-bpoC-iter4"]
        print(f"| {budget} | {model} | {sims} sims | "
              f"{r_i0['winrate']*100:.1f}% | {r_i4['winrate']*100:.1f}% |")

    print()
    print("## vs random sanity floor")
    print()
    print("All cells should clear ~90% vs random; failures mean the model isn't")
    print("meaningfully better than coin-flip.")
    print()
    for cand_name, budget, model in CANDIDATES:
        r = results[cand_name]["random"]
        flag = "" if r["winrate"] >= 0.90 else " ← below 90%"
        print(f"  {cand_name:30s} {r['winrate']*100:5.1f}% ({r['wins']}/{r['total']}){flag}")

    print()
    print("Wrote: results.json")


if __name__ == "__main__":
    main()
