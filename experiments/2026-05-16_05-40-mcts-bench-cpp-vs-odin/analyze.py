#!/usr/bin/env python3
"""Aggregate data.csv -> summary + figures/throughput.png."""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def mean_and_ci95(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, 0.0)
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return m, 1.96 * sd / math.sqrt(len(xs))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(HERE, "data.csv"))
    p.add_argument("--fig", default=os.path.join(HERE, "figures", "throughput.png"))
    args = p.parse_args()

    by_backend: dict[str, list[float]] = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            by_backend[row["backend"]].append(float(row["sims_per_sec"]))

    summary = {}
    for backend, xs in by_backend.items():
        m, ci = mean_and_ci95(xs)
        summary[backend] = (m, ci, len(xs))
        print(f"{backend:>6}: {m:>10,.0f} ± {ci:>6,.0f} sims/s (95% CI, n={len(xs)})")

    if "odin" in summary and "cpp" in summary:
        odin_m = summary["odin"][0]
        cpp_m = summary["cpp"][0]
        ratio = odin_m / cpp_m if cpp_m > 0 else float("inf")
        gap_pct = (odin_m - cpp_m) / cpp_m * 100 if cpp_m > 0 else 0.0
        print(f"\nOdin / C++ = {ratio:.3f}x   ({gap_pct:+.1f}%)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping figure.", flush=True)
        return 0

    backends = ["cpp", "odin"]
    means = [summary.get(b, (0, 0, 0))[0] for b in backends]
    cis = [summary.get(b, (0, 0, 0))[1] for b in backends]
    colors = ["#4c72b0", "#dd8452"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(backends, means, yerr=cis, color=colors, capsize=8,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MCTS simulations / sec (single thread)")
    ax.set_title("MCTS throughput: C++ pybind11 vs Odin ctypes\n"
                 "9x9, 1600 sims/move, 32 moves, uniform evaluator")
    for bar, m, ci in zip(bars, means, cis):
        ax.annotate(f"{m:,.0f}±{ci:,.0f}",
                    xy=(bar.get_x() + bar.get_width()/2, m),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=140)
    print(f"\nwrote {args.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
