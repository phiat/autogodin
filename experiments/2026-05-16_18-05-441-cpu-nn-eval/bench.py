#!/usr/bin/env python3
"""441: CPU-only NN-eval A/B over the four MCTS Python paths.

Wraps a randomly-initialized SizeInvariantGoResNet(channels=32, n_blocks=4)
as evaluators for:
  - run_simulations          (dict)
  - run_simulations_flat     (cz9 scratch-ndarray)
  - run_simulations_batched  (dict)
  - run_simulations_batched_flat   (cg0)
  - run_simulations_threaded (n_threads = 2, 4, 8)

Characterizes the throughput x batch x threading envelope with a *real*
(though small) torch evaluator on CPU. The 9x9 GoResNet here is the same
shape ydh.5 actually trained (76,323 params).

Run with autogo/.venv on PYTHONPATH= python:autogo/src.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from statistics import mean, stdev

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))

import alpha_go_odin as ao
from alpha_go.model import SizeInvariantGoResNet

SIZE = 9
KOMI = 7.5
N_ACTIONS = SIZE * SIZE + 1
PASS_IDX = N_ACTIONS - 1
PASS_ACTION = ao.PASS_ACTION


def make_net():
    torch.manual_seed(0)
    net = SizeInvariantGoResNet(channels=32, n_blocks=4, value_hidden=32)
    net.eval()
    return net


# --- single-leaf evaluators ----------------------------------------------

def make_dict_evaluator(net):
    def evaluator(board):
        legal = board.get_legal_moves_flat()
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
        out = {a: float(probs[a]) for a in legal}
        out[PASS_ACTION] = float(probs[PASS_IDX])
        return out, float(torch.tanh(vl[0]).item())
    return evaluator


def make_flat_evaluator(net):
    def evaluator(board, out_actions, out_probs):
        legal = board.get_legal_moves_flat()
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
        n = len(legal)
        out_actions[:n] = legal
        out_actions[n] = PASS_ACTION
        for i, a in enumerate(legal):
            out_probs[i] = probs[a]
        out_probs[n] = probs[PASS_IDX]
        return n + 1, float(torch.tanh(vl[0]).item())
    return evaluator


# --- batched evaluators --------------------------------------------------

def make_batched_dict_evaluator(net):
    def evaluator(views):
        B = len(views)
        batch = np.zeros((B, SIZE, SIZE), dtype=np.int64)
        for i, v in enumerate(views):
            raw = v.to_numpy()
            cur = v.to_play()
            opp = ao.BLACK if cur == ao.WHITE else ao.WHITE
            batch[i][raw == cur] = 1
            batch[i][raw == opp] = 2
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(batch),
                         torch.ones(B, SIZE, SIZE))
        probs = torch.softmax(pl, dim=-1).numpy()
        values = torch.tanh(vl).numpy()
        policies = []
        for i in range(B):
            legal = views[i].get_legal_moves_flat()
            out = {a: float(probs[i, a]) for a in legal}
            out[PASS_ACTION] = float(probs[i, PASS_IDX])
            policies.append(out)
        return policies, [float(v) for v in values]
    return evaluator


def make_batched_flat_evaluator(net):
    def evaluator(views, out_actions, out_probs, out_counts, out_values):
        B = len(views)
        batch = np.zeros((B, SIZE, SIZE), dtype=np.int64)
        for i, v in enumerate(views):
            raw = v.to_numpy()
            cur = v.to_play()
            opp = ao.BLACK if cur == ao.WHITE else ao.WHITE
            batch[i][raw == cur] = 1
            batch[i][raw == opp] = 2
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(batch),
                         torch.ones(B, SIZE, SIZE))
        probs = torch.softmax(pl, dim=-1).numpy()
        values = torch.tanh(vl).numpy()
        for i in range(B):
            legal = views[i].get_legal_moves_flat()
            n = len(legal)
            out_actions[i, :n] = legal
            out_actions[i, n] = PASS_ACTION
            for j, a in enumerate(legal):
                out_probs[i, j] = probs[i, a]
            out_probs[i, n] = probs[i, PASS_IDX]
            out_counts[i] = n + 1
            out_values[i] = float(values[i])
    return evaluator


# --- trial harness -------------------------------------------------------

def run_trial(mode: str, num_sims: int, moves: int, seed: int,
              batch_size: int = 1, n_threads: int = 0):
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.lambda_ = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.dirichlet_weight = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = 100
    net = make_net()

    board = ao.GoBoard(SIZE, KOMI)
    total_sims = 0
    t0 = time.perf_counter()
    for move in range(moves):
        if board.is_game_over():
            break
        tree = ao.MCTSTree(board, cfg, seed=seed * 1000 + move)
        if mode == "seq_dict":
            tree.run_simulations(num_sims, make_dict_evaluator(net))
        elif mode == "seq_flat":
            tree.run_simulations_flat(num_sims, make_flat_evaluator(net))
        elif mode == "batched_dict":
            tree.run_simulations_batched(num_sims, batch_size, make_batched_dict_evaluator(net))
        elif mode == "batched_flat":
            tree.run_simulations_batched_flat(num_sims, batch_size, make_batched_flat_evaluator(net))
        elif mode == "threaded":
            tree.run_simulations_threaded(num_sims, n_threads, make_dict_evaluator(net))
        else:
            raise ValueError(mode)
        action = tree.select_action(0.0)
        if action == ao.PASS_ACTION:
            board.pass_move()
        else:
            ok = board.play_flat(action)
            if not ok:
                board.pass_move()
        total_sims += num_sims
    dt = time.perf_counter() - t0
    return total_sims, dt


def bench_cell(label: str, mode: str, num_sims: int, moves: int, trials: int, **kw):
    _ = run_trial(mode, num_sims, moves, seed=42, **kw)
    rates = []
    for i in range(trials):
        sims, dt = run_trial(mode, num_sims, moves, seed=100 + i, **kw)
        rates.append(sims / dt)
    m = mean(rates)
    sd = stdev(rates) if len(rates) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    print(f"  {label:<30s} {m:>8,.0f} +-{ci:>3.0f} sims/sec")
    return m, ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-sims", type=int, default=800,
                   help="sims per move (default 800 for CPU NN)")
    p.add_argument("--moves", type=int, default=16,
                   help="moves per trial (default 16)")
    p.add_argument("--trials", type=int, default=2)
    args = p.parse_args()

    print(f"=== 441: CPU-only NN-eval A/B (SizeInvariantGoResNet 32ch x 4b) ===")
    print(f"config: {args.num_sims} sims/move x {args.moves} moves x "
          f"{args.trials} trials per cell")
    print(f"torch threads: {torch.get_num_threads()}, "
          f"interop: {torch.get_num_interop_threads()}")
    print()

    print("Sequential (single-leaf eval per simulation):")
    bench_cell("seq dict eval", "seq_dict", args.num_sims, args.moves, args.trials)
    bench_cell("seq flat eval (cz9)", "seq_flat", args.num_sims, args.moves, args.trials)
    print()

    print("Batched (leaf-parallel + virtual loss):")
    for bs in (8, 32, 128):
        bench_cell(f"batched dict bs={bs}", "batched_dict",
                   args.num_sims, args.moves, args.trials, batch_size=bs)
    for bs in (32, 128):
        bench_cell(f"batched flat bs={bs} (cg0)", "batched_flat",
                   args.num_sims, args.moves, args.trials, batch_size=bs)
    print()

    # Threaded path: each MCTS worker thread calls torch independently.
    # If torch keeps its default (=ncores) intraop threads, the MCTS
    # threads * torch's intraop threads contend for CPU. Set torch to 1
    # thread so MCTS threading provides the parallelism. Run baseline at
    # the same setting for fairness.
    orig_t = torch.get_num_threads()
    torch.set_num_threads(1)
    print(f"Threaded (root-parallel; torch threads now {torch.get_num_threads()}):")
    bench_cell(f"  seq dict (torch=1, baseline)", "seq_dict",
               args.num_sims, args.moves, args.trials)
    for n in (1, 2, 4, 8):
        bench_cell(f"threaded n={n}", "threaded",
                   args.num_sims, args.moves, args.trials, n_threads=n)
    torch.set_num_threads(orig_t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
