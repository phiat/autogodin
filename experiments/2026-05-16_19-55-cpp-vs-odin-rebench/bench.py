#!/usr/bin/env python3
"""Tier-1 rebench: Odin vs C++ throughput on miniwini in current state.

Closes two loose ends in one experiment:

1. The README's previous throughput numbers were not traceable to a single
   committed experiment (the C++ baseline '8,713' appears nowhere; the
   in-process Odin '76,159' was from cz9 but on miniwini in a state that
   today reproduces at roughly half-throughput). This bench locks fresh
   numbers to one host state.

2. The README admitted we had never run Odin vs C++ head-to-head with a
   real NN evaluator. Both backends actually expose `run_simulations_batched`
   (verified today — the 'C++ has no batched API' claim in earlier commit
   bodies was wrong). This bench runs both backends on the same NN forward
   in sequential and batched mode, so the realistic-workload ratio is no
   longer hand-waved.

All cells use the same evaluator factory across backends (same Python
callback for sequential; same body for batched, with the signature
difference C++ vs Odin handled by two factory functions). Both backends
go through their Python ctypes / pybind11 surface — that's the workload
a real user would see.

Run on miniwini (autogo/.venv has torch + alpha_go_cpp; alpha_go_odin
on PYTHONPATH=python).
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
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# Make alpha_go_odin (a source package in this repo) importable.
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

# alpha_go_cpp must come from a venv where the pybind11 wheel is
# installed (locally autogo/.venv-cpponly; on miniwini autogo/.venv).
# Pre-resolve PYTHONPATH or invoke this script from such a venv.
# We MUST NOT pick up the python/odin_backend shim — that aliases
# alpha_go_cpp -> alpha_go_odin and silently turns "C++ vs Odin" into
# "Odin vs Odin".

import alpha_go_cpp as ac
import alpha_go_odin as ao
from alpha_go.model import SizeInvariantGoResNet

# Sanity: alpha_go_cpp must NOT be the Odin shim.
assert ac.__name__ == "alpha_go_cpp" and "alpha_go_odin" not in ac.__file__, (
    f"alpha_go_cpp resolved to shim: {ac.__file__}. "
    f"Drop python/odin_backend from PYTHONPATH for this bench.")
assert ac.MCTSTree is not ao.MCTSTree, (
    "ac.MCTSTree is ao.MCTSTree — shim is still winning")

SIZE = 9
KOMI = 7.5
N_CELLS = SIZE * SIZE
N_ACTIONS = N_CELLS + 1
PASS_IDX = N_CELLS


# ----- evaluators --------------------------------------------------------

def make_uniform_evaluator(pass_action: int):
    """Cheap synthetic evaluator (value=0, uniform priors over legal).

    Same body works for both backends via the pass_action closure."""
    def ev(board):
        legal = board.get_legal_moves_flat()
        p = 1.0 / (len(legal) + 1)
        out = {a: p for a in legal}
        out[pass_action] = p
        return out, 0.0
    return ev


def _board_to_planes(board) -> np.ndarray:
    raw = board.to_numpy()
    cur = board.to_play()
    opp = ao.BLACK if cur == ao.WHITE else ao.WHITE
    b = np.zeros_like(raw, dtype=np.int64)
    b[raw == cur] = 1
    b[raw == opp] = 2
    return b


def make_net():
    torch.manual_seed(0)
    net = SizeInvariantGoResNet(channels=32, n_blocks=4, value_hidden=32)
    net.eval()
    return net


def make_nn_evaluator(net, pass_action: int):
    """Sequential NN evaluator — same body for both backends."""
    def ev(board):
        b = _board_to_planes(board)
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(b).unsqueeze(0),
                         torch.ones(1, SIZE, SIZE))
        probs = torch.softmax(pl[0], dim=-1).numpy()
        legal = board.get_legal_moves_flat()
        out = {a: float(probs[a]) for a in legal}
        out[pass_action] = float(probs[PASS_IDX])
        return out, float(torch.tanh(vl[0]).item())
    return ev


# C++ batched signature: callable(list[GoBoard]) -> list[(policy_dict, value)]
def make_cpp_batched_nn_evaluator(net, pass_action: int):
    def ev(views):
        B = len(views)
        batch = np.zeros((B, SIZE, SIZE), dtype=np.int64)
        for i, v in enumerate(views):
            batch[i] = _board_to_planes(v)
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(batch),
                         torch.ones(B, SIZE, SIZE))
        probs = torch.softmax(pl, dim=-1).numpy()
        values = torch.tanh(vl).numpy()
        out = []
        for i in range(B):
            legal = views[i].get_legal_moves_flat()
            policy = {a: float(probs[i, a]) for a in legal}
            policy[pass_action] = float(probs[i, PASS_IDX])
            out.append((policy, float(values[i])))
        return out
    return ev


# Odin batched signature: callable(list[GoBoard]) -> (list[policy_dict], list[value])
def make_odin_batched_nn_evaluator(net, pass_action: int):
    def ev(views):
        B = len(views)
        batch = np.zeros((B, SIZE, SIZE), dtype=np.int64)
        for i, v in enumerate(views):
            batch[i] = _board_to_planes(v)
        with torch.no_grad():
            pl, vl = net(torch.from_numpy(batch),
                         torch.ones(B, SIZE, SIZE))
        probs = torch.softmax(pl, dim=-1).numpy()
        values = torch.tanh(vl).numpy()
        policies = []
        for i in range(B):
            legal = views[i].get_legal_moves_flat()
            policy = {a: float(probs[i, a]) for a in legal}
            policy[pass_action] = float(probs[i, PASS_IDX])
            policies.append(policy)
        return policies, [float(v) for v in values]
    return ev


# ----- harness -----------------------------------------------------------

def make_cfg(backend, c_puct=1.0, max_depth=100):
    cfg = backend.MCTSConfig()
    cfg.c_puct = c_puct
    cfg.dirichlet_weight = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.lambda_ = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = max_depth
    return cfg


def run_trial(mode: str, num_sims: int, moves: int, seed: int,
              batch_size: int = 1, net=None):
    backend = ac if mode.startswith("cpp") else ao
    cfg = make_cfg(backend)
    board = backend.GoBoard(SIZE, KOMI)
    pass_action = backend.PASS_ACTION

    total_sims = 0
    t0 = time.perf_counter()
    for move in range(moves):
        if board.is_game_over():
            break
        if backend is ao:
            tree = ao.MCTSTree(board, cfg, seed=seed * 1000 + move)
        else:
            tree = ac.MCTSTree(board, cfg)

        if mode == "cpp_seq_uniform":
            tree.run_simulations(num_sims, make_uniform_evaluator(pass_action))
        elif mode == "odin_seq_uniform":
            tree.run_simulations(num_sims, make_uniform_evaluator(pass_action))
        elif mode == "cpp_seq_nn":
            tree.run_simulations(num_sims, make_nn_evaluator(net, pass_action))
        elif mode == "odin_seq_nn":
            tree.run_simulations(num_sims, make_nn_evaluator(net, pass_action))
        elif mode == "cpp_batched_nn":
            tree.run_simulations_batched(num_sims, batch_size,
                                          make_cpp_batched_nn_evaluator(net, pass_action))
        elif mode == "odin_batched_nn":
            tree.run_simulations_batched(num_sims, batch_size,
                                          make_odin_batched_nn_evaluator(net, pass_action))
        else:
            raise ValueError(mode)

        action = tree.select_action(0.0)
        if action == pass_action:
            board.pass_move()
        else:
            ok = board.play_flat(action)
            if not ok:
                board.pass_move()
        total_sims += num_sims
    dt = time.perf_counter() - t0
    return total_sims, dt


def bench_cell(label: str, mode: str, num_sims: int, moves: int,
               trials: int, batch_size: int = 1, net=None):
    _ = run_trial(mode, num_sims, moves, seed=42, batch_size=batch_size, net=net)
    rates = []
    for i in range(trials):
        sims, dt = run_trial(mode, num_sims, moves, seed=100 + i,
                             batch_size=batch_size, net=net)
        rates.append(sims / dt)
    m = mean(rates)
    sd = stdev(rates) if len(rates) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    print(f"  {label:<32s} {m:>10,.0f} +- {ci:>5,.0f} sims/sec   "
          f"(n={trials})")
    return m, ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uniform-sims", type=int, default=1600)
    p.add_argument("--uniform-moves", type=int, default=32)
    p.add_argument("--uniform-trials", type=int, default=3)
    p.add_argument("--nn-sims", type=int, default=800)
    p.add_argument("--nn-moves", type=int, default=16)
    p.add_argument("--nn-trials", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    print(f"=== Tier-1 rebench: Odin vs C++ throughput (miniwini, as-is) ===")
    print(f"torch threads: {torch.get_num_threads()}, "
          f"interop: {torch.get_num_interop_threads()}")
    print()

    print(f"== Uniform-policy evaluator  ({args.uniform_sims} sims x "
          f"{args.uniform_moves} moves x {args.uniform_trials} trials) ==")
    cpp_u, _ = bench_cell("cpp_seq_uniform", "cpp_seq_uniform",
                          args.uniform_sims, args.uniform_moves,
                          args.uniform_trials)
    odin_u, _ = bench_cell("odin_seq_uniform", "odin_seq_uniform",
                           args.uniform_sims, args.uniform_moves,
                           args.uniform_trials)
    print(f"  ratio odin/cpp = {odin_u/cpp_u:.2f}x (uniform-policy)")
    print()

    net = make_net()
    print(f"== Real NN evaluator (SizeInvariantGoResNet 32x4, random-init)  "
          f"({args.nn_sims} sims x {args.nn_moves} moves x "
          f"{args.nn_trials} trials) ==")
    print("Sequential:")
    cpp_s, _ = bench_cell("cpp_seq_nn", "cpp_seq_nn",
                          args.nn_sims, args.nn_moves, args.nn_trials,
                          net=net)
    odin_s, _ = bench_cell("odin_seq_nn", "odin_seq_nn",
                           args.nn_sims, args.nn_moves, args.nn_trials,
                           net=net)
    print(f"  ratio odin/cpp = {odin_s/cpp_s:.2f}x (sequential NN)")
    print()

    print(f"Batched (bs={args.batch}):")
    cpp_b, _ = bench_cell(f"cpp_batched_nn bs={args.batch}", "cpp_batched_nn",
                          args.nn_sims, args.nn_moves, args.nn_trials,
                          batch_size=args.batch, net=net)
    odin_b, _ = bench_cell(f"odin_batched_nn bs={args.batch}",
                           "odin_batched_nn",
                           args.nn_sims, args.nn_moves, args.nn_trials,
                           batch_size=args.batch, net=net)
    print(f"  ratio odin/cpp = {odin_b/cpp_b:.2f}x (batched NN bs={args.batch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
