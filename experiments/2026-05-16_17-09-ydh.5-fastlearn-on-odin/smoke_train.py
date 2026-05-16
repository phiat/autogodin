#!/usr/bin/env python3
"""ydh.5: smoke-train autogo's SizeInvariantGoResNet on Odin-generated data.

Goal: confirm the Odin backend (via `alpha_go_cpp` shim) plugs into
autogo's training stack end-to-end. Not a fastlearn Phase-A repro
(which needs parent's dataset-it10 + GPU); a sanity check that the
shim feeds the trainer and an MCTS evaluator both work.

Run with PYTHONPATH=python/odin_backend:python:autogo/src and a
GAME_DATA_DIR pointing at our Odin-generated random-selfplay set.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import alpha_go_cpp  # via shim — should resolve to python/odin_backend/alpha_go_cpp.py
from alpha_go.dataset import GoDataset
from alpha_go.model import SizeInvariantGoResNet, count_parameters

# --------------------------------------------------------------------
# Section 1: confirm shim wiring
# --------------------------------------------------------------------
shim_path = alpha_go_cpp.__file__
assert "odin_backend" in shim_path, f"alpha_go_cpp NOT resolving to shim: {shim_path}"
print(f"✓ alpha_go_cpp resolves to Odin shim: {shim_path}")

# --------------------------------------------------------------------
# Section 2: load Odin-generated self-play data
# --------------------------------------------------------------------
data_dir = Path(os.environ.get("GAME_DATA_DIR", "/tmp/ydh5-game-data")) / "ydh5-smoke-random"
print(f"loading data from {data_dir}")

ds = GoDataset(data_dir=str(data_dir),
               load_mcts_policy=False, load_is_teacher=False)
print(f"✓ GoDataset: {len(ds)} positions across {len(list(data_dir.glob('*.npz')))} games")
sample = ds[0]
print(f"  sample keys: {list(sample.keys())}")
print(f"  board shape: {sample['board'].shape}, move: {sample['move'].tolist()}")

# --------------------------------------------------------------------
# Section 3: build a small SizeInvariantGoResNet and run training
# --------------------------------------------------------------------
torch.manual_seed(0)
net = SizeInvariantGoResNet(channels=32, n_blocks=4, value_hidden=32)
n_params = count_parameters(net)
print(f"✓ SizeInvariantGoResNet built: {n_params:,} params")

# The model needs (board, mask) on the spatial axes — single fixed 9x9 here.
BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
PASS_INDEX = N_ACTIONS - 1


def collate(batch):
    boards = torch.stack([b["board"] for b in batch])  # (B, 9, 9), values 0/1/2
    moves = torch.stack([b["move"] for b in batch])  # (B, 2)
    winners = torch.tensor([b["winner"] for b in batch], dtype=torch.float32)
    mask = torch.ones(boards.shape[0], BOARD_SIZE, BOARD_SIZE)
    # Model takes raw (B, H, W) labels (0=empty, 1=self, 2=opp) and does
    # scatter internally; mask is (B, H, W) of 1.0s for size-homogeneous batches.
    flat = torch.where(moves[:, 0] < 0, torch.tensor(PASS_INDEX),
                       moves[:, 0] * BOARD_SIZE + moves[:, 1])
    return boards, mask, flat, winners


loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate, num_workers=0)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
net.train()
step = 0
t0 = time.perf_counter()
losses = []
for epoch in range(5):
    for boards, mask, move_idx, winners in loader:
        out = net(boards, mask)
        # SizeInvariantGoResNet returns (policy_logits, value_logits)
        if isinstance(out, tuple):
            policy_logits, value_logits = out
        else:
            policy_logits = out["policy"]
            value_logits = out["value"]
        policy_loss = F.cross_entropy(policy_logits, move_idx)
        value_loss = F.binary_cross_entropy_with_logits(
            value_logits.squeeze(-1), winners
        )
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        step += 1
        if step % 5 == 0:
            print(f"  step {step:3d}  loss={loss:.4f}  "
                  f"policy={policy_loss:.4f}  value={value_loss:.4f}")
        if step >= 50:
            break
    if step >= 50:
        break

dt = time.perf_counter() - t0
print(f"✓ training: {step} steps in {dt:.2f}s "
      f"({step / dt:.1f} steps/sec); loss {losses[0]:.4f} → {losses[-1]:.4f}")

# --------------------------------------------------------------------
# Section 4: confirm MCTSTree (Odin) wired to a Python NN evaluator works
# --------------------------------------------------------------------
net.eval()
print("\n--- MCTS smoke through Odin shim ---")
cfg = alpha_go_cpp.MCTSConfig()
cfg.c_puct = 1.0
cfg.dirichlet_weight = 0.0
cfg.temperature = 1.0
cfg.max_depth = 50

board = alpha_go_cpp.GoBoard(BOARD_SIZE, 7.5)


def nn_evaluator(b):
    legal = b.get_legal_moves_flat()
    raw = b.to_numpy()  # (H, W) int8 with values BLACK/WHITE/EMPTY
    cur = b.to_play()
    opp = alpha_go_cpp.GoBoard.BLACK if cur == alpha_go_cpp.GoBoard.WHITE \
        else alpha_go_cpp.GoBoard.WHITE
    board_flipped = np.zeros_like(raw)
    board_flipped[raw == cur] = 1
    board_flipped[raw == opp] = 2
    board_tensor = torch.from_numpy(board_flipped.astype(np.int64)).unsqueeze(0)
    mask = torch.ones(1, BOARD_SIZE, BOARD_SIZE)
    with torch.no_grad():
        pl, vl = net(board_tensor, mask)
    probs = torch.softmax(pl[0], dim=-1).cpu().numpy()
    out = {a: float(probs[a]) for a in legal}
    out[alpha_go_cpp.PASS_ACTION] = float(probs[PASS_INDEX])
    return out, float(torch.tanh(vl[0]).item())


tree = alpha_go_cpp.MCTSTree(board, cfg)
t0 = time.perf_counter()
tree.run_simulations(200, nn_evaluator)
dt = time.perf_counter() - t0
probs = tree.get_action_probabilities(1.0)
top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
print(f"✓ MCTS 200 sims in {dt:.2f}s ({200/dt:.0f} sims/sec)")
print(f"  top-5 action probs: {[(a, f'{p:.3f}') for a, p in top]}")

print("\n=== ydh.5 smoke: PASSED ===")
print("  - alpha_go_cpp shim resolves to Odin")
print("  - autogo GoDataset reads Odin-generated NPZ")
print("  - SizeInvariantGoResNet trains end-to-end on Odin data")
print("  - MCTSTree+NN evaluator works through the shim")
