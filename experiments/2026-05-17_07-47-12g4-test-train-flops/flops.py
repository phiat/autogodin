#!/usr/bin/env python3
"""Per-move FLOP estimator for the 12g.4 sweep.

Counts the dense forward-pass FLOPs of a SizeInvariantGoResNet config
on a 9x9 board, multiplied by sims-per-move. Conv FLOPs use:
    2 * C_in * C_out * K * K * H * W
per conv layer. ResNet block has 2 convs (3x3) plus SE; SE is small
relative to convs so we account for it but the conv terms dominate.

Tied to autogo's model.py shape:
  - stem: 9x9x{INPUT_PLANES=17} -> 9x9xC
  - n_blocks of (conv 3x3 -> conv 3x3 -> SE)
  - policy head: 1x1 conv -> linear to 82 logits
  - value head: 1x1 conv -> avg-pool -> linear chain to 1 scalar

Numbers are rough (we don't count adds explicitly, biases, activations,
batchnorm, etc.) — purpose is to land within ~20% of true FLOPs so that
matched-budget configurations are *fair*, not exact.

Smoke / dry-run:
    python flops.py
"""
from __future__ import annotations

from dataclasses import dataclass

INPUT_PLANES = 17  # matches autogo/src/alpha_go/model.py
BOARD_HW = 9 * 9   # 81 spatial positions
NUM_ACTIONS = 82   # 81 board moves + 1 pass


@dataclass
class ModelSpec:
    channels: int
    n_blocks: int
    policy_channels: int = 2
    value_channels: int = 1
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.channels}ch-{self.n_blocks}b"


def conv_flops(c_in: int, c_out: int, k: int, hw: int = BOARD_HW) -> int:
    """FLOPs of one Conv2d layer: 2 * C_in * C_out * K * K * H * W."""
    return 2 * c_in * c_out * k * k * hw


def se_flops(c: int, reduction: int = 4) -> int:
    """SE block: global avg pool (negligible) + 2 dense layers."""
    mid = max(c // reduction, 8)
    return 2 * c * mid + 2 * mid * c


def block_flops(c: int) -> int:
    """One residual block: 2x (3x3 conv) + SE."""
    return 2 * conv_flops(c, c, 3) + se_flops(c)


def forward_flops(spec: ModelSpec) -> int:
    """Total FLOPs for one forward pass through the model."""
    flops = 0
    # Stem: 3x3 conv from INPUT_PLANES to channels
    flops += conv_flops(INPUT_PLANES, spec.channels, 3)
    # Residual tower
    flops += spec.n_blocks * block_flops(spec.channels)
    # Policy head: 1x1 conv -> flatten -> linear to 82
    flops += conv_flops(spec.channels, spec.policy_channels, 1)
    flops += 2 * spec.policy_channels * BOARD_HW * NUM_ACTIONS
    # Value head: 1x1 conv -> avg-pool -> linear -> linear
    flops += conv_flops(spec.channels, spec.value_channels, 1)
    flops += 2 * spec.value_channels * 64       # value_hidden = 64
    flops += 2 * 64 * 1                          # to scalar
    return flops


def param_count(spec: ModelSpec) -> int:
    """Approximate parameter count (conv weights + SE + heads)."""
    p = 0
    # Stem
    p += INPUT_PLANES * spec.channels * 9
    # Blocks: 2x conv3x3 + SE
    p += spec.n_blocks * (2 * spec.channels * spec.channels * 9
                           + 2 * (spec.channels * max(spec.channels // 4, 8)))
    # Policy head
    p += spec.channels * spec.policy_channels + spec.policy_channels * BOARD_HW * NUM_ACTIONS
    # Value head
    p += spec.channels * spec.value_channels + spec.value_channels * 64 + 64
    return p


def fmt_flops(n: int) -> str:
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}K"
    return str(n)


CANDIDATES = [
    ModelSpec(128, 10, policy_channels=32, value_channels=32, name="small-3M"),
    ModelSpec(192, 12, policy_channels=48, value_channels=48, name="mid-7M"),
    ModelSpec(256, 14, policy_channels=64, value_channels=64, name="big-18M"),
]


def matched_pairs(target_flops_per_move: float) -> list[tuple[ModelSpec, int, float]]:
    """For each candidate, return (spec, sims, actual_total_FLOPs) such
    that sims * forward_flops(spec) is closest to target without going
    below 1 sim."""
    out = []
    for spec in CANDIDATES:
        ff = forward_flops(spec)
        sims = max(1, round(target_flops_per_move / ff))
        out.append((spec, sims, ff * sims))
    return out


BUDGETS = [
    ("budget-S", 5e9),    # ~5 GFLOPs/move
    ("budget-M", 50e9),   # ~50 GFLOPs/move
    ("budget-L", 200e9),  # ~200 GFLOPs/move
]


def main():
    print(f"{'config':>20} {'params':>10} {'fwd FLOPs':>12} ")
    print("-" * 50)
    for spec in CANDIDATES:
        print(f"{spec.name:>20} {fmt_flops(param_count(spec)):>10} {fmt_flops(forward_flops(spec)):>12}")

    print()
    for budget_name, target in BUDGETS:
        print(f"=== {budget_name} (~{fmt_flops(int(target))}/move) ===")
        print(f"{'config':>20} {'sims':>6} {'actual':>10}")
        for spec, sims, total in matched_pairs(target):
            print(f"{spec.name:>20} {sims:>6} {fmt_flops(int(total)):>10}")
        print()


if __name__ == "__main__":
    main()
