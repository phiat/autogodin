"""alpha_go_cpp shim — redirects `import alpha_go_cpp` to alpha_go_odin.

Activated by putting this directory first on PYTHONPATH; use
`scripts/run_with_odin_backend.sh <cmd>` for the canonical entry point.

Goal: upstream autogo test/driver code that does `import alpha_go_cpp` and
calls the pybind11 surface (`GoBoard`, `MCTSConfig`, `MCTSTree`,
`PASS_ACTION`, `run_mcts`) runs unchanged against the Odin backend.

If you add a new symbol to alpha_go_cpp upstream, add the corresponding
re-export here. If it's missing on the Odin side, surface a clear
AttributeError rather than silently fall through.
"""
from __future__ import annotations

import alpha_go_odin as _ao

# Core OO API.
GoBoard = _ao.GoBoard
MCTSConfig = _ao.MCTSConfig
MCTSTree = _ao.MCTSTree

# Module-level constants pybind11 exposes.
PASS_ACTION = _ao.PASS_ACTION


def run_mcts(state, num_simulations, config, evaluator, temperature: float = 1.0):
    """pybind11 alpha_go_cpp.run_mcts equivalent.

    Run num_simulations MCTS playouts from state under config, then return
    a dict[action -> probability] sampled at the given temperature
    (temperature=0 → argmax-style one-hot peak).
    """
    tree = MCTSTree(state, config)
    tree.run_simulations(num_simulations, evaluator)
    return tree.get_action_probabilities(temperature)


__all__ = ["GoBoard", "MCTSConfig", "MCTSTree", "PASS_ACTION", "run_mcts"]
