#!/usr/bin/env python3
"""7km regression: alpha_go_odin batched trampoline accepts both
evaluator return shapes.

Background: `python/odin_backend/alpha_go_cpp.py` aliases
`alpha_go_cpp.MCTSTree` to `alpha_go_odin.MCTSTree` so upstream autogo
code that does `import alpha_go_cpp` runs unchanged against Odin. But
`run_simulations_batched` has historically been called with two
DIFFERENT evaluator return signatures depending on which backend the
caller was written for:

  - alpha_go_cpp shape:  evaluator(views) -> list[(policy_dict, value)]
  - alpha_go_odin shape: evaluator(views) -> (list[policy_dict], list[value])

Before this fix, the Odin trampoline only accepted the second shape;
running CppMCTSAgent + LeafBatchedNNEvaluator through the shim crashed
on `policies, values = result` unpack with the C++ shape (hit during
autogodin-ydh.8 step 3).

This script exercises both shapes against alpha_go_odin.MCTSTree
directly (no shim, no C++ wheel needed) and asserts they produce
identical tree state.

Run:
  PYTHONPATH=python autogo/.venv/bin/python python/parity/batched_shape_compat.py
"""
from __future__ import annotations

import sys

import alpha_go_odin as ao


SIZE = 9
KOMI = 7.5
N_CELLS = SIZE * SIZE
NUM_SIMS = 100
BATCH_SIZE = 16


def make_evaluator_odin_shape(pass_action: int):
    """Native Odin batched evaluator: returns (policies_list, values_list)."""
    def ev(views):
        policies = []
        values = []
        for v in views:
            legal = v.get_legal_moves_flat()
            p = 1.0 / (len(legal) + 1)
            policy = {a: p for a in legal}
            policy[pass_action] = p
            policies.append(policy)
            values.append(0.0)
        return policies, values
    return ev


def make_evaluator_cpp_shape(pass_action: int):
    """alpha_go_cpp batched evaluator: returns list[(policy, value)]."""
    def ev(views):
        out = []
        for v in views:
            legal = v.get_legal_moves_flat()
            p = 1.0 / (len(legal) + 1)
            policy = {a: p for a in legal}
            policy[pass_action] = p
            out.append((policy, 0.0))
        return out
    return ev


def run_one(label: str, ev_factory) -> dict:
    cfg = ao.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_weight = 0.0
    cfg.lambda_ = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = 100

    board = ao.GoBoard(SIZE, KOMI)
    tree = ao.MCTSTree(board, cfg, seed=0xC0FFEE)
    tree.run_simulations_batched(NUM_SIMS, BATCH_SIZE,
                                  ev_factory(ao.PASS_ACTION))

    vc = tree.get_child_visit_counts()
    root_visits = tree.get_root_visit_count()
    return {
        "label": label,
        "tree_size": tree.tree_size(),
        "root_visits": root_visits,
        "n_children": len(vc),
        "visit_total": sum(vc.values()),
        "visits": tuple(sorted(vc.items())),
    }


def main() -> int:
    print("=== 7km: batched-evaluator return-shape compatibility ===")
    print(f"config: {NUM_SIMS} sims, batch={BATCH_SIZE}, seed=0xC0FFEE, "
          f"deterministic uniform-policy eval")
    print()

    odin_res = run_one("odin-shape (policies, values)",
                       make_evaluator_odin_shape)
    cpp_res  = run_one("cpp-shape  list[(p, v)]    ",
                       make_evaluator_cpp_shape)

    for r in (odin_res, cpp_res):
        print(f"  {r['label']}  tree_size={r['tree_size']}  "
              f"root_visits={r['root_visits']}  n_children={r['n_children']}  "
              f"sum(child_visits)={r['visit_total']}")

    # Both runs use the same seed + same evaluator semantics, so the tree
    # state must match bit-for-bit. Differences here mean the adapter
    # introduced a subtle re-ordering or different value mapping.
    fails = []
    for key in ("tree_size", "root_visits", "n_children", "visit_total"):
        if odin_res[key] != cpp_res[key]:
            fails.append(f"{key}: odin={odin_res[key]} cpp={cpp_res[key]}")
    if odin_res["visits"] != cpp_res["visits"]:
        fails.append("per-child visits differ")

    if fails:
        print("\n=== FAIL ===")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\n=== PASS: both shapes produce identical tree state ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
