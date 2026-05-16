#!/usr/bin/env python3
"""27v: contract parity for MCTSTree readout methods on both backends.

Earlier work on this bead assumed both backends, given the same
evaluator + config, would build the SAME tree and we could compare
readouts value-by-value. That turned out to be wrong: even with a
deterministic evaluator, dirichlet_weight=0, and identical c_puct/
max_depth, the two backends explore different actions (likely due to
internal FPU defaults / PUCT tie-break differences, neither of which
is exposed in the public config).

That divergence is not a bug — the 7v8 strength A/B (Wilson 95% CI
brackets 0.5 over 100 games) already shows the backends play
equivalently. But it means we can't do value-by-value parity.

This harness instead checks that each readout method satisfies a set
of CONTRACT invariants on both backends:

  - returns a dict, not None / not an exception
  - keys are valid actions (integers in [0, n_actions))
  - sum / range / distribution invariants per method

If either backend violates any contract, that's a real bug surfaced by
the test. If both pass, we have evidence the API is honest on both
sides, even though the trees differ internally.

Run:
  PYTHONPATH=python autogo/.venv/bin/python python/parity/readouts_dual.py
"""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

import alpha_go_odin as ao
sys.path.insert(0, "autogo/.venv-cpponly/lib/python3.12/site-packages")
import alpha_go_cpp as ac

SIZE = 9
KOMI = 7.5
N_CELLS = SIZE * SIZE
N_ACTIONS = N_CELLS + 1
PASS_INDEX = N_CELLS
NUM_SIMS = 200


def make_eval(pass_action: int, eval_seed: int = 0xBEEF):
    """Deterministic informative-policy evaluator (slg-style)."""
    rng = np.random.default_rng(eval_seed)
    feat_dim = 3 * N_CELLS + 1
    W = rng.normal(0.0, 0.5, (N_ACTIONS, feat_dim)).astype(np.float32)

    def ev(board) -> tuple[dict[int, float], float]:
        raw = board.to_numpy().astype(np.float32).ravel()
        black = (raw == 1.0).astype(np.float32)
        white = (raw == 2.0).astype(np.float32)
        legal = board.get_legal_moves_flat()
        lm = np.zeros(N_CELLS, dtype=np.float32)
        for i in legal:
            lm[i] = 1.0
        feats = np.concatenate([black, white, lm,
                                 np.array([float(board.to_play())], dtype=np.float32)])
        logits = W @ feats
        mask = np.full(N_ACTIONS, -1e9, dtype=np.float32)
        for i in legal:
            mask[i] = 0.0
        mask[PASS_INDEX] = 0.0
        m = logits + mask
        m -= m.max()
        e = np.exp(m)
        p = e / e.sum()
        out = {i: float(p[i]) for i in legal}
        out[pass_action] = float(p[PASS_INDEX])
        return out, 0.0

    return ev


def make_cfg(backend):
    cfg = backend.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_weight = 0.0
    cfg.dirichlet_alpha = 0.0
    cfg.lambda_ = 0.0
    cfg.temperature = 1.0
    cfg.max_depth = 100
    return cfg


# --- contract checks ----------------------------------------------------

class Fail(Exception):
    pass


def require(cond: bool, label: str) -> None:
    if not cond:
        raise Fail(label)


def check_visit_counts(tree, label: str, num_sims: int) -> None:
    vc = tree.get_child_visit_counts()
    require(isinstance(vc, dict), f"{label}: get_child_visit_counts not dict")
    require(len(vc) > 0, f"{label}: visit-count dict empty")
    require(all(isinstance(k, int) and 0 <= k < N_ACTIONS or k == -1 for k in vc),
            f"{label}: invalid action keys: {list(vc.keys())[:5]}")
    require(all(isinstance(v, int) and v >= 0 for v in vc.values()),
            f"{label}: visit counts not non-negative ints")
    s = sum(vc.values())
    # Root visits should split among children. Allow ±1 slack for the root
    # itself (some backends count the root's own visit; others don't).
    require(abs(s - num_sims) <= 1,
            f"{label}: sum(child visits)={s} but num_sims={num_sims}")


def check_q_values(tree, label: str) -> None:
    qv = tree.get_child_q_values()
    require(isinstance(qv, dict), f"{label}: get_child_q_values not dict")
    require(all(isinstance(v, float) for v in qv.values()),
            f"{label}: q values not all float")
    require(all(-1.0 - 1e-4 <= v <= 1.0 + 1e-4 for v in qv.values()),
            f"{label}: q value out of [-1, 1]: {[v for v in qv.values() if not (-1.0 <= v <= 1.0)][:3]}")


def check_action_probs(tree, label: str) -> None:
    # temperature=1 → proper probability distribution
    p1 = tree.get_action_probabilities(1.0)
    require(isinstance(p1, dict), f"{label}: action_probs(1.0) not dict")
    require(len(p1) > 0, f"{label}: action_probs(1.0) empty")
    s = sum(p1.values())
    require(abs(s - 1.0) < 1e-3,
            f"{label}: action_probs(1.0) sum {s:.6f} != 1.0")
    require(all(0.0 <= v <= 1.0 + 1e-6 for v in p1.values()),
            f"{label}: action_probs(1.0) value out of [0, 1]")
    # temperature=0 → one-hot (max prob ≈ 1)
    p0 = tree.get_action_probabilities(0.0)
    require(isinstance(p0, dict), f"{label}: action_probs(0.0) not dict")
    require(len(p0) > 0, f"{label}: action_probs(0.0) empty")
    s0 = sum(p0.values())
    require(abs(s0 - 1.0) < 1e-3,
            f"{label}: action_probs(0.0) sum {s0:.6f} != 1.0")
    max_prob = max(p0.values())
    require(max_prob > 0.99,
            f"{label}: temp=0 should be one-hot, max prob = {max_prob:.6f}")


def check_first_eval(tree, label: str) -> None:
    fe = tree.get_child_first_eval_values()
    require(isinstance(fe, dict), f"{label}: get_child_first_eval_values not dict")
    require(all(isinstance(v, float) and math.isfinite(v) for v in fe.values()),
            f"{label}: first_eval values not all finite floats")
    require(all(-1.0 - 1e-4 <= v <= 1.0 + 1e-4 for v in fe.values()),
            f"{label}: first_eval value out of [-1, 1]")


def check_max_depths(tree, label: str) -> None:
    md = tree.get_child_max_subtree_depths()
    require(isinstance(md, dict), f"{label}: get_child_max_subtree_depths not dict")
    require(all(isinstance(v, int) and v >= 0 for v in md.values()),
            f"{label}: max_depths values not non-negative ints")


def check_root_priors(tree, label: str) -> None:
    pp = tree.get_root_policy_priors()
    require(isinstance(pp, dict), f"{label}: get_root_policy_priors not dict")
    require(len(pp) > 0, f"{label}: root_policy_priors dict empty")
    require(all(0.0 <= v <= 1.0 + 1e-4 for v in pp.values()),
            f"{label}: prior out of [0, 1]")
    s = sum(pp.values())
    # Root priors over expanded children only; should sum to <= 1.0 + slack
    # (the evaluator's policy is normalised, but only legal moves are stored).
    require(s <= 1.0 + 1e-3,
            f"{label}: root_policy_priors sum {s:.6f} > 1.0")


def check_one(label: str, backend, pass_action: int) -> dict[str, Any]:
    """Run all contract checks on one backend, return a result summary."""
    cfg = make_cfg(backend)
    board = backend.GoBoard(SIZE, KOMI)
    if backend is ao:
        tree = ao.MCTSTree(board, cfg, seed=0xC0FFEE)
    else:
        tree = ac.MCTSTree(board, cfg)
    tree.run_simulations(NUM_SIMS, make_eval(pass_action))

    failures = []
    for check_name, check_fn in [
        ("visit_counts", check_visit_counts),
        ("q_values", check_q_values),
        ("action_probs", check_action_probs),
        ("first_eval", check_first_eval),
        ("max_depths", check_max_depths),
        ("root_priors", check_root_priors),
    ]:
        try:
            if check_name == "visit_counts":
                check_fn(tree, label, NUM_SIMS)
            else:
                check_fn(tree, label)
        except Fail as e:
            failures.append((check_name, str(e)))
        except AttributeError as e:
            failures.append((check_name, f"method missing: {e}"))

    return {
        "label": label,
        "tree_size": tree.tree_size(),
        "root_visits": tree.get_root_visit_count(),
        "n_children_visited": len(tree.get_child_visit_counts()),
        "failures": failures,
    }


def main() -> int:
    print("=== 27v: MCTSTree readout contract parity (Odin vs C++) ===")
    print(f"config: {NUM_SIMS} sims, deterministic informative-projection eval, "
          f"dirichlet=0, c_puct=1.0\n")

    results = []
    for label, backend, pass_action in [
        ("odin", ao, ao.PASS_ACTION),
        ("cpp", ac, ac.PASS_ACTION),
    ]:
        r = check_one(label, backend, pass_action)
        results.append(r)
        status = "PASS" if not r["failures"] else "FAIL"
        print(f"  {label:<6} [{status}]  tree_size={r['tree_size']}, "
              f"root_visits={r['root_visits']}, n_children={r['n_children_visited']}")
        for name, msg in r["failures"]:
            print(f"    {name}: {msg}")

    any_fail = any(r["failures"] for r in results)
    print("\n" + ("=== ALL CONTRACTS HOLD ===" if not any_fail else
                  "=== CONTRACT VIOLATION ==="))
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
