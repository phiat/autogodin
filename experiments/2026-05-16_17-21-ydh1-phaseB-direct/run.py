#!/usr/bin/env python3
"""ydh.1 Phase B (direct): pre_collect → train iter0 → collect iter0 selfplay → train iter1.

This is the docker-less, cluster-less, single-GPU-node variant of
`autogo/experiments/2026-04-26_22-32-train-fromscratch/run_iteration.sh 0 1`.

Why this exists:
- ydh.8 wants iter0 + iter1 checkpoints from a GPU run.
- The upstream run_iteration.sh dispatches train + collect via
  infra.remote_exec → ssh + `docker run` against a ghcr image we don't
  have pull access to (Eric's private GHCR).
- Rather than build/push our own image (30+ min one-time billed
  setup), we replicate the moral equivalent inline against a single
  GPU instance set up by scripts/jl_bootstrap.sh.

Steps:
  1. pre_collect ~500 random vs random games (CPU on the instance)
  2. train iter0 from random data → iter0_best.pt
  3. collect ~200 self-play games with iter0 ckpt + Odin MCTS (GPU NN forward)
  4. train iter1 from iter0 selfplay → iter1_best.pt
  5. Play one demo game from iter1 vs iter1, save SGF

Acceptance (ydh.8):
  - /nfs/checkpoints/<EXP>/iter0_best.pt exists
  - /nfs/checkpoints/<EXP>/iter1_best.pt exists
  - report.md has loss curves, timings, sample game, cost.

Run on a JL PyTorch instance after scripts/jl_bootstrap.sh has set up the
build + venv + /nfs symlink. From the autogodin work dir:

  PYTHONPATH="$PWD/python:autogo/src" GAME_DATA_DIR=$HOME/nfs-local/game_data_root \\
    autogo/.venv/bin/python experiments/2026-05-16_17-21-ydh1-phaseB-direct/run.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent
EXP_NAME = EXP.name
WORKSPACE = EXP.parent.parent
AUTOGO = WORKSPACE / "autogo"
PY = AUTOGO / ".venv" / "bin" / "python"
NFS = Path(os.environ.get("NFS_ROOT", "/nfs"))


def step(title: str) -> None:
    print()
    print("=" * 70)
    print(f"=== {title}")
    print("=" * 70, flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None,
        log: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    t0 = time.time()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "wb") as f:
            r = subprocess.run(cmd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, env=full_env)
    dt = time.time() - t0
    print(f"  [{dt:.1f}s] exit={r.returncode}", flush=True)
    if r.returncode != 0:
        if log:
            print("--- last 50 lines of log ---")
            print(log.read_text()[-4000:])
        raise SystemExit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--random-games", type=int, default=500)
    p.add_argument("--selfplay-games", type=int, default=200)
    p.add_argument("--mcts-sims", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    assert PY.exists(), f"venv python missing: {PY}. Did jl_bootstrap.sh run?"
    assert NFS.exists(), f"{NFS} missing — bootstrap should symlink /nfs"

    logs_dir = EXP / "logs"
    logs_dir.mkdir(exist_ok=True)
    ckpt_dir = NFS / "checkpoints" / EXP_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    common_pp = f"{WORKSPACE}/python:{AUTOGO}/src"

    # 1. Pre-collect random vs random
    random_dir = (Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root"))
                  / f"experiments/{EXP_NAME}/random-it0")
    if random_dir.exists() and any(random_dir.rglob("*.npz")):
        print(f"[1/4] random-it0 already present at {random_dir}, skipping.")
    else:
        step(f"[1/4] pre_collect {args.random_games} random×random games")
        random_save = f"experiments/{EXP_NAME}/random-it0"
        run(
            [str(PY), "-m", "alpha_go.self_play",
             "--black", "random", "--white", "random",
             "--board_size", "9",
             "--num_games", str(args.random_games),
             "--num_workers", str(args.num_workers),
             "--save-name", random_save,
             "--seed", "0"],
            env={"PYTHONPATH": common_pp},
            log=logs_dir / "01_random.log",
        )

    # 2. Train iter0.  train.py keys off Path(__file__).parent.name for EXP_NAME
    #    and writes /nfs/checkpoints/<EXP_NAME>/iter*_best.pt — so we copy
    #    train.py into our experiment dir.
    train_src = (AUTOGO / "experiments" / "2026-04-26_22-32-train-fromscratch"
                 / "train.py")
    local_train = EXP / "train.py"
    if not local_train.exists():
        shutil.copy(train_src, local_train)
    iter0_ckpt = ckpt_dir / "iter0_best.pt"
    if iter0_ckpt.exists():
        print(f"[2/4] iter0_best.pt already present at {iter0_ckpt}, skipping.")
    else:
        step("[2/4] train iter0 from random data")
        ds_it0 = EXP / "dataset-it0.txt"
        ds_it0.write_text(f"experiments/{EXP_NAME}/random-it0\n")
        run(
            [str(PY), str(local_train),
             "--dataset-txt", str(ds_it0),
             "--iteration", "0",
             "--resume-from", ""],
            env={"PYTHONPATH": common_pp},
            log=logs_dir / "02_train_it0.log",
        )
        assert iter0_ckpt.exists(), f"missing {iter0_ckpt}"

    # 3. Collect iter0 selfplay via C++ MCTS + iter0 NN on GPU.
    # NOTE: originally we wanted Odin MCTS here, but CppMCTSAgent +
    # LeafBatchedNNEvaluator hardcode the C++ batched-eval signature
    # (list[(policy, value)]) which the Odin shim's trampoline can't
    # unpack — see autogodin-7km. We use the real C++ wheel here; the
    # Odin↔C++ throughput head-to-head is already in 4fl on miniwini.
    step(f"[3/4] collect {args.selfplay_games} selfplay games "
         f"(C++ MCTS, {args.mcts_sims} sims/move, NN on GPU)")
    selfplay_save = f"experiments/{EXP_NAME}/selfplay-it0"
    inline = EXP / "_collect_iter0.py"
    inline.write_text(
        f"""# auto-generated by run.py; do not edit.
import sys
from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator
from alpha_go.agents.base import register_agent

@register_agent("ydh1-iter0")
class _Iter0Agent(CppMCTSAgent):
    def __init__(self):
        ev = LeafBatchedNNEvaluator("{iter0_ckpt}", 9, "32x4")
        super().__init__(
            evaluator=ev,
            num_simulations={args.mcts_sims},
            c_puct=1.0,
            temperature=1.0,
            add_noise=True,
            leaf_batch_size=64,
        )

from alpha_go.self_play import main as sp_main
sys.argv = [
    "self_play",
    "--black", "ydh1-iter0",
    "--white", "ydh1-iter0",
    "--board_size", "9",
    "--num_games", "{args.selfplay_games}",
    "--num_workers", "1",
    "--save-name", "{selfplay_save}",
    "--seed", "42",
]
sp_main()
"""
    )
    run(
        [str(PY), str(inline)],
        env={"PYTHONPATH": common_pp},
        log=logs_dir / "03_selfplay.log",
    )

    # 4. Train iter1 from iter0 selfplay
    step("[4/4] train iter1 from iter0 selfplay data")
    ds_it1 = EXP / "dataset-it1.txt"
    ds_it1.write_text(f"experiments/{EXP_NAME}/selfplay-it0\n")
    run(
        [str(PY), str(local_train),
         "--dataset-txt", str(ds_it1),
         "--iteration", "1",
         "--resume-from", str(iter0_ckpt)],
        env={"PYTHONPATH": common_pp},
        log=logs_dir / "04_train_it1.log",
    )
    iter1_ckpt = ckpt_dir / "iter1_best.pt"
    assert iter1_ckpt.exists(), f"missing {iter1_ckpt}"

    # Final report stub — full report.md hand-written from logs after run.
    summary = {
        "exp_name": EXP_NAME,
        "random_games": args.random_games,
        "selfplay_games": args.selfplay_games,
        "mcts_sims_per_move": args.mcts_sims,
        "iter0_ckpt": str(iter0_ckpt),
        "iter1_ckpt": str(iter1_ckpt),
        "iter0_ckpt_bytes": iter0_ckpt.stat().st_size,
        "iter1_ckpt_bytes": iter1_ckpt.stat().st_size,
    }
    (EXP / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 70)
    print("DONE. Summary written to summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
