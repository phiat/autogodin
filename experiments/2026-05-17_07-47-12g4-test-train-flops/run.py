#!/usr/bin/env python3
"""12g.4: test-time vs train-time compute trade-off.

Trains 3 model sizes (small/mid/big) on a shared dataset, then plays a
gauntlet of each (model, sims) cell against a reference panel at 3
matched-FLOP budgets. See design.md.

Two reference checkpoints are uploaded from local: bpoC iter4_best.pt
(carry-forward champion) and iter0_best.pt (random-init baseline);
random agent is registered in-process.

Mirrors the bpoC/ydh.1 runner pattern: idempotent (skips completed
phases), self-contained, runs on a JL L4. Bootstrap is the same
scripts/jl_bootstrap.sh.

Sanity dry-run (no GPU):
    python run.py --dry-run

Real run on JL (after bootstrap):
    PYTHONPATH="$PWD/python/odin_backend:$PWD/python:autogo/src" \\
      GAME_DATA_DIR=$HOME/nfs-local/game_data_root \\
      ALPHAGO_BACKEND=odin \\
      autogo/.venv/bin/python experiments/.../run.py
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

# Reuse the flops calculator co-located with this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flops import CANDIDATES, BUDGETS, matched_pairs, forward_flops, fmt_flops

EXP = Path(__file__).resolve().parent
EXP_NAME = EXP.name
WORKSPACE = EXP.parent.parent
AUTOGO = WORKSPACE / "autogo"
PY = AUTOGO / ".venv" / "bin" / "python"
NFS = Path(os.environ.get("NFS_ROOT", "/nfs"))

# Reference panel: trained checkpoints we bring along.
REF_PANEL = [
    # name, local_path, eval_sims
    ("ref-bpoC-iter0", WORKSPACE / "experiments/2026-05-17_07-40-bpoC-rerun-postfix/checkpoints/iter0_best.pt", 200),
    ("ref-bpoC-iter4", WORKSPACE / "experiments/2026-05-17_07-40-bpoC-rerun-postfix/checkpoints/iter4_best.pt", 200),
]


def step(title: str) -> None:
    print()
    print("=" * 70)
    print(f"=== {title}")
    print("=" * 70, flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None,
        log: Path | None = None, dry_run: bool = False) -> float:
    print(f"$ {' '.join(cmd)}", flush=True)
    if dry_run:
        print("  [dry-run] skipped", flush=True)
        return 0.0
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
            print("--- last 80 lines of log ---")
            print(log.read_text()[-6000:])
        raise SystemExit(r.returncode)
    return dt


def make_train_script(channels: int, n_blocks: int, batch: int = 512) -> Path:
    """Copy upstream train.py and patch the model-size constants."""
    train_src = (AUTOGO / "experiments" / "2026-04-26_22-32-train-fromscratch"
                 / "train.py")
    dst = EXP / f"train_{channels}ch_{n_blocks}b.py"
    if dst.exists():
        return dst
    shutil.copy(train_src, dst)
    text = dst.read_text()
    patches = [
        ("MODEL_CHANNELS = 128", f"MODEL_CHANNELS = {channels}"),
        ("MODEL_N_BLOCKS = 10", f"MODEL_N_BLOCKS = {n_blocks}"),
        ("BATCH_SIZE = 128", f"BATCH_SIZE = {batch}"),
        ('MODEL_NAME = "SizeInvariantGoResNet-128ch-10b"',
         f'MODEL_NAME = "SizeInvariantGoResNet-{channels}ch-{n_blocks}b"'),
    ]
    new_text = text
    for old, new in patches:
        if old != new and old not in new_text:
            raise RuntimeError(f"train.py upstream patch missed: {old!r}")
        new_text = new_text.replace(old, new)
    dst.write_text(new_text)
    return dst


def make_gauntlet_agent_script(name: str, ckpt: Path, sims: int, model_tag: str) -> Path:
    """Generate a one-off agent-registration script for the gauntlet."""
    out = EXP / f"_agent_{name}.py"
    out.write_text(
        f"""# auto-generated; do not edit.
from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator
from alpha_go.agents.base import register_agent

@register_agent("{name}")
class _Agent(CppMCTSAgent):
    def __init__(self):
        ev = LeafBatchedNNEvaluator("{ckpt}", 9, "{model_tag}")
        super().__init__(
            evaluator=ev, num_simulations={sims},
            c_puct=1.0, temperature=0.0,  # argmax for eval
            add_noise=False, leaf_batch_size=64,
        )
"""
    )
    return out


def head_to_head(black: str, white: str, num_games: int, save_name: str,
                  agent_imports: list[str], dry_run: bool = False,
                  log: Path | None = None) -> float:
    """Run num_games of black vs white via self_play.main."""
    runner = EXP / f"_match_{save_name.replace('/', '_')}.py"
    imports = "\n".join(f"import {m}" for m in agent_imports)
    runner.write_text(
        f"""# auto-generated; do not edit.
import sys
{imports}
from alpha_go.self_play import main as sp_main
sys.argv = [
    "self_play",
    "--black", "{black}",
    "--white", "{white}",
    "--board_size", "9",
    "--num_games", "{num_games}",
    "--num_workers", "4",
    "--save-name", "{save_name}",
    "--seed", "12345",
    "--batched-inference",
]
sp_main()
"""
    )
    return run([str(PY), str(runner)],
               env={"PYTHONPATH": f"{WORKSPACE}/python/odin_backend:{WORKSPACE}/python:{AUTOGO}/src",
                    "ALPHAGO_BACKEND": "odin"},
               log=log, dry_run=dry_run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-games", type=int, default=5000,
                   help="Random pre-collect for shared training dataset.")
    p.add_argument("--time-budget", type=int, default=900,
                   help="Seconds per model training. Default 900 (matches bpoC).")
    p.add_argument("--games-per-pair", type=int, default=20,
                   help="N games per (candidate, reference) gauntlet pair, each side.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands and FLOPs table; don't execute training/eval.")
    args = p.parse_args()

    # --- 1. Print the matched-FLOPs plan (always) --------------------------
    step("Matched-FLOPs plan")
    print(f"{'budget':>10} {'config':>20} {'sims':>6} {'actual':>10}")
    cells = []  # (budget_name, spec, sims) tuples to evaluate
    for budget_name, target in BUDGETS:
        for spec, sims, total in matched_pairs(target):
            cells.append((budget_name, spec, sims))
            print(f"{budget_name:>10} {spec.name:>20} {sims:>6} {fmt_flops(int(total)):>10}")
    print(f"\n{len(cells)} candidate cells × {len(REF_PANEL)+1} references "
          f"× {args.games_per_pair*2} games = "
          f"{len(cells)*(len(REF_PANEL)+1)*args.games_per_pair*2} eval games")

    if args.dry_run:
        print("\n[dry-run] not launching any compute.")
        return

    # --- 2. Sanity-check we're on the right kind of host -------------------
    assert PY.exists(), f"venv python missing: {PY}. Did jl_bootstrap.sh run?"
    assert NFS.exists(), f"{NFS} missing — bootstrap should symlink /nfs"
    for name, ckpt, _ in REF_PANEL:
        assert ckpt.exists(), f"missing reference checkpoint: {ckpt}"

    logs_dir = EXP / "logs"
    logs_dir.mkdir(exist_ok=True)
    ckpt_dir = NFS / "checkpoints" / EXP_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    common_pp = f"{WORKSPACE}/python:{AUTOGO}/src"
    odin_pp = f"{WORKSPACE}/python/odin_backend:{common_pp}"

    timings: dict[str, float] = {}

    # --- 3. Pre-collect shared training data -------------------------------
    random_dir = (Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root"))
                  / f"experiments/{EXP_NAME}/random-it0")
    if random_dir.exists() and len(list(random_dir.rglob("*.npz"))) >= args.random_games:
        print(f"[bootstrap] random-it0 present, skipping.")
    else:
        step(f"pre_collect {args.random_games} random×random games")
        random_save = f"experiments/{EXP_NAME}/random-it0"
        timings["pre_collect"] = run(
            [str(PY), "-m", "alpha_go.self_play",
             "--black", "random", "--white", "random",
             "--board_size", "9",
             "--num_games", str(args.random_games),
             "--num_workers", "8",
             "--save-name", random_save, "--seed", "0"],
            env={"PYTHONPATH": common_pp},
            log=logs_dir / "00_random.log",
        )

    ds_txt = EXP / "dataset.txt"
    ds_txt.write_text(f"experiments/{EXP_NAME}/random-it0\n")

    # --- 4. Train 3 model sizes on the shared dataset ----------------------
    trained_ckpts: dict[str, Path] = {}  # spec.name -> ckpt
    for spec in CANDIDATES:
        ckpt = ckpt_dir / f"{spec.name}.pt"
        trained_ckpts[spec.name] = ckpt
        if ckpt.exists():
            print(f"[train:{spec.name}] checkpoint present, skipping.")
            continue
        step(f"train {spec.name} ({spec.channels}ch x {spec.n_blocks}b) "
             f"on shared dataset")
        train_py = make_train_script(spec.channels, spec.n_blocks)
        timings[f"train_{spec.name}"] = run(
            [str(PY), str(train_py),
             "--dataset-txt", str(ds_txt),
             "--iteration", "0",
             "--resume-from", "",
             "--time-budget", str(args.time_budget)],
            env={"PYTHONPATH": common_pp},
            log=logs_dir / f"01_train_{spec.name}.log",
        )
        # train.py writes to /nfs/checkpoints/<EXP_NAME>/iter0_best.pt;
        # we move it to a stable per-spec name.
        produced = ckpt_dir / "iter0_best.pt"
        if produced.exists() and produced != ckpt:
            shutil.move(str(produced), str(ckpt))
        assert ckpt.exists(), f"missing {ckpt}"

    # --- 5. Stage reference checkpoints on /nfs ----------------------------
    nfs_ref: dict[str, Path] = {}
    for name, local_ckpt, _ in REF_PANEL:
        dst = ckpt_dir / f"{name}.pt"
        if not dst.exists():
            shutil.copy(str(local_ckpt), str(dst))
        nfs_ref[name] = dst

    # --- 6. Gauntlet matches -----------------------------------------------
    # For each (budget, spec, sims) candidate, play vs each reference
    # (and vs `random`). games_per_pair as black + games_per_pair as white.
    results = []
    for budget_name, target in BUDGETS:
        for spec, sims, total_flops in matched_pairs(target):
            cand_name = f"{budget_name}-{spec.name}"
            cand_script = make_gauntlet_agent_script(
                cand_name, trained_ckpts[spec.name], sims,
                model_tag=f"{spec.channels}x{spec.n_blocks}")
            for ref_name, _, ref_sims in REF_PANEL:
                ref_script = make_gauntlet_agent_script(
                    ref_name, nfs_ref[ref_name], ref_sims, model_tag="256x10")
                # candidate as black, reference as white
                save = f"experiments/{EXP_NAME}/gauntlet/{cand_name}-vs-{ref_name}/asB"
                step(f"gauntlet {cand_name} (B) vs {ref_name} (W) — "
                     f"{args.games_per_pair} games")
                dt = head_to_head(
                    cand_name, ref_name, args.games_per_pair, save,
                    [f"_agent_{cand_name}", f"_agent_{ref_name}"],
                    log=logs_dir / f"02_g_{cand_name}-vs-{ref_name}_asB.log")
                results.append((cand_name, ref_name, "B", args.games_per_pair, dt))
                # reference as black, candidate as white
                save = f"experiments/{EXP_NAME}/gauntlet/{cand_name}-vs-{ref_name}/asW"
                step(f"gauntlet {ref_name} (B) vs {cand_name} (W) — "
                     f"{args.games_per_pair} games")
                dt = head_to_head(
                    ref_name, cand_name, args.games_per_pair, save,
                    [f"_agent_{ref_name}", f"_agent_{cand_name}"],
                    log=logs_dir / f"02_g_{cand_name}-vs-{ref_name}_asW.log")
                results.append((cand_name, ref_name, "W", args.games_per_pair, dt))
            # vs random
            for color in ("B", "W"):
                save = f"experiments/{EXP_NAME}/gauntlet/{cand_name}-vs-random/as{color}"
                step(f"gauntlet vs random ({color}) — {args.games_per_pair} games")
                if color == "B":
                    black, white = cand_name, "random"
                else:
                    black, white = "random", cand_name
                dt = head_to_head(
                    black, white, args.games_per_pair, save,
                    [f"_agent_{cand_name}"],
                    log=logs_dir / f"02_g_{cand_name}-vs-random_as{color}.log")
                results.append((cand_name, "random", color, args.games_per_pair, dt))

    # --- 7. Summary --------------------------------------------------------
    summary = {
        "exp_name": EXP_NAME,
        "config": {
            "random_games": args.random_games,
            "time_budget_s": args.time_budget,
            "games_per_pair": args.games_per_pair,
            "candidates": [s.name for s in CANDIDATES],
            "budgets": [b for b, _ in BUDGETS],
            "reference_panel": [n for n, _, _ in REF_PANEL] + ["random"],
        },
        "timings_seconds": timings,
        "match_results_meta": [
            {"cand": c, "ref": r, "color": col, "games": g, "wall_s": dt}
            for (c, r, col, g, dt) in results
        ],
    }
    (EXP / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 70)
    print("DONE. Summary written to summary.json")
    print("Next: parse the per-match NPZs under "
          f"/nfs/game_data_root/experiments/{EXP_NAME}/gauntlet/ "
          "into a winrate matrix + ELO scoring.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
