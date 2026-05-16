# Contributing

Thanks for your interest in autogodin. The repo is small and the surface
area is well-defined; the workflow below is what we use day-to-day.

## Repo layout (where to put what)

| Path                          | What lives there                                 |
|-------------------------------|--------------------------------------------------|
| `odin/alpha_go/`              | GoBoard, Game-vtable adapter, C-ABI exports      |
| `odin/vendor/mcts-odin/`      | Pinned copy of the generic MCTS package (do not edit; bumps come from upstream) |
| `odin/tests/`                 | Odin `@(test)` procs — board + adapter integration |
| `python/alpha_go_odin/`       | Python ctypes wrapper                            |
| `python/odin_backend/`        | Drop-in shim that makes `import alpha_go_cpp` resolve to alpha_go_odin |
| `python/parity/`              | Zobrist-fingerprint parity harness               |
| `scripts/`                    | `build_odin.sh`, `run_with_odin_backend.sh`, `setup_autogo.sh` |
| `experiments/<date>-<slug>/`  | Self-contained experiments (bench + report.md)   |

## MCTS lives in a sibling repo

The generic MCTS package lives at [phiat/mcts-odin](https://github.com/phiat/mcts-odin).
This repo vendors it as a pinned copy under `odin/vendor/mcts-odin/`.

**Bug fixes / new features in the MCTS algorithm itself go in mcts-odin,
not here.** Game-side or FFI changes go here. If you're unsure where a
change belongs: anything that touches a `mcts.*` symbol is generic and
likely belongs upstream.

## Build + test gates

You need [Odin](https://odin-lang.org/) nightly, a C compiler, and
Python 3.10+. We use [just](https://github.com/casey/just) as the
runner, but the underlying scripts work directly.

```bash
just build      # builds build/libalpha_go_odin.so
just test       # runs odin/tests (37 cases, ~5 sec)
just parity     # Zobrist-fingerprint parity check vs the committed fixture
just check      # the pre-push gate: build + test + parity
```

**Run `just check` before pushing.** The parity gate is load-bearing —
the SHA-256 fingerprint in `python/parity/random_games.py` is committed
and must not drift. If a board-rule change is intentional, the
fingerprint is regenerated and the change is called out explicitly in
the PR.

## Commit / PR style

Look at `git log --oneline -10` for the pattern in use:

```
<bead-id-or-slug>: <verb> + <one-line summary>

<paragraph: what changed and why; numbers if relevant>

Closes <bead-id>.
```

PRs:
- One topic per PR. If a change incidentally improves three other
  things, split.
- Include reproduce command + machine for any perf claim. We measure
  on a 16-core CPU host (codename: miniwini) — *which* host the
  numbers come from matters.
- The optional internal task tracker is [beads](https://github.com/gastownhall/beads).
  External contributors don't need it; existing collaborators
  reference bead IDs in commits.

## Performance work

We track throughput on a single 9×9 micro-bench (1600 sims/move × 32
moves). Sequential and batched grids are committed under `experiments/`.
For a perf change, include:

- before / after numbers from the same host
- a `±95% CI` (3+ trials, `mean ± 1.96·σ/√n`)
- which experiment dir the numbers came from

We prefer fewer high-signal benches over many low-signal ones.

## What we will not accept

- New algorithm work in the vendored MCTS path (`odin/vendor/mcts-odin/`).
  File it upstream.
- Backend-specific perf hacks that break board parity. The fingerprint
  is the gate.
- Force-pushes to `main` (use a feature branch).

## Questions?

Open an issue. The repo is small enough that any reasonable question
gets a reply.
