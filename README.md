# autogodin

Odin port of the C++ MCTS + Go-board core from [ericjang/autogo](https://github.com/ericjang/autogo), plus a workspace for experiments and research around it.

## Layout

- `autogo/` — upstream clone (Python + C++ + experiments)
- `odin/alpha_go/` — Odin port (scaffold; real port lands under epic `autogodin-3xv`)
- `odin/tests/` — Odin unit tests
- `scripts/build_odin.sh` — build the Odin shared library to `build/libalpha_go_odin.so`
- `.beads/` — issue tracker; run `bd ready` for the current work queue

## Build

```bash
./scripts/build_odin.sh
# -> build/libalpha_go_odin.so with C-ABI exports (alphago_odin_*)
```

Override optimization with `ODIN_OPT="-o:speed -no-bounds-check" ./scripts/build_odin.sh`.

## Test

```bash
odin test odin/tests
```

## Workflow

Tasks live in `bd` (beads). See `AGENTS.md` for compute hosts, GPU policy, and parallel-agent conventions. Start with `bd ready`.
