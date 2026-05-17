# Refreshing the vendored mcts-odin

`odin/vendor/mcts-odin/` is a flat copy of mcts-odin's `mcts/` package at a
pinned commit. No git submodule. The header in
[`odin/vendor/mcts-odin/VERSION`](../odin/vendor/mcts-odin/VERSION) records
which commit we're on. This document is the procedure for moving that
pin to a newer mcts-odin release without silently regressing autogodin.

The design rationale for vendoring (vs depending on a published package
or a submodule) lives in
[`design-mcts-odin-integration.md`](design-mcts-odin-integration.md).

## Golden rule

**Never edit anything under `odin/vendor/mcts-odin/`.** That tree is a
mirror; the next refresh will overwrite it. MCTS algorithm changes belong
upstream in [`mcts-odin`](https://github.com/phiat/mcts-odin); bug fixes
filed in autogodin against vendored code should turn into bd issues in
mcts-odin's tracker (see `AGENTS.md`).

The only files you should edit in `odin/vendor/mcts-odin/` are:

- `VERSION` — bump the `commit:` and `synced:` lines; append a one-line
  headline of what changed (the same line you'll use in the autogodin
  commit message).

## Procedure

### 1. Confirm the upstream HEAD

```bash
cd ../mcts-odin
git fetch
git log -1 --format='%h %s'        # what commit you'd be moving to
git tag --sort=-creatordate | head # latest release tags
```

Prefer pinning at a tagged release (`v0.X.Y`) over an arbitrary commit so
the bump message can name a release. If you're tracking unreleased work
because of a fix you need, note that in `VERSION`.

### 2. Inventory the diff

```bash
cd ../mcts-odin
# Files we currently mirror:
ls mcts/

# What's changed since the pinned commit (replace OLD_COMMIT):
git diff --stat OLD_COMMIT..HEAD -- mcts/
```

Things to watch for that imply a breaking change to autogodin:

- New or removed files in `mcts/` — adjust the cp glob and re-verify the
  Odin build picks them all up.
- Signature changes to **`Config`**, **`Game`** (vtable), **`Evaluator`**,
  or **`Tree`** struct layout — autogodin's `odin/alpha_go/exports.odin`
  and `python/alpha_go_odin/__init__.py` reach into these directly.
- Signature changes to `run_simulations`, `run_simulations_batched`,
  `run_simulations_threaded`, `select_action`, `reuse_root` — same.
- New required `Game` vtable methods — `go_adapter.odin` would need to
  implement them.

If any of those apply, expect to touch `go_adapter.odin` and/or
`exports.odin` as part of the refresh. **Plan the autogodin-side edits
before copying files** so the tree never has half-applied state.

### 3. Copy the files

```bash
# From autogodin/ root:
cp ../mcts-odin/mcts/*.odin odin/vendor/mcts-odin/mcts/
cp ../mcts-odin/LICENSE     odin/vendor/mcts-odin/LICENSE
```

Then update `odin/vendor/mcts-odin/VERSION` with the new `commit:`,
`synced:` (today's date), and a one-line headline of what's new.

If new files appeared under `mcts-odin/mcts/`, the wildcard above will
pick them up. If files were *removed*, delete the local copies by hand —
the cp won't clean them. `git status` will surface the orphans.

### 4. Smoke gate

```bash
just check       # build + 37 Odin tests + Zobrist parity fingerprint
```

`just check` is the minimum bar. It catches:

- API drift that breaks `go_adapter.odin` or `exports.odin` (build fail).
- Behaviour regressions visible to the 37 `@(test)` cases.
- Zobrist parity regressions vs the committed fixture
  (`python/parity/fixtures/random_games_v0.json`) — this is the
  byte-identical SHA-256 fingerprint against upstream `alpha_go_cpp`,
  so a divergence here means either we or upstream's port drifted.

If autogo's `.venv` is set up (`scripts/setup_autogo.sh` did its job),
also run the contract-parity checks:

```bash
just parity-readouts    # MCTSTree readout-method contract vs C++
just parity-batched     # batched-evaluator return-shape compat (7km)
```

These catch silent API-shape drift that the Odin-side tests can't see
(e.g. evaluator return shape, `MCTSTree.get_*` methods).

### 5. Throughput sanity (`just bench`)

```bash
just bench odin 3   # 3 trials, 1600 sims × 32 moves, uniform evaluator
```

Compare sims/sec against the last value in the most recent
`experiments/*cpp-vs-odin-rebench*/report.md`. **Acceptable drift: ±10%
on the same host.** A drop bigger than that signals either a real
regression in the upstream perf work or a config-tuning change you missed
(e.g. mcts-odin v0.7.0's batched dirichlet rewrite needed
`use_tree_rng = false` semantics threaded through; v0.4.0 changed the
sample_packed_action signature).

If the drift is large or the workload mix changed materially upstream
(new RNG, new memory layout, etc.), spend the ~5 minutes to write a
dated `cpp-vs-odin-rebench-vX.Y.Z` experiment dir and link the new
number from README's "Throughput" pointer. The README explicitly defers
to the per-experiment report for current numbers — this is the
mechanism.

### 6. Config-schema A/B (when needed)

If `Config` grew or shrunk fields in this refresh, our defaults may now
behave differently. Re-run a quick `slg`-style 20-game A/B before
declaring the refresh done:

```bash
# Adapt the slg recipe (autogodin-slg / experiments/2026-05-16_14-27-*)
# to the current commit; 2 trials × 20 games is enough to catch a
# config-default regression at p<0.1.
```

For the `19l` refresh this wasn't needed — `Config` was unchanged. For
`ci2.14`-class refreshes that touch defaults (FPU, batched, threaded)
it is.

### 7. Commit + push

Two commits is cleaner than one:

```bash
git add odin/vendor/mcts-odin/
git commit -m "vendor: bump mcts-odin <old>->><new>  (<headline change>)"
```

If you had to touch `go_adapter.odin` / `exports.odin` /
`python/alpha_go_odin/__init__.py` to match an API change, that goes in
its own commit so the vendor bump is reviewable in isolation:

```bash
git add odin/alpha_go/ python/alpha_go_odin/
git commit -m "<bd-id>: adapt to mcts-odin <new> API change (<what>)"
```

Push, then update the bd issue tracking the refresh (e.g. `bd close
autogodin-19l --reason="vendor bump + smoke gates clean"`).

## What a "good" refresh looks like

Reference: `19l` (v0.4.0 → v0.7.0, commit
[`d0fac1d`](https://github.com/phiat/autogodin/commit/d0fac1d)). Steps in
that refresh:

1. Verified upstream HEAD at v0.7.0 tag (commit `2517513`).
2. Diffed 8 changed files in `mcts/`: `batched.odin`, `mcts.odin`,
   `playout.odin`, `readout.odin`, `rng.odin` (new ~90 LOC),
   `threaded.odin`, `version.odin`, `game.odin`.
3. No `Config` / vtable / `Tree` struct changes → no `go_adapter.odin`
   edits needed. Confirmed before copying.
4. cp'd files, bumped `VERSION` with the v0.7.0 headline (Xoshiro256++
   RNG + batched dirichlet rewrite).
5. `just check` passed (build, 37 tests, Zobrist parity fingerprint).
6. `just parity-readouts` and `just parity-batched` passed.
7. Throughput re-measurement explicitly deferred — the v0.7.0 perf wins
   are in tree-walk-heavy regimes where our shimmed Python evaluator
   isn't the bottleneck. Filed as future-work pointer in the bd close
   reason.
8. Single commit, message named the headline change.

The whole refresh fit in one work session. The smoke gate caught nothing
because nothing was broken — that's the desired outcome, not evidence
the gate is unnecessary.

## When NOT to refresh

- **A release is mid-flight.** Wait until the autogodin bench you're
  running finishes before swapping the MCTS layer underneath it.
- **No tagged release exists since the last refresh.** Upstream's
  trunk between tags is fair game for experimental commits; pin at
  tags unless you specifically need an untagged fix.
- **A `Config` schema change is in flight upstream.** Refresh after
  the change settles, not in the middle.
