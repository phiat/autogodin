# ydh.6: top-3 hotspots in the Odin port

## Setup

- **Target**: in-process Odin bench (`experiments/2026-05-16_12-50-4ig-inprocess-bench/bench.odin`).
  Uniform-policy evaluator, 1600 sims × 32 moves × 5 trials, 9×9 Go. No Python
  in the loop — measures the pure Odin MCTS + GoBoard cost.
- **Build**: `odin build ... -o:speed -debug` (optimized + DWARF symbols).
- **Profiler**: `perf record -e task-clock -F 999 --call-graph dwarf`.
  Hardware counters aren't exposed under WSL2's Microsoft kernel, but
  software-clock sampling at 999Hz is fine for proportional CPU breakdown.
- **Samples**: 5,850 over a ~6s run. Statistically tight.
- **Artifacts** (in this dir): `bench.perf.data` (full perf data, `perf
  report -i ...` it for interactive drill-down), `perf-flat.txt` (flat
  self-time ranking), `perf-callgraph.txt` (callgraph-annotated).
- **No SVG flamegraph** — FlameGraph clone was denied by the sandbox.
  `perf.data` is preserved; render later with `stackcollapse-perf.pl +
  flamegraph.pl` if needed.

## Top of perf flat ranking (self-time)

```
13.86%  alpha_go::get_group_and_liberties
12.31%  runtime::mem_alloc_bytes
 8.62%  __memmove_avx_unaligned_erms          (libc)
 6.60%  runtime::default_temp_allocator_proc
 6.32%  mcts::perform_playout
 5.50%  runtime::_append_elem
 5.13%  alpha_go::play_flat_unchecked
 5.09%  runtime::arena_alloc
 4.21%  alpha_go::is_legal_flat
 3.81%  __$map_set$$map[u64]struct{}
 3.15%  runtime::mem_free_with_size
 3.03%  runtime::map_kvh_data_dynamic
 2.51%  __$map_get$$map[u64]struct{}
```

Read by category:

| category                    | total |
|-----------------------------|-------|
| allocator + map runtime     | ~46%  |
| memmove/memset (mostly clone + map growth) | ~12% |
| `get_group_and_liberties` (group flood-fill) | ~14% |
| our own hot procs (`play_flat_unchecked`, `is_legal_flat`, adapter) | ~12% |
| MCTS algorithm (`perform_playout` self) | ~6% |

Almost half the CPU is allocator and map machinery. The MCTS algorithm
itself is a small slice — the Go-rules cost dominates, especially the
legality check.

## Hotspot 1 — `is_legal_flat`'s clone-and-simulate path

**Cost picture**: ~30-35% of total CPU. Source:

- `clone_for_sim` (1.23% self) does `slice.clone(b.board)` → memmove on the
  N-cell buffer → drives the ~8.6% `__memmove_avx_unaligned_erms`.
- The clone then calls `play_flat_unchecked`, which writes
  `b.seen_hashes[b.current_hash] = {}` (PSK record) → drives
  `map_set / map_insert_hash_dynamic / map_kvh_data_dynamic / mem_alloc_bytes /
  arena_alloc / default_temp_allocator_proc` — ~25-30% combined.
- The clone's `seen_hashes` map is **never read** afterwards (the PSK
  lookup in `is_legal_flat:349` uses `b.seen_hashes`, not `tmp.seen_hashes`).
  All the map traffic on the clone is pure overhead.

**Mitigation options** (in order of size):

a. **Eliminate the clone entirely.** Both downstream checks can be done
   without one:
   - *multi-stone suicide*: simulate capture in place using the existing
     `get_group_and_liberties` machinery (already walks the relevant
     neighbour groups). If our hypothetical-played group ends with zero
     liberties after captures, illegal.
   - *PSK*: compute the prospective post-move Zobrist hash incrementally —
     XOR in our stone's zobrist value, XOR out each captured stone's. Look
     up against `b.seen_hashes`. No clone of the map or the board needed.

b. **Cheaper fallback if (a) is too invasive**: in `clone_for_sim`, set
   `dst.seen_hashes = nil` (don't `make` a map). Then `play_flat_unchecked`
   needs a nil-guard on the `seen_hashes[current_hash] = {}` line. Kills
   the map runtime entirely on the clone path while preserving current
   structure.

**Estimated gain**: option (a) ~25-30% throughput; option (b) ~15-20%
throughput. Both can land.

## Hotspot 2 — `get_group_and_liberties` does too much work per call

**Cost picture**: 13.86% self-time. Called from two sites in
`is_legal_flat`: capture detection on the legality check (~half the calls)
and from `play_flat_unchecked` for actual captures (the other half).

**Why it's heavy**: the proc flood-fills the entire opponent group AND
collects every liberty. For the capture-detection caller in
`is_legal_flat`, we only check `len(libs) == 1 && libs[0] == index`. We
don't need the full liberty set — we just need to know:

- *Is the liberty count exactly 1?* — early-exit-able at 2 distinct
  liberties.
- *Is that single liberty `index`?* — first liberty found tells us.

**Mitigation**:

c. Add a specialized `is_single_liberty_at(b, opp_group_start, candidate_idx) -> bool`
   that flood-fills the group while counting distinct liberties; bails
   the moment we see a second distinct liberty. No dynamic arrays for
   `group`/`libs`; a stack-allocated `[n_cells]bool` visited bitset is enough.

**Estimated gain**: ~5-8% throughput. Smaller than hotspot 1 but
self-contained — easy win.

## Hotspot 3 — `get_legal_moves_flat` allocator pressure

**Cost picture**: ~5.5% in `runtime::_append_elem` + a slice of
`mem_alloc_bytes`/`arena_alloc`. Driven by the bench's `uniform_evaluator`
calling `get_legal_moves_flat(b, context.temp_allocator)` on every leaf —
a fresh `[dynamic]int` each time, growing by `append`.

This is a representative cost: any real NN evaluator will also need to
build a legal-action list per leaf (for the policy mask). Reducing it
helps both this bench and the NN-eval path.

**Mitigation**:

d. Provide a `fill_legal_moves_flat(b, out: ^[N+1]int) -> int` variant
   that writes into a caller-owned fixed buffer (max N+1 = `n_cells +
   pass`). Zero allocations per call. The adapter
   (`adapter_legal_actions`) already writes into a caller-owned
   `[dynamic]int` from mcts — extending the contract upward to the
   evaluator avoids the alloc layer.

**Estimated gain**: ~3-5% throughput.

## Combined headline

Landing all three: rough order-of-magnitude **+30-45% throughput** on the
in-process bench. From the current 25,618 sims/sec → ~33,000-37,000 sims/sec.
The C++ ratio would go from 2.96× to ~3.8-4.3×.

All three mitigations land in `odin/alpha_go/` only — no `mcts-odin`
vendor changes needed, no FFI changes.

## Follow-up beads to file

| candidate         | maps to        | priority |
|-------------------|----------------|----------|
| Skip clone-and-simulate in `is_legal_flat` | hotspot 1 | P2 (biggest win) |
| Early-exit liberty count in `is_legal_flat` capture probe | hotspot 2 | P3 |
| Caller-owned `legal_moves_flat` buffer | hotspot 3 | P3 |
