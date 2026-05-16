# MCTS Optimization Research Notes

Scope: single-thread, 9x9 Go, CPU-only, deterministic uniform-policy evaluator. Current gap is C++ pybind11 = 8.5k sims/s vs Odin = 3.9k sims/s (ydh.2). Foundation pieces 1 and 2 are landed (hoisted board tables, per-tree `virtual.Arena`, zero-copy FFI view). Piece 3 (move-delta + do/undo on a working_board) is in progress. This document picks the next moves with the highest engineering ROI for that gap, not a survey of MCTS.

## Executive Summary

The current node is the bottleneck. Each `MCTSNode` carries a cloned `GoBoard` plus two `map`s (`children`, `logP_A`) - that is several allocations and ~hundreds of bytes per node, with all selection iteration going through a hash map. The C++ baseline does not pay any of that. The five things we should consider next, in priority order:

1. **Finish piece 3 (do/undo on a per-tree working_board).** This is the single biggest unforced error and the C++ baseline already does it. Expected: closes most of the gap on its own.
2. **Replace per-node `map[int]int` / `map[int]f32` children with two parallel `[]struct{...}` slices** sized at expansion time. Linear scans over 5-30 children are faster than Odin map hashing every PUCT iteration, and they make SIMD / branchless argmax trivial later. Medium-high ROI.
3. **Pack the per-child PUCT state (move, N, Q, P) into a hot struct kept separate from the rest of the node** (an AoSoA-ish split). Cache lines touched per descent become 1-2 instead of N. Medium ROI.
4. **Cache `sqrt(N_parent)` once at the top of each `select_child` call**, and store priors as `logP` (we already do this). Low-cost, high-confidence win.
5. **Swap `map[u64]struct{}` for `seen_hashes` with a small open-addressed `[]u64` set** (linear probe). Go games have <500 plies; a 1024-entry power-of-two table with linear probing beats a generic map on every metric. Low-effort, low-risk.

Things 1-3 directly attack the data layout that distinguishes us from the C++ baseline. Items 4-5 are cleanup that has shown up consistently in the engineering literature.

Everything else surveyed below (Gumbel AlphaZero, sequential halving, KataGo training-side tricks, graph search, virtual loss / batching) is either an algorithmic improvement orthogonal to the sims/s gap, or only pays off with a real NN evaluator. They are noted for completeness, not for the next sprint.

---

## 1. Memory + Data Layout for MCTS Trees

### Move-delta replay vs lazy snapshots vs full-state nodes

**Sources.**
- Chessprogramming Wiki, *Copy-Make* and *Make/Unmake* pages: <https://www.chessprogramming.org/Copy-Make>
- OpenChess forum thread "Copy Board vs Unmake Move": <https://open-chess.org/viewtopic.php?t=665>
- Hacker News discussion of copy/make vs move/undo: <https://news.ycombinator.com/item?id=6993330>

**What the literature says.** The chess community's empirical consensus is: copy-make wins only when the state is tiny (e.g. a packed 64-byte bitboard struct). For Go-style state (board grid + chain table + liberty counts + ko + history hash + seen-set), the per-move *delta* is 1-3 stones touched plus a small list of captures, and undo is bounded. Crafty-style engines copy entire mailbox boards and still get 30M copy/makes/sec only because their state fits in a few cache lines; ours does not. KataGo and Leela both maintain a single working board and replay moves down each descent.

**Why it matters for us.** Cloning a `GoBoard` per node (current piece-2 code) means: a `Point` array clone, a chain-id array clone, a liberty-set clone, an `seen_hashes` map clone, plus map allocations for `children` / `logP_A`. On 9x9 with 1600 sims, that is ~1600 clones per move - the C++ baseline does zero. Piece 3 is exactly the right call.

**ROI for our project. HIGH.** This is the file-the-bug-and-fix-it issue.

### SoA vs AoS for MCTSNode arrays

**Sources.**
- *AoS and SoA* (Algorithmica HPC book): <https://en.algorithmica.org/hpc/cpu-cache/aos-soa/>
- Wikipedia, *AoS and SoA*: <https://en.wikipedia.org/wiki/AoS_and_SoA>
- Lovekesh Azad, "SoA vs AoS Deep Dive": <https://medium.com/@azad217/structure-of-arrays-soa-vs-array-of-structures-aos-in-c-a-deep-dive-into-cache-optimized-13847588232e>

**What the literature says.** The Algorithmica writeup is blunt: AoS is the default; SoA only wins when you sweep one or two fields across many elements and skip the rest. For irregular pointer-chase patterns (which is exactly what MCTS descent looks like - node -> selected child -> selected child -> ...), AoS is up to 3x faster because each node access pulls one cache line that has everything you need. The 3-4x SoA speedups quoted in graphics / SIMD literature assume long sequential sweeps over one column.

**Where this leaves MCTS.** The descent loop touches one node at a time and reads several of its fields (N, children pointer, sum of child Ns, board state). That is the AoS workload. SoA only pays inside the *per-node* PUCT loop, which sweeps the children's (N, Q, P) - and there the right answer is a small AoS hot-struct *per child*, not a tree-wide SoA split.

**ROI for our project. LOW for tree-wide SoA. MEDIUM for child-local hot struct.** Don't restructure the tree. Do restructure the per-node children into something like `children_hot: []ChildEdge` where `ChildEdge :: struct { move: i16; N: i32; Q: f32; logP: f32; child_idx: i32 }` (24 bytes, well under a cache line for the typical 5-15 expanded children).

### Per-tree arenas vs slab allocators vs freelist pools

**Sources.**
- Bill Hall (gingerBill), *Memory Allocation Strategies, Part 4 (Pool Allocators)*: <https://www.gingerbill.org/article/2019/02/16/memory-allocation-strategies-004/>
- MaxGCoding, *Cache Friendly Linked Data Structures*: <https://www.maxgcoding.com/cache-friendly-linked-data-structures-node-pools-and-free-lists>
- Molecular Musings, *Memory allocation strategies: a pool allocator*: <https://blog.molecular-matters.com/2012/09/17/memory-allocation-strategies-a-pool-allocator/>
- Zig stdlib `ArenaAllocator` discussion: <https://news.ycombinator.com/item?id=40690220>

**What the literature says.** Arena (bump) allocation is unbeatable when (a) allocations are heterogeneous in size and (b) the lifetime is "free everything at once". Pool / slab allocators win when allocations are fixed-size and you need to *free individual items* during lifetime (e.g. reusing nodes from a discarded subtree). Freelists give O(1) recycle, but require fixed cell size.

**For our shape.** Per-tree arena (current code) is the right default. The one case where a pool would help: if/when we add tree-reuse-across-moves (keep the subtree under the chosen move, drop the rest), an arena can't reclaim the discarded subtree mid-search. Two options when that lands: (a) per-move arenas and copy the kept subtree, or (b) a fixed-size MCTSNode pool with a freelist. The arena flush itself is essentially free - it just resets the bump pointer and unmaps pages.

**ROI for our project. Current arena is fine. LOW priority for change.** Revisit only when implementing search-tree reuse across moves.

### Compact node encodings

**Sources.**
- KataGo source discussion (release notes for v1.9): <https://github.com/lightvector/KataGo/releases/tag/v1.9.0>
- Leela Zero issue tracker, generally on node packing.

**What practitioners do.** KataGo uses 32-bit visit counts and 32-bit floats per child; visit counts in the tens of thousands for 9x9 fit easily in u16, but KataGo deliberately keeps headroom for endgame analysis. For 1600-sim runs, **u16 visits and f32 Q is plenty** - Q values rarely need f64, and pinning the struct under one cache line is the win. Packing Q and N into a single 64-bit word is cute but adds masking on every read; not worth it.

**ROI for our project. LOW-MEDIUM.** Sizing fields once we've done the layout work is worth 5-10% from cache pressure, no more. Don't pack into a u64.

---

## 2. PUCT Selection Hot-Loop Optimizations

### Branchless / SIMD UCB

**Sources.**
- Encyclopedia of MCTS, Winands: <https://dke.maastrichtuniversity.nl/m.winands/documents/Encyclopedia_MCTS.pdf>
- A recent (Aug 2025) writeup on array-based, branchless MCTS reported up to 2.8x wall-clock speedup over pointer-based trees by flattening children into pre-allocated arrays and turning argmax into masked SIMD reductions. (Surfaced via search; primary source is recent arXiv and not yet canonicalised, treat as directional rather than gospel.)
- KataGo PUCT description in `KataGoMethods.md`: <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>

**What's actually portable.** The arithmetic of PUCT - `Q + c * P * sqrt(N_parent) / (1 + N_child)` - is two multiplies, one divide-or-reciprocal, one add per child. With children in a flat slice of (Q, P, N) you can run an 8-wide AVX2 reduction. Odin has `core:simd` and `intrinsics`; this is straightforward once the data layout is right.

**The order of operations matters.** First get children into a flat `[]ChildEdge`. *Then* worry about SIMD. Today the inner loop walks an Odin map - that overhead dwarfs the arithmetic.

**ROI for our project. MEDIUM after layout fix; HIGH if your branch predictor is mispredicting on max-update.** Use a scalar branchless max first (`m = m_n > m_o ? m_n : m_o`-style ternaries that compile to `cmov`). SIMD is a follow-on; on 9x9 mid-game the expanded child count is usually 10-30, which is one or two AVX2 vectors at best.

### Sorting children by visit count for early-exit

**Sources.**
- Discussion in Leela Zero issue tracker on FPU and policy ordering: <https://github.com/leela-zero/leela-zero/issues/696>
- Chessprogramming wiki, *Move Ordering* (the general principle): <https://www.chessprogramming.org/Move_Ordering>

**What's claimed.** If children are sorted by descending P (prior), and FPU is set so unvisited children have a stable predictable score, you can sometimes early-exit a PUCT scan once you know the remaining unvisited children can't beat the current best. In practice this saves cycles only when the child count is very large (Go 19x19 with ~200 moves). On 9x9 with 5-30 expanded children, the inner loop is short enough that branchless argmax beats a sorted-with-early-exit scan.

**ROI for our project. LOW.** Not worth the bookkeeping at 9x9 child counts.

### Caching `sqrt(N_parent)` per descent

**Sources.**
- Standard practice in lc0 and KataGo; documented as PUCT formula context in lc0 primer: <https://lczero.org/dev/lc0/search/alphazero/>

**What it is.** `N_parent` does not change inside one PUCT child-scan, so its sqrt should be hoisted out of the child loop. The combined exploration factor `c_puct * sqrt(N_parent)` collapses to one multiply per child.

**ROI for our project. HIGH (cheap and certain).** This is a 30-second fix if not already done; check the current select_child once piece 3 lands.

### Lazy / log priors

**Sources.**
- KataGo and lc0 both store priors as plain f32 P, not logP. The training pipeline outputs logits but stores normalized P in the tree.

**Verdict.** logP is useful when you want to combine priors with FPU or temperature without re-exponentiating; for vanilla PUCT, plain P is fine and we're already there. The current `logP_A` naming is misleading if the values are actually P (worth a check), but storing logP adds an exp on the hot path.

**ROI for our project. LOW.** Verify naming; don't switch.

---

## 3. Leaf Evaluation + FFI Cost Amortization

### Single-leaf vs leaf-parallel + virtual loss

**Sources.**
- Leela Chess Zero, *Gathering larger batches*: <https://lczero.org/dev/old/lc2/batching/>
- Leela Zero issue, *MCTS - virtual loss*: <https://github.com/leela-zero/leela-zero/issues/631>
- Oracle Developers, *Lessons from AlphaZero, Part 5*: <https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e>
- KataGo source, virtual loss semantics in `search.cpp`.

**What the literature says.** Virtual loss exists for GPU batching. lc0 reports batch sizes around 1200-1800 *with all the virtual-loss tricks turned up*, and notes that beyond ~32 you start paying meaningful policy quality cost (you visit nodes you wouldn't have visited if you'd seen earlier results). Below batch ~8 the GPU isn't saturated. The Oracle writeup quotes ~95% GPU utilization with virtual loss + parallel games.

**For our project, which has a Python *callback* not a GPU.** Batching only helps if the Python side is the bottleneck *and* the per-call Python overhead amortizes well over batch size. The current ydh.2 benchmark uses a deterministic uniform-policy evaluator, so there is no NN cost and effectively no batching ROI - the gap is pure tree-side. When a real NN evaluator goes in, batching at the FFI boundary will matter; published numbers suggest batch=8 captures most of the GPU win (~3-5x throughput vs batch=1) and batch=32 saturates (~6-8x). Going larger hurts strength.

**ROI for our project. LOW for the current benchmark; HIGH later when a real evaluator lands.** Don't add virtual loss now. The infrastructure for it (the `N_virt` field is already in `MCTSNode`) is ready when needed.

### Pure-Python callback overhead

**Sources.**
- ctypes / cffi cost discussions in general numpy / pybind11 literature.
- pybind11 docs on call-overhead: numbers in the few-hundred-nanoseconds-per-call range for trivial conversions.

**What to do when Python is the bottleneck.** The standard tricks: (a) batch the FFI boundary so Python pays one call per N leaves, not N calls; (b) keep the FFI payload as a contiguous byte buffer (we already have zero-copy view, good); (c) consider releasing the GIL on the Odin side and letting Python prepare the next batch concurrently. lc0 essentially does (a) + (c) for GPU inference.

**ROI for our project. Out of scope for the current gap (uniform evaluator).** Note it for when the NN evaluator integrates.

---

## 4. Position Hashing + Superko Bookkeeping

### Zobrist incremental vs recompute

**Sources.**
- Wikipedia *Zobrist hashing*: <https://en.wikipedia.org/wiki/Zobrist_hashing>
- Chessprogramming wiki *Zobrist Hashing*: <https://www.chessprogramming.org/Zobrist_Hashing>
- GNU Go Board Library docs: <http://www.gnu.org/software/gnugo/gnugo_15.html>

**What's standard.** Incremental XOR is two ops per stone change. Recompute is N ops for an N-cell board. Common pitfalls: (a) forgetting to XOR the side-to-move key, (b) not XOR'ing ko-state into the hash (two boards differ only in ko legality and you'll mis-transpose), (c) reusing the same random table across processes if you serialize hashes.

**Where we are.** Piece 1 already moved the Zobrist table out of `GoBoard` into a shared `BoardTables`. Confirm in piece 3 that do/undo updates the hash incrementally rather than recomputing on each replay - if it recomputes, that's a hidden 81-XOR per descent step.

**ROI for our project. HIGH if recompute is happening; otherwise N/A.** Verify.

### Bloom / cuckoo filters for seen_hashes

**Sources.**
- Cuckoo Filter paper, Fan et al., CoNEXT'14: <https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf>
- *Cuckoo filter* Wikipedia: <https://en.wikipedia.org/wiki/Cuckoo_filter>

**What's claimed.** Cuckoo filters beat Bloom on space/false-positive tradeoff and support deletion (which a generic Bloom does not). Both have nonzero false-positive rates.

**Why this is a bad idea for superko.** A superko false positive means rejecting a *legal* move as illegal, which silently distorts MCTS by pruning real game tree. Even a 1e-4 FPR at 1600 sims/move means roughly 0.16 false illegalizations per move - small but nonzero, and it biases search in ways you can't audit. Superko correctness matters more than the speed of the set.

**The right move is a faster *exact* set.** Open-addressed hash table with linear probing keyed on u64 is ~3-5x faster than Odin's generic `map[u64]struct{}` for sets that are sized in the low hundreds. Power-of-two capacity, mask instead of mod, no tombstones needed if entries only get added during a game.

**ROI for our project. MEDIUM.** Cheap, low-risk replacement. Skip the probabilistic filters.

### Compact representation of seen_hashes

**Same as above.** A `[]u64` open-addressed set, 1024 slots, will fit one or two cache lines for the average lookup and is far cheaper than `map[u64]struct{}` which goes through the runtime map machinery on every probe.

**ROI: MEDIUM.**

---

## 5. Go-Specific Micro-Optimizations

### Bitboard chain tracking + incremental liberty counts

**Sources.**
- KataGo `Board.cpp` and DeepWiki summary: <https://deepwiki.com/lightvector/KataGo>
- GNU Go Board Library, section 15: <http://www.gnu.org/software/gnugo/gnugo_15.html>
- Wikipedia *Bitboard*: <https://en.wikipedia.org/wiki/Bitboard>

**What KataGo / GNU Go do.** Maintain per-chain: stones, libertyCount, head (canonical stone). Liberty counts are *incrementally maintained* on every move: place a stone -> update adjacent chains' liberties, merge chains if same-color neighbors, capture if liberty -> 0. The naive alternative (flood-fill the chain on every legality check) is the single biggest cost in a slow Go engine.

**For 9x9 specifically.** 81 points fits in a `[2]u64` bitboard per color. A union-find with path compression over 81 cells is trivial. Even without bitboards, the right structure is parent[81] for union-find + libCount[chain_id] + a small "stones in chain" linked list using a `next[81]` array. Allocation-free.

**Where we are.** I haven't profiled this codepath, but if the current `GoBoard` does per-move flood-fill for capture detection, that's the second-biggest unforced error after node cloning. Verify before optimizing elsewhere.

**ROI for our project. HIGH if currently flood-filling; LOW if incremental.** Worth a 10-minute audit of `go_game.odin`.

### Pre-computed neighbor tables

**Already done** (BoardTables, piece 1). One additional easy win: precompute `is_on_board[81][4]` or pad the board to 11x11 with sentinel cells so neighbor iteration becomes branchless. KataGo uses sentinel padding; GNU Go uses a 1D coordinate scheme with explicit bounds checks.

**ROI: LOW-MEDIUM**, only relevant if the inner neighbor loop shows up in profiles after pieces 3 + the chain rewrite.

---

## 6. Things Deliberately Excluded

- **Multi-threaded MCTS, virtual loss, lock-free trees.** We are single-thread by design and the benchmark is single-thread. Adding parallelism trades complexity for sims/s only after the serial kernel is tight.
- **Graph search / transposition tables** (KataGo `GraphSearch.md`: <https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md>). Helps with deep search where transpositions multiply; 9x9 with 1600 sims is too shallow to benefit, and the bookkeeping (128-bit hash, edge-vs-node visits, Q staleness across parents) is non-trivial. Revisit if/when sim counts go up an order of magnitude.
- **Gumbel AlphaZero / Sequential Halving root selection** (Danihelka et al., ICLR 2022: <https://openreview.net/forum?id=bERaNdoegnO>). These improve *policy quality at low sim counts* with a learned policy. Our benchmark uses a uniform evaluator and measures throughput, not playing strength. Worth revisiting once a real NN evaluator is in.
- **Policy target pruning, playout cap randomization, forced playouts** (KataGo, Wu 2019: <https://arxiv.org/abs/1902.10565>). Training-side improvements; don't move sims/s.
- **MuZero-style learned dynamics models.** Out of scope - we have the real Go rules and want them fast, not a learned approximation.
- **TensorRT INT8 / GPU inference plumbing** (Oracle writeup: <https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e>). Relevant for when we plug in a real network. Today's benchmark uses a uniform evaluator.
- **Multi-step / Speedy MCTS variants.** Algorithmic refinements; orthogonal to the per-sim cost gap we're chasing.
- **AoSoA / SIMD argmax over children.** Listed in section 2 but explicitly deprioritized: the win is small at 9x9 child counts and requires the basic flat-children layout to land first.

---

## Summary Table

| # | Change | Section | ROI | Notes |
|---|--------|---------|-----|-------|
| 1 | Finish piece 3 (do/undo on working_board) | 1 | HIGH | Already in progress; biggest single win. |
| 2 | Replace per-node child maps with flat `[]ChildEdge` | 1, 2 | HIGH | Enables every other inner-loop optimization. |
| 3 | Audit `seen_hashes` / Zobrist for recompute vs incremental | 4 | HIGH if recompute | 10-min audit. |
| 4 | Audit chain/liberty tracking for flood-fill | 5 | HIGH if flood-fill | 10-min audit. |
| 5 | Hoist `sqrt(N_parent)` and `c * sqrt(N_parent)` out of child loop | 2 | HIGH | Trivial, certain. |
| 6 | Open-addressed `[]u64` set replacing `map[u64]struct{}` for seen | 4 | MEDIUM | Low risk. |
| 7 | Pack hot child fields under one cache line | 1, 2 | MEDIUM | After #2. |
| 8 | Branchless argmax, then SIMD | 2 | MEDIUM | After #2; scalar cmov first. |
| 9 | Visit counts as u16, Q as f32, no further packing | 1 | LOW-MEDIUM | Cache pressure only. |
| 10 | Virtual loss + batched evaluator | 3 | DEFER | When NN evaluator lands. |
| 11 | Graph search / transposition table | 6 | DEFER | When sim counts ~10x current. |
| 12 | Gumbel / sequential halving root | 6 | DEFER | Strength, not throughput. |
