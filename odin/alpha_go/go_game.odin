package alpha_go

// Go rules engine. Pure CPU, no MCTS / NN — those live in
// odin/vendor/mcts-odin/ and the Python NN side respectively. This file is
// the leaf of the dependency tree: GoBoard + legality + Zobrist + scoring.
//
// AlphaZero reader's map:
//   - GoBoard is the env state (one position). `to_play` = whose turn.
//   - play_flat / play / pass_move = env.step. is_legal_flat is the action mask.
//   - score / get_winner = terminal reward signal (TT area + komi, Black - White).
//   - The MCTS algorithm itself is generic and lives in vendor/mcts-odin/; this
//     file is wired to it via go_adapter.odin (Game vtable).
//
// Notation: PSK = positional superko (Zobrist-incremental, see seen_hashes).
// TT = Tromp-Taylor (used for both legality "no suicide" rule and area scoring).

import "base:runtime"
import "core:slice"
import "core:sync"

EMPTY :: i8(0)
BLACK :: i8(1)
WHITE :: i8(2)

KOMI_DEFAULT :: f32(7.5)
NO_KO :: -1
PASS_ACTION :: -1

// Compile-time board-size hint. When > 0, hot paths use this constant
// instead of loading `b.size` and multiplying. Runtime board size still
// works — the hint just enables better codegen for the common case.
//
// Set via:  odin build ... -define:BOARD_SIZE_HINT=9
// Default 0 = fully runtime (no specialization).
BOARD_SIZE_HINT :: #config(BOARD_SIZE_HINT, 0)

// n_cells: number of cells on the board. When BOARD_SIZE_HINT > 0 this folds
// to a compile-time constant in the caller, eliminating the field load + IMUL.
@(private)
n_cells :: #force_inline proc "contextless" (b: ^GoBoard) -> int {
	when BOARD_SIZE_HINT > 0 {
		return BOARD_SIZE_HINT * BOARD_SIZE_HINT
	} else {
		return b.size * b.size
	}
}

// board_dim: linear board size. Same compile-time-fold pattern as n_cells.
@(private)
board_dim :: #force_inline proc "contextless" (b: ^GoBoard) -> int {
	when BOARD_SIZE_HINT > 0 {
		return BOARD_SIZE_HINT
	} else {
		return b.size
	}
}

Neighbors4 :: struct {
	indices: [4]int,
	count:   int,
}

// Per-size shared tables. neighbors + zobrist are a pure function of board size
// and never mutate, so every GoBoard of the same size points at the same instance.
// Built lazily on first request via get_board_tables(size); never freed during
// the process lifetime (singleton).
BoardTables :: struct {
	size:      int,
	neighbors: []Neighbors4,
	zobrist:   [][3]u64,
}

@(private)
_board_tables_cache: map[int]^BoardTables
@(private)
_board_tables_mu: sync.Mutex

get_board_tables :: proc(size: int) -> ^BoardTables {
	sync.mutex_lock(&_board_tables_mu)
	defer sync.mutex_unlock(&_board_tables_mu)
	if t, ok := _board_tables_cache[size]; ok {
		return t
	}
	// Singletons must outlive any per-test/per-tree allocator, so pin them to the
	// default heap regardless of what `context.allocator` is set to by the caller.
	context.allocator = runtime.default_allocator()
	if _board_tables_cache == nil {
		_board_tables_cache = make(map[int]^BoardTables)
	}
	t := new(BoardTables)
	t.size = size
	n := size * size
	t.neighbors = make([]Neighbors4, n)
	t.zobrist = make([][3]u64, n)
	init_neighbors_table(t)
	init_zobrist_table(t)
	_board_tables_cache[size] = t
	return t
}

// Singleton teardown — call from test runners or process shutdown if you want the
// memory tracker to report 0 live allocations. Idempotent.
release_board_tables_cache :: proc() {
	for _, t in _board_tables_cache {
		delete(t.neighbors)
		delete(t.zobrist)
		free(t)
	}
	delete(_board_tables_cache)
	_board_tables_cache = nil
}

// One Go position. Owns its row-major board buffer + a PSK history (seen_hashes)
// and points at a per-size shared neighbors+zobrist table (not owned).
GoBoard :: struct {
	size:               int,                 // Board dimension (9 for 9x9).
	komi:               f32,                 // White's compensation; added to white_score in `score`.
	board:              []i8,                // size*size cells in {EMPTY,BLACK,WHITE}. Row-major.
	to_play:            i8,                  // BLACK or WHITE; flips on every play / pass.
	ko_point:           int,                 // Flat index forbidden by simple ko; NO_KO (= -1) when none.
	consecutive_passes: int,                 // 2 consecutive passes → is_game_over.
	move_count:         int,                 // Half-moves played from the empty start (incl. passes).

	tables:       ^BoardTables,              // Shared per-size singleton (neighbors + zobrist). Not owned.
	current_hash: u64,                       // Incremental Zobrist hash; XOR'd on every stone placement/removal.
	seen_hashes:  map[u64]struct{},          // PSK history: hashes of all past positions; lazy-allocated.

	allocator:    runtime.Allocator,         // Allocator that owns `board` + `seen_hashes`. Used at destroy.
}

// Fresh empty board. `board` is zeroed (EMPTY everywhere), `seen_hashes` is left
// nil — the first play_flat_unchecked lazily allocates it. That keeps cloned
// boards used inside is_legal_flat's temp_allocator probes cheap (no map alloc
// until they actually play a move).
make_go_board :: proc(size: int = 9, komi: f32 = KOMI_DEFAULT, allocator := context.allocator) -> GoBoard {
	when BOARD_SIZE_HINT > 0 {
		assert(size == BOARD_SIZE_HINT,
			"BOARD_SIZE_HINT was set at compile time; runtime board size must match")
	}
	context.allocator = allocator
	n := size * size
	b := GoBoard {
		size      = size,
		komi      = komi,
		board     = make([]i8, n),
		to_play   = BLACK,
		ko_point  = NO_KO,
		tables    = get_board_tables(size),
		allocator = allocator,
	}
	return b
}

destroy_go_board :: proc(b: ^GoBoard) {
	delete(b.board, b.allocator)
	delete(b.seen_hashes)
	b^ = {}
}

// Clone with full PSK history. Caller owns all dst-allocated buffers.
clone_go_board :: proc(src: ^GoBoard, allocator := context.allocator) -> GoBoard {
	context.allocator = allocator
	dst := GoBoard {
		size               = src.size,
		komi               = src.komi,
		board              = slice.clone(src.board),
		to_play            = src.to_play,
		ko_point           = src.ko_point,
		consecutive_passes = src.consecutive_passes,
		move_count         = src.move_count,
		tables             = src.tables, // shared pointer; no clone needed
		current_hash       = src.current_hash,
		allocator          = allocator,
	}
	dst.seen_hashes = make(map[u64]struct{}, len(src.seen_hashes))
	for h in src.seen_hashes {
		dst.seen_hashes[h] = {}
	}
	return dst
}

// In-place legality probe used by is_legal_flat for the multi-stone-suicide
// and PSK checks. Replaces the older clone-and-simulate path that dominated
// ~30-35% of CPU through map_alloc/map_insert/memcpy traffic on the discarded
// clone (see ydh.6 perf profile). Mirrors play_flat_unchecked's capture
// ordering exactly — neighbour-by-neighbour, in 4-neighbour order — so the
// legality semantics are byte-identical to "play then check".
//
// Caller is responsible for the cheap rejects (bounds, empty, ko, fast
// neighbour-scan); reaching here means we actually need to know whether the
// resulting position would be a no-op suicide or a position repeat. The proc
// mutates b, then restores it before returning — the net effect on b is
// identity. seen_hashes is never touched (PSK is checked against the
// prospective hash, not the simulated state).
@(private = "file")
probe_legal_flat :: proc(b: ^GoBoard, index: int, check_suicide, check_psk: bool) -> bool {
	player := b.to_play
	opp := opponent_of(player)
	saved_hash := b.current_hash

	// Place our stone hypothetically.
	b.board[index] = player
	b.current_hash ~= b.tables.zobrist[index][int(player)]

	// Capture dead opponent groups. Same iteration shape as play_flat_unchecked:
	// for each opponent neighbour still on the board, compute group liberties;
	// if zero, remove all stones in that group and XOR their Zobrist contributions
	// out of current_hash. Track removed cells in `captured` so we can restore.
	captured := make([dynamic]int, 0, 8, context.temp_allocator)
	defer delete(captured)

	nb := b.tables.neighbors[index]
	for k in 0 ..< nb.count {
		ni := nb.indices[k]
		if b.board[ni] != opp {continue} // EMPTY after a prior capture -> skip
		group, libs := get_group_and_liberties(b, ni, context.temp_allocator)
		if len(libs) == 0 {
			for g in group {
				b.board[g] = EMPTY
				b.current_hash ~= b.tables.zobrist[g][int(opp)]
				append(&captured, g)
			}
		}
		delete(group)
		delete(libs)
	}

	legal := true

	if check_suicide {
		// Our group needs at least one liberty (counting capture-vacated cells).
		_, our_libs := get_group_and_liberties(b, index, context.temp_allocator)
		if len(our_libs) == 0 {legal = false}
		delete(our_libs)
	}

	if legal && check_psk {
		if _, ok := b.seen_hashes[b.current_hash]; ok {legal = false}
	}

	// Undo: restore captured opp stones, remove our stone, reset hash.
	for g in captured {
		b.board[g] = opp
	}
	b.board[index] = EMPTY
	b.current_hash = saved_hash

	return legal
}

@(private = "file")
init_neighbors_table :: proc(t: ^BoardTables) {
	size := t.size
	for row in 0 ..< size {
		for col in 0 ..< size {
			idx := row * size + col
			n := &t.neighbors[idx]
			n.count = 0
			if row > 0 {n.indices[n.count] = (row - 1) * size + col; n.count += 1}
			if row < size - 1 {n.indices[n.count] = (row + 1) * size + col; n.count += 1}
			if col > 0 {n.indices[n.count] = row * size + (col - 1); n.count += 1}
			if col < size - 1 {n.indices[n.count] = row * size + (col + 1); n.count += 1}
		}
	}
}

@(private = "file")
splitmix64 :: proc(seed: ^u64) -> u64 {
	seed^ += 0x9E3779B97F4A7C15
	z := seed^
	z = (z ~ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ~ (z >> 27)) * 0x94D049BB133111EB
	return z ~ (z >> 31)
}

@(private = "file")
init_zobrist_table :: proc(t: ^BoardTables) {
	seed := u64(0x9E3779B97F4A7C15) ~ u64(t.size)
	n := t.size * t.size
	for i in 0 ..< n {
		t.zobrist[i][EMPTY] = 0
		t.zobrist[i][BLACK] = splitmix64(&seed)
		t.zobrist[i][WHITE] = splitmix64(&seed)
	}
}

flat_index :: proc(b: ^GoBoard, row, col: int) -> int {
	return row * board_dim(b) + col
}

row_col :: proc(b: ^GoBoard, flat: int) -> (row, col: int) {
	dim := board_dim(b)
	return flat / dim, flat % dim
}

at :: proc(b: ^GoBoard, row, col: int) -> i8 {
	return b.board[row * board_dim(b) + col]
}

at_flat :: proc(b: ^GoBoard, index: int) -> i8 {
	return b.board[index]
}

is_game_over :: proc(b: ^GoBoard) -> bool {
	return b.consecutive_passes >= 2
}

@(private = "file")
opponent_of :: proc(c: i8) -> i8 {
	return WHITE if c == BLACK else BLACK
}

// DFS flood-fill: returns the connected group at `index` and its liberties.
// Both [dynamic] returns are allocated with the supplied allocator.
get_group_and_liberties :: proc(
	b: ^GoBoard,
	index: int,
	allocator := context.allocator,
) -> (
	group: [dynamic]int,
	liberties: [dynamic]int,
) {
	group = make([dynamic]int, 0, 16, allocator)
	liberties = make([dynamic]int, 0, 16, allocator)
	color := b.board[index]
	if color == EMPTY {
		return
	}
	n := n_cells(b)
	visited := make([]bool, n, context.temp_allocator)
	defer delete(visited, context.temp_allocator)
	lib_visited := make([]bool, n, context.temp_allocator)
	defer delete(lib_visited, context.temp_allocator)
	stack := make([dynamic]int, 0, 16, context.temp_allocator)
	defer delete(stack)

	append(&stack, index)
	visited[index] = true

	for len(stack) > 0 {
		current := pop(&stack)
		append(&group, current)
		nb := b.tables.neighbors[current]
		for k in 0 ..< nb.count {
			ni := nb.indices[k]
			v := b.board[ni]
			if v == EMPTY {
				if !lib_visited[ni] {
					lib_visited[ni] = true
					append(&liberties, ni)
				}
			} else if v == color && !visited[ni] {
				visited[ni] = true
				append(&stack, ni)
			}
		}
	}
	return
}

// Specialised liberty check for is_legal_flat's capture-detection neighbour
// scan: returns true iff the group at `start_cell` would be captured by
// playing at `candidate_lib_idx` — i.e. `candidate_lib_idx` is the group's
// sole liberty. Bails immediately on the first liberty != candidate_lib_idx
// (no need to count all liberties, no need to materialise the group cells).
// No [dynamic] allocation for the liberty/group return lists. See ydh.6
// hotspot #2.
@(private = "file")
would_capture_group_at :: proc(b: ^GoBoard, start_cell, candidate_lib_idx: int) -> bool {
	color := b.board[start_cell]
	if color == EMPTY {return false}

	n := n_cells(b)
	visited := make([]bool, n, context.temp_allocator)
	defer delete(visited, context.temp_allocator)
	stack := make([dynamic]int, 0, 16, context.temp_allocator)
	defer delete(stack)

	append(&stack, start_cell)
	visited[start_cell] = true
	saw_candidate := false

	for len(stack) > 0 {
		current := pop(&stack)
		nb := b.tables.neighbors[current]
		for k in 0 ..< nb.count {
			ni := nb.indices[k]
			v := b.board[ni]
			if v == EMPTY {
				if ni == candidate_lib_idx {
					saw_candidate = true
				} else {
					return false // a liberty other than candidate → not single-lib-at-candidate
				}
			} else if v == color && !visited[ni] {
				visited[ni] = true
				append(&stack, ni)
			}
		}
	}
	return saw_candidate
}

remove_group :: proc(b: ^GoBoard, group: []int) -> int {
	for idx in group {
		b.current_hash ~= b.tables.zobrist[idx][int(b.board[idx])]
		b.board[idx] = EMPTY
	}
	return len(group)
}

is_legal :: proc(b: ^GoBoard, row, col: int) -> bool {
	return is_legal_flat(b, row * board_dim(b) + col)
}

// Legality check. Cascade of cheap-to-expensive tests:
//   1. bounds + empty + simple-ko (O(1))
//   2. neighbor scan: any empty neighbor? friendly neighbor? capturing neighbor?
//      (O(neighbors), at most 4 + occasional flood fill on opponent groups)
//   3. only if neither empty nor friendly nor capturing → fast suicide reject
//   4. only if (1-3) all pass AND we'd be playing-into-no-liberty OR PSK is
//      tracked → clone-and-simulate to check multi-stone suicide + PSK
// Step 4 is the only expensive path; steps 1-3 reject ~all illegal moves on
// real boards. `context.temp_allocator` carries the clone — the FFI entry point
// (ffi_tree_run_simulations in exports.odin) free_all's it after each call.
is_legal_flat :: proc(b: ^GoBoard, index: int) -> bool {
	if index < 0 || index >= n_cells(b) {return false}
	if b.board[index] != EMPTY {return false}
	if b.ko_point == index {return false}

	player := b.to_play
	opponent := opponent_of(player)

	has_friendly := false
	has_empty := false
	captures := false

	nb := b.tables.neighbors[index]
	loop: for k in 0 ..< nb.count {
		ni := nb.indices[k]
		v := b.board[ni]
		if v == EMPTY {
			has_empty = true
			break loop
		} else if v == player {
			has_friendly = true
		} else if v == opponent && !captures {
			if would_capture_group_at(b, ni, index) {
				captures = true
			}
		}
	}

	// Single-stone-suicide fast reject.
	if !has_empty && !has_friendly && !captures {
		return false
	}

	need_suicide_check := !has_empty && !captures
	need_psk_check := len(b.seen_hashes) > 0
	if need_suicide_check || need_psk_check {
		return probe_legal_flat(b, index, need_suicide_check, need_psk_check)
	}
	return true
}

get_legal_moves_flat :: proc(b: ^GoBoard, allocator := context.allocator) -> [dynamic]int {
	n := n_cells(b)
	moves := make([dynamic]int, 0, n, allocator)
	for i in 0 ..< n {
		if is_legal_flat(b, i) {
			append(&moves, i)
		}
	}
	return moves
}

play :: proc(b: ^GoBoard, row, col: int) -> bool {
	return play_flat(b, row * board_dim(b) + col)
}

play_flat :: proc(b: ^GoBoard, index: int) -> bool {
	if !is_legal_flat(b, index) {return false}
	play_flat_unchecked(b, index)
	return true
}

// Applies a move assuming legality already checked. Dual-use: real plays go
// through play_flat (which calls this), and is_legal_flat calls it on a temp
// clone to detect multi-stone suicide. Side effects:
//   - flips the stone into place + XOR-updates current_hash
//   - records the PRE-move hash in seen_hashes (so future PSK checks see it)
//   - captures any zero-liberty opponent groups, undoing their Zobrist contributions
//   - sets ko_point only when exactly one single-stone capture happened
play_flat_unchecked :: proc(b: ^GoBoard, index: int) {
	// Record the pre-move state hash in seen_hashes (for PSK on future moves).
	b.seen_hashes[b.current_hash] = {}

	b.board[index] = b.to_play
	b.current_hash ~= b.tables.zobrist[index][int(b.to_play)]
	opp := opponent_of(b.to_play)
	b.ko_point = NO_KO

	total_captured := 0
	last_captured := -1

	nb := b.tables.neighbors[index]
	for k in 0 ..< nb.count {
		ni := nb.indices[k]
		if b.board[ni] == opp {
			group, libs := get_group_and_liberties(b, ni, context.temp_allocator)
			if len(libs) == 0 {
				if len(group) == 1 {
					last_captured = group[0]
				}
				total_captured += remove_group(b, group[:])
			}
			delete(group)
			delete(libs)
		}
	}

	our_group, our_libs := get_group_and_liberties(b, index, context.temp_allocator)
	if len(our_libs) == 0 {
		remove_group(b, our_group[:])
	} else if total_captured == 1 && len(our_group) == 1 && len(our_libs) == 1 {
		b.ko_point = last_captured
	}
	delete(our_group)
	delete(our_libs)

	b.consecutive_passes = 0
	b.move_count += 1
	b.to_play = opp
}

pass_move :: proc(b: ^GoBoard) -> bool {
	b.seen_hashes[b.current_hash] = {}
	b.consecutive_passes += 1
	b.move_count += 1
	b.to_play = opponent_of(b.to_play)
	b.ko_point = NO_KO
	return true
}

// =============================================================================
// Reversible moves (do_move / undo_move) — used by MCTS to mutate a single
// working_board while descending/ascending the tree. The board's state after
// do_move + undo_move is bit-identical to before do_move.
//
// Captures (both opponent-captures and the own-suicide branch) are pushed onto
// `captures` as (index, color) records. undo_move pops them back.
//
// NOTE: do_move does NOT check legality — it mirrors play_flat_unchecked.
// Callers must verify legality (or accept the resulting state).
// =============================================================================

CaptureRecord :: struct {
	index: i32,
	color: i8,
}

MoveDelta :: struct {
	action:                  int, // PASS_ACTION or [0, size*size)
	capture_start:           int, // index into captures stack
	capture_count:           int,
	prev_ko_point:           int,
	prev_consecutive_passes: int,
	prev_move_count:         int,
	prev_current_hash:       u64,
	prev_to_play:            i8,
	seen_hash_added:         u64, // hash inserted into seen_hashes by this move
	seen_hash_was_new:       bool, // if false, undo must NOT remove it
}

do_move :: proc(b: ^GoBoard, action: int, captures: ^[dynamic]CaptureRecord) -> MoveDelta {
	delta := MoveDelta {
		action                  = action,
		capture_start           = len(captures),
		prev_ko_point           = b.ko_point,
		prev_consecutive_passes = b.consecutive_passes,
		prev_move_count         = b.move_count,
		prev_current_hash       = b.current_hash,
		prev_to_play            = b.to_play,
	}

	// Record + insert seen_hashes entry for the position BEFORE this move.
	_, was_seen := b.seen_hashes[b.current_hash]
	b.seen_hashes[b.current_hash] = {}
	delta.seen_hash_added = b.current_hash
	delta.seen_hash_was_new = !was_seen

	if action == PASS_ACTION {
		b.consecutive_passes += 1
		b.move_count += 1
		b.to_play = opponent_of(b.to_play)
		b.ko_point = NO_KO
		return delta
	}

	// Mirrors play_flat_unchecked, but records every captured stone on `captures`.
	b.board[action] = b.to_play
	b.current_hash ~= b.tables.zobrist[action][int(b.to_play)]
	opp := opponent_of(b.to_play)
	b.ko_point = NO_KO

	total_captured := 0
	last_captured := -1

	nb := b.tables.neighbors[action]
	for k in 0 ..< nb.count {
		ni := nb.indices[k]
		if b.board[ni] == opp {
			group, libs := get_group_and_liberties(b, ni, context.temp_allocator)
			if len(libs) == 0 {
				if len(group) == 1 {last_captured = group[0]}
				for idx in group {
					append(captures, CaptureRecord{index = i32(idx), color = opp})
				}
				total_captured += remove_group(b, group[:])
			}
			delete(group)
			delete(libs)
		}
	}

	our_group, our_libs := get_group_and_liberties(b, action, context.temp_allocator)
	if len(our_libs) == 0 {
		// Multi-stone suicide: our own group gets removed. Record those captures
		// under our own color so undo can restore them correctly.
		for idx in our_group {
			append(captures, CaptureRecord{index = i32(idx), color = b.to_play})
		}
		remove_group(b, our_group[:])
	} else if total_captured == 1 && len(our_group) == 1 && len(our_libs) == 1 {
		b.ko_point = last_captured
	}
	delete(our_group)
	delete(our_libs)

	b.consecutive_passes = 0
	b.move_count += 1
	b.to_play = opp

	delta.capture_count = len(captures) - delta.capture_start
	return delta
}

undo_move :: proc(b: ^GoBoard, delta: MoveDelta, captures: ^[dynamic]CaptureRecord) {
	// Restore captured stones first (they hold board-cell state). For non-pass
	// moves the played stone is also at delta.action — clear it before restoring
	// captures in case the action cell itself was part of an own-suicide.
	if delta.action != PASS_ACTION {
		b.board[delta.action] = EMPTY
	}
	for i in 0 ..< delta.capture_count {
		rec := captures[delta.capture_start + i]
		b.board[rec.index] = rec.color
	}
	resize(captures, delta.capture_start)

	// Scalars: restore wholesale.
	b.current_hash = delta.prev_current_hash
	b.ko_point = delta.prev_ko_point
	b.consecutive_passes = delta.prev_consecutive_passes
	b.move_count = delta.prev_move_count
	b.to_play = delta.prev_to_play

	// Remove the seen_hashes entry we added, but only if it wasn't there before.
	if delta.seen_hash_was_new {
		delete_key(&b.seen_hashes, delta.seen_hash_added)
	}
}

// Tromp-Taylor area score: stones-on-board + empty territory bordered by
// exactly one color, with komi added to white. Returned as Black - White
// (positive = Black wins). NOTE: pure TT doesn't recognize dead stones —
// dead groups still count as stones-on-board. For training data near the
// end of the game this is wrong by margins of ~80pts on contrived dead-group
// fixtures (see experiments/2026-05-16_13-12-12g.1-scoring-fix/). Fine for
// MCTS rollouts inside generated games; potentially wrong as a final reward
// signal for SL data.
score :: proc(b: ^GoBoard) -> f32 {
	black_score := f32(0)
	white_score := b.komi
	n := n_cells(b)
	for i in 0 ..< n {
		if b.board[i] == BLACK {
			black_score += 1
		} else if b.board[i] == WHITE {
			white_score += 1
		}
	}

	visited := make([]bool, n, context.temp_allocator)
	defer delete(visited, context.temp_allocator)

	for i in 0 ..< n {
		if b.board[i] != EMPTY || visited[i] {continue}

		territory := make([dynamic]int, 0, 8, context.temp_allocator)
		defer delete(territory)
		stack := make([dynamic]int, 0, 8, context.temp_allocator)
		defer delete(stack)

		append(&stack, i)
		visited[i] = true
		borders_black := false
		borders_white := false

		for len(stack) > 0 {
			current := pop(&stack)
			append(&territory, current)
			nbrs := b.tables.neighbors[current]
			for k in 0 ..< nbrs.count {
				ni := nbrs.indices[k]
				v := b.board[ni]
				if v == EMPTY {
					if !visited[ni] {
						visited[ni] = true
						append(&stack, ni)
					}
				} else if v == BLACK {
					borders_black = true
				} else if v == WHITE {
					borders_white = true
				}
			}
		}

		if borders_black && !borders_white {
			black_score += f32(len(territory))
		} else if borders_white && !borders_black {
			white_score += f32(len(territory))
		}
	}

	return black_score - white_score
}

// Sign convention: BLACK if score>0, WHITE if score<0, 0 (EMPTY sentinel) on
// tie. Tie is rare with non-integer komi (7.5 default) but legal.
get_winner :: proc(b: ^GoBoard) -> i8 {
	s := score(b)
	if s > 0 {return BLACK}
	if s < 0 {return WHITE}
	return 0
}

set_from_array :: proc(b: ^GoBoard, data: []i8, to_play: i8) {
	n := n_cells(b)
	b.current_hash = 0
	for i in 0 ..< n {
		b.board[i] = data[i]
		if b.board[i] != EMPTY {
			b.current_hash ~= b.tables.zobrist[i][int(b.board[i])]
		}
	}
	clear(&b.seen_hashes)
	b.to_play = to_play
	b.ko_point = NO_KO
	b.consecutive_passes = 0
	mc := 0
	for i in 0 ..< n {
		if b.board[i] != EMPTY {mc += 1}
	}
	b.move_count = mc
}
