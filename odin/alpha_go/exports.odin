package alpha_go

import "base:runtime"
import "core:c"
import "core:math"
import "core:mem"

import mcts "../vendor/mcts-odin/mcts"

// ============================================================================
// C-ABI exports.
//
// Handle-based API. Every exported proc has `proc "c"` calling convention and a
// stable `alphago_*` link_name so the Python ctypes wrapper can dlopen this .so
// and bind via a small surface.
//
// CONTEXT: Odin runtime needs a context for allocators/etc. We materialize a
// default context at the top of each export via `runtime.default_context()`.
//
// MCTS is delegated to the vendored mcts-odin package (odin/vendor/mcts-odin/).
// This file is the C-ABI surface; go_adapter.odin is the Game-vtable bridge.
// ============================================================================

@(export, link_name = "alphago_pass_action")
ffi_pass_action :: proc "c" () -> c.int {return c.int(PASS_ACTION)}

@(export, link_name = "alphago_empty")
ffi_empty :: proc "c" () -> c.char {return c.char(EMPTY)}

@(export, link_name = "alphago_black")
ffi_black :: proc "c" () -> c.char {return c.char(BLACK)}

@(export, link_name = "alphago_white")
ffi_white :: proc "c" () -> c.char {return c.char(WHITE)}

@(export, link_name = "alphago_komi_default")
ffi_komi_default :: proc "c" () -> c.float {return c.float(KOMI_DEFAULT)}

// -------------------- GoBoard --------------------

@(export, link_name = "alphago_goboard_new")
ffi_goboard_new :: proc "c" (size: c.int, komi: c.float) -> rawptr {
	context = runtime.default_context()
	b := new(GoBoard)
	b^ = make_go_board(int(size), f32(komi))
	return rawptr(b)
}

@(export, link_name = "alphago_goboard_free")
ffi_goboard_free :: proc "c" (h: rawptr) {
	context = runtime.default_context()
	if h == nil {return}
	b := cast(^GoBoard)h
	destroy_go_board(b)
	free(b)
}

@(export, link_name = "alphago_goboard_copy")
ffi_goboard_copy :: proc "c" (h: rawptr) -> rawptr {
	context = runtime.default_context()
	src := cast(^GoBoard)h
	dst := new(GoBoard)
	dst^ = clone_go_board(src)
	return rawptr(dst)
}

@(export, link_name = "alphago_goboard_size")
ffi_goboard_size :: proc "c" (h: rawptr) -> c.int {
	b := cast(^GoBoard)h; return c.int(b.size)
}

@(export, link_name = "alphago_goboard_to_play")
ffi_goboard_to_play :: proc "c" (h: rawptr) -> c.char {
	b := cast(^GoBoard)h; return c.char(b.to_play)
}

@(export, link_name = "alphago_goboard_move_count")
ffi_goboard_move_count :: proc "c" (h: rawptr) -> c.int {
	b := cast(^GoBoard)h; return c.int(b.move_count)
}

@(export, link_name = "alphago_goboard_komi")
ffi_goboard_komi :: proc "c" (h: rawptr) -> c.float {
	b := cast(^GoBoard)h; return c.float(b.komi)
}

@(export, link_name = "alphago_goboard_ko_point")
ffi_goboard_ko_point :: proc "c" (h: rawptr) -> c.int {
	b := cast(^GoBoard)h; return c.int(b.ko_point)
}

@(export, link_name = "alphago_goboard_consecutive_passes")
ffi_goboard_consecutive_passes :: proc "c" (h: rawptr) -> c.int {
	b := cast(^GoBoard)h; return c.int(b.consecutive_passes)
}

@(export, link_name = "alphago_goboard_current_hash")
ffi_goboard_current_hash :: proc "c" (h: rawptr) -> u64 {
	b := cast(^GoBoard)h; return b.current_hash
}

@(export, link_name = "alphago_goboard_at_flat")
ffi_goboard_at_flat :: proc "c" (h: rawptr, idx: c.int) -> c.char {
	b := cast(^GoBoard)h; return c.char(b.board[int(idx)])
}

@(export, link_name = "alphago_goboard_play")
ffi_goboard_play :: proc "c" (h: rawptr, row, col: c.int) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return 1 if play(b, int(row), int(col)) else 0
}

@(export, link_name = "alphago_goboard_play_flat")
ffi_goboard_play_flat :: proc "c" (h: rawptr, idx: c.int) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return 1 if play_flat(b, int(idx)) else 0
}

@(export, link_name = "alphago_goboard_pass")
ffi_goboard_pass :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	pass_move(b)
	return 1
}

@(export, link_name = "alphago_goboard_is_legal")
ffi_goboard_is_legal :: proc "c" (h: rawptr, row, col: c.int) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return 1 if is_legal(b, int(row), int(col)) else 0
}

@(export, link_name = "alphago_goboard_is_legal_flat")
ffi_goboard_is_legal_flat :: proc "c" (h: rawptr, idx: c.int) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return 1 if is_legal_flat(b, int(idx)) else 0
}

@(export, link_name = "alphago_goboard_is_game_over")
ffi_goboard_is_game_over :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return 1 if is_game_over(b) else 0
}

@(export, link_name = "alphago_goboard_score")
ffi_goboard_score :: proc "c" (h: rawptr) -> c.float {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return c.float(score(b))
}

@(export, link_name = "alphago_goboard_get_winner")
ffi_goboard_get_winner :: proc "c" (h: rawptr) -> c.char {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	return c.char(get_winner(b))
}

@(export, link_name = "alphago_goboard_get_legal_moves_flat")
ffi_goboard_get_legal_moves_flat :: proc "c" (
	h: rawptr,
	out_buf: ^c.int,
	max_n: c.int,
) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	if out_buf == nil || max_n <= 0 {
		// No caller buffer — fall back to a one-shot count via the dynamic
		// variant (rare path; Python shim always supplies a sized buffer).
		moves := get_legal_moves_flat(b)
		defer delete(moves)
		return c.int(len(moves))
	}
	// Write directly into caller-owned out_buf. Use a small stack scratch sized
	// to the max legal-move count (n_cells), copy the prefix the caller asked for.
	// Stack-allocated int[]; no heap, no append-grow churn (the old path went
	// through [dynamic]int → 5.5% _append_elem in the ydh.6 perf profile).
	scratch: [19 * 19]int = ---
	count := fill_legal_moves_flat(b, scratch[:n_cells(b)])
	n := min(int(max_n), count)
	dst := mem.slice_ptr(out_buf, n)
	for i in 0 ..< n {dst[i] = c.int(scratch[i])}
	return c.int(count)
}

@(export, link_name = "alphago_goboard_to_array")
ffi_goboard_to_array :: proc "c" (h: rawptr, out_buf: ^c.char, max_n: c.int) {
	b := cast(^GoBoard)h
	if out_buf == nil {return}
	n := min(int(max_n), len(b.board))
	dst := mem.slice_ptr(out_buf, n)
	for i in 0 ..< n {dst[i] = c.char(b.board[i])}
}

@(export, link_name = "alphago_goboard_set_from_array")
ffi_goboard_set_from_array :: proc "c" (h: rawptr, data: ^c.char, to_play: c.char) {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	n := b.size * b.size
	src := mem.slice_ptr(data, n)
	scratch := make([]i8, n, context.temp_allocator)
	defer delete(scratch, context.temp_allocator)
	for i in 0 ..< n {scratch[i] = i8(src[i])}
	set_from_array(b, scratch, i8(to_play))
}

// -------------------- MCTSConfig --------------------
//
// FFI handle wraps mcts.Config directly. Field set is identical to autogodin's
// pre-vendor MCTSConfig, so the Python ctypes setter signature is unchanged.

@(export, link_name = "alphago_mcts_config_new")
ffi_config_new :: proc "c" () -> rawptr {
	context = runtime.default_context()
	c := new(mcts.Config)
	c^ = mcts.default_config()
	return rawptr(c)
}

@(export, link_name = "alphago_mcts_config_free")
ffi_config_free :: proc "c" (h: rawptr) {
	context = runtime.default_context()
	if h == nil {return}
	c := cast(^mcts.Config)h
	if c.pcr_sims != nil {delete(c.pcr_sims)}
	if c.pcr_probs != nil {delete(c.pcr_probs)}
	free(c)
}

@(export, link_name = "alphago_mcts_config_set")
ffi_config_set :: proc "c" (
	h: rawptr,
	c_puct, lambda_, dirichlet_alpha, dirichlet_weight, temperature, rollout_temperature: c.float,
	max_depth: c.int,
) {
	cfg := cast(^mcts.Config)h
	cfg.c_puct = f32(c_puct)
	cfg.lambda = f32(lambda_)
	cfg.dirichlet_alpha = f32(dirichlet_alpha)
	cfg.dirichlet_weight = f32(dirichlet_weight)
	cfg.temperature = f32(temperature)
	cfg.rollout_temperature = f32(rollout_temperature)
	cfg.max_depth = int(max_depth)
}

@(export, link_name = "alphago_mcts_config_set_pcr")
ffi_config_set_pcr :: proc "c" (
	h: rawptr,
	sims: ^c.int,
	probs: ^c.float,
	n: c.int,
) {
	context = runtime.default_context()
	cfg := cast(^mcts.Config)h
	if cfg.pcr_sims != nil {delete(cfg.pcr_sims)}
	if cfg.pcr_probs != nil {delete(cfg.pcr_probs)}
	if n <= 0 {return}
	ss := make([]int, int(n))
	pp := make([]f32, int(n))
	src_s := mem.slice_ptr(sims, int(n))
	src_p := mem.slice_ptr(probs, int(n))
	for i in 0 ..< int(n) {ss[i] = int(src_s[i]); pp[i] = f32(src_p[i])}
	cfg.pcr_sims = ss
	cfg.pcr_probs = pp
}

// -------------------- MCTSTree --------------------
//
// FFI tree handle bundles mcts.Tree + the Game vtable + the board size so the
// pass-action translation (Python uses -1, mcts uses size*size) can be done
// at the C-ABI boundary without leaking either convention.

TreeHandle :: struct {
	tree:       mcts.Tree,
	game:       mcts.Game,
	board_size: int,
}

@(private = "file")
to_python_action :: #force_inline proc(action, size: int) -> int {
	return PASS_ACTION if action == size * size else action
}

@(private = "file")
to_mcts_action :: #force_inline proc(action, size: int) -> int {
	return size * size if action == PASS_ACTION else action
}

@(export, link_name = "alphago_mcts_tree_new")
ffi_tree_new :: proc "c" (board: rawptr, config: rawptr, seed: u64) -> rawptr {
	context = runtime.default_context()
	src := cast(^GoBoard)board
	cfg := cast(^mcts.Config)config

	h := new(TreeHandle)
	h.board_size = src.size
	h.game = go_game_vtable(src.size)

	// mcts.init takes ownership of the state and frees it via game.free. The
	// Python-side GoBoard handle stays valid for the caller — we hand the tree
	// an independent clone.
	working := new(GoBoard)
	working^ = clone_go_board(src)

	mcts.init(&h.tree, &h.game, rawptr(working), cfg^, seed)
	return rawptr(h)
}

@(export, link_name = "alphago_mcts_tree_free")
ffi_tree_free :: proc "c" (h: rawptr) {
	context = runtime.default_context()
	if h == nil {return}
	t := cast(^TreeHandle)h
	mcts.destroy(&t.tree)
	free(t)
}

@(export, link_name = "alphago_mcts_tree_size")
ffi_tree_size :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h; return c.int(mcts.tree_size(&t.tree))
}

@(export, link_name = "alphago_mcts_tree_root_visits")
ffi_tree_root_visits :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h; return c.int(mcts.get_root_visit_count(&t.tree))
}

@(export, link_name = "alphago_mcts_tree_root_q")
ffi_tree_root_q :: proc "c" (h: rawptr) -> c.float {
	context = runtime.default_context()
	t := cast(^TreeHandle)h; return c.float(mcts.get_root_q_value(&t.tree))
}

// C-ABI evaluator signature.
//
// `goboard` is a non-owning rawptr to the leaf's Odin GoBoard view. The
// callback must NOT free or take ownership — the MCTS tree owns the lifetime.
// `out_actions`/`out_probs` are caller-allocated buffers of `max_n` entries
// (typically size*size + 1). The callback writes (action, prob) pairs using
// the PYTHON action convention (pass = -1) and the leaf value, returning the
// number of pairs written.
CEvaluator :: #type proc "c" (
	goboard:     rawptr,
	out_actions: ^c.int,
	out_probs:   ^c.float,
	max_n:       c.int,
	out_value:   ^c.float,
	user_data:   rawptr,
) -> c.int

@(private = "file")
CallbackCtx :: struct {
	cb:        CEvaluator,
	user_data: rawptr,
}

// Trampoline from mcts.Evaluator (size*size pass id) to the Python-facing
// CEvaluator (PASS_ACTION = -1). Marshals one leaf at a time.
@(private = "file")
mcts_evaluator_trampoline :: proc(
	state:       rawptr,
	out_actions: []int,
	out_probs:   []f32,
	out_value:   ^f32,
	user_data:   rawptr,
) -> int {
	ctx := cast(^CallbackCtx)user_data
	b := cast(^GoBoard)state
	max_n := b.size * b.size + 1

	// Python-facing scratch buffers (action id in Python convention, pass = -1).
	py_actions := make([]c.int, max_n, context.temp_allocator)
	defer delete(py_actions, context.temp_allocator)
	py_probs := make([]c.float, max_n, context.temp_allocator)
	defer delete(py_probs, context.temp_allocator)

	c_value: c.float
	written := int(ctx.cb(
		rawptr(b),
		raw_data(py_actions),
		raw_data(py_probs),
		c.int(max_n),
		&c_value,
		ctx.user_data,
	))

	// Translate Python action ids -> mcts action ids while copying into mcts's
	// caller-owned out_actions/out_probs. mcts's contract says we may write up
	// to `len(out_actions)` entries; cap at the buffer length defensively.
	n := min(written, len(out_actions), len(out_probs))
	for i in 0 ..< n {
		out_actions[i] = to_mcts_action(int(py_actions[i]), b.size)
		out_probs[i]   = f32(py_probs[i])
	}
	out_value^ = f32(c_value)
	return n
}

@(export, link_name = "alphago_mcts_tree_run_simulations")
ffi_tree_run_simulations :: proc "c" (
	h: rawptr,
	num_simulations: c.int,
	cb: CEvaluator,
	user_data: rawptr,
) {
	context = runtime.default_context()
	// The adapter (go_adapter.odin) parks short-lived Adapter_Delta + captures
	// allocations on context.temp_allocator inside the MCTS hot loop, and the
	// trampoline parks scratch action/prob buffers there too. Reset before AND
	// after so the arena stays bounded across many run_simulations calls.
	free_all(context.temp_allocator)
	defer free_all(context.temp_allocator)
	t := cast(^TreeHandle)h
	cctx := CallbackCtx{cb = cb, user_data = user_data}
	mcts.run_simulations(&t.tree, int(num_simulations), mcts_evaluator_trampoline, &cctx)
}

// Root-parallel MCTS: n_threads workers each descend / expand / backup via
// shared tree; each worker has its own per-thread context (own temp_allocator,
// own state clone). CFUNCTYPE callbacks from Python auto-acquire the GIL, so
// dict-style EvaluatorFn / FlatEvaluatorFn both work transparently — but they
// serialize on the GIL inside the evaluator body, so threading wins only
// materialise to the extent the Odin-side work (descent + backup) dominates.
// Threading an Odin-only evaluator (in-process) gives full parallelism.
// See autogodin-i5d.
@(export, link_name = "alphago_mcts_tree_run_simulations_threaded")
ffi_tree_run_simulations_threaded :: proc "c" (
	h: rawptr,
	num_simulations: c.int,
	n_threads: c.int,
	cb: CEvaluator,
	user_data: rawptr,
) {
	context = runtime.default_context()
	free_all(context.temp_allocator)
	defer free_all(context.temp_allocator)
	t := cast(^TreeHandle)h
	cctx := CallbackCtx{cb = cb, user_data = user_data}
	mcts.run_simulations_threaded(
		&t.tree,
		int(num_simulations),
		int(n_threads),
		mcts_evaluator_trampoline,
		&cctx,
	)
}

// -------------------- batched evaluator path --------------------
//
// Mirrors the sequential CEvaluator but for batched search.
//
// `states` is a contiguous block of `batch_size` GoBoard* (non-owning).
// `out_actions` and `out_probs` are flat row-major buffers of
// `batch_size * max_n_per_state` entries — row i belongs to state i.
// `out_counts[i]` and `out_values[i]` are per-state outputs.
// All action ids use the PYTHON convention (pass = -1).
CEvaluatorBatched :: #type proc "c" (
	batch_size:      c.int,
	states:          ^rawptr,
	out_actions:     ^c.int,
	out_probs:       ^c.float,
	out_counts:      ^c.int,
	out_values:      ^c.float,
	max_n_per_state: c.int,
	user_data:       rawptr,
)

@(private = "file")
BatchedCallbackCtx :: struct {
	cb:        CEvaluatorBatched,
	user_data: rawptr,
}

// Trampoline from mcts.Evaluator_Batched (per-state action slices, mcts action
// ids) to the Python-facing CEvaluatorBatched (one flat row-major buffer, pass
// = -1). Allocates per-call scratch on temp_allocator — the FFI entry point
// free_all's it on each run_simulations_batched call.
@(private = "file")
mcts_evaluator_batched_trampoline :: proc(
	states:      []rawptr,
	out_actions: [][]int,
	out_probs:   [][]f32,
	out_counts:  []int,
	out_values:  []f32,
	user_data:   rawptr,
) {
	ctx := cast(^BatchedCallbackCtx)user_data
	if len(states) == 0 {return}

	// Infer board size from the first state; mcts.run_simulations_batched
	// is currently single-game so all leaves share size.
	b0 := cast(^GoBoard)states[0]
	size := b0.size
	max_n := size * size + 1
	n := len(states)

	// Flat row-major scratch for the Python callback.
	py_states  := make([]rawptr, n,                context.temp_allocator)
	py_actions := make([]c.int,  n * max_n,        context.temp_allocator)
	py_probs   := make([]c.float, n * max_n,       context.temp_allocator)
	py_counts  := make([]c.int,  n,                context.temp_allocator)
	py_values  := make([]c.float, n,               context.temp_allocator)
	defer {
		delete(py_states,  context.temp_allocator)
		delete(py_actions, context.temp_allocator)
		delete(py_probs,   context.temp_allocator)
		delete(py_counts,  context.temp_allocator)
		delete(py_values,  context.temp_allocator)
	}

	for i in 0 ..< n {py_states[i] = states[i]}

	ctx.cb(
		c.int(n),
		raw_data(py_states),
		raw_data(py_actions),
		raw_data(py_probs),
		raw_data(py_counts),
		raw_data(py_values),
		c.int(max_n),
		ctx.user_data,
	)

	// Unpack into mcts's caller-owned slices, translating action ids.
	for i in 0 ..< n {
		written := int(py_counts[i])
		row_lim := min(written, max_n, len(out_actions[i]), len(out_probs[i]))
		row_base := i * max_n
		for k in 0 ..< row_lim {
			out_actions[i][k] = to_mcts_action(int(py_actions[row_base + k]), size)
			out_probs[i][k]   = f32(py_probs[row_base + k])
		}
		out_counts[i] = row_lim
		out_values[i] = f32(py_values[i])
	}
}

@(export, link_name = "alphago_mcts_tree_run_simulations_batched")
ffi_tree_run_simulations_batched :: proc "c" (
	h: rawptr,
	num_simulations: c.int,
	batch_size: c.int,
	cb: CEvaluatorBatched,
	user_data: rawptr,
) {
	context = runtime.default_context()
	free_all(context.temp_allocator)
	defer free_all(context.temp_allocator)
	t := cast(^TreeHandle)h
	cctx := BatchedCallbackCtx{cb = cb, user_data = user_data}
	mcts.run_simulations_batched(
		&t.tree, int(num_simulations), int(batch_size),
		mcts_evaluator_batched_trampoline, &cctx,
	)
}

@(export, link_name = "alphago_mcts_tree_select_action")
ffi_tree_select_action :: proc "c" (h: rawptr, temperature: c.float) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h
	a := mcts.select_action(&t.tree, f32(temperature))
	return c.int(to_python_action(a, t.board_size))
}

// Writes (action, count) pairs to out_actions/out_counts; returns count written.
@(export, link_name = "alphago_mcts_tree_child_visits")
ffi_tree_child_visits :: proc "c" (
	h: rawptr,
	out_actions: ^c.int,
	out_counts: ^c.int,
	max_n: c.int,
) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h
	cv := mcts.get_child_visit_counts(&t.tree)
	defer delete(cv)
	n := min(int(max_n), len(cv))
	if out_actions == nil || out_counts == nil {return c.int(len(cv))}
	a := mem.slice_ptr(out_actions, n)
	c_ := mem.slice_ptr(out_counts, n)
	i := 0
	for action, count in cv {
		if i >= n {break}
		a[i] = c.int(to_python_action(action, t.board_size))
		c_[i] = c.int(count); i += 1
	}
	return c.int(len(cv))
}

@(export, link_name = "alphago_mcts_tree_child_q_values")
ffi_tree_child_q_values :: proc "c" (
	h: rawptr,
	out_actions: ^c.int,
	out_q: ^c.float,
	max_n: c.int,
) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h
	cq := mcts.get_child_q_values(&t.tree)
	defer delete(cq)
	n := min(int(max_n), len(cq))
	if out_actions == nil || out_q == nil {return c.int(len(cq))}
	a := mem.slice_ptr(out_actions, n)
	q := mem.slice_ptr(out_q, n)
	i := 0
	for action, qv in cq {
		if i >= n {break}
		a[i] = c.int(to_python_action(action, t.board_size))
		q[i] = c.float(qv); i += 1
	}
	return c.int(len(cq))
}

@(export, link_name = "alphago_mcts_tree_action_probabilities")
ffi_tree_action_probabilities :: proc "c" (
	h: rawptr,
	temperature: c.float,
	out_actions: ^c.int,
	out_probs: ^c.float,
	max_n: c.int,
) -> c.int {
	context = runtime.default_context()
	t := cast(^TreeHandle)h
	probs := mcts.get_action_probabilities(&t.tree, f32(temperature))
	defer delete(probs)
	n := min(int(max_n), len(probs))
	if out_actions == nil || out_probs == nil {return c.int(len(probs))}
	a := mem.slice_ptr(out_actions, n)
	p := mem.slice_ptr(out_probs, n)
	i := 0
	for action, pv in probs {
		if i >= n {break}
		a[i] = c.int(to_python_action(action, t.board_size))
		p[i] = c.float(pv); i += 1
	}
	return c.int(len(probs))
}

// Suppress "unused import" warning.
@(private = "file")
_unused_math :: math.PI
