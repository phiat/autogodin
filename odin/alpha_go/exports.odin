package alpha_go

import "base:runtime"
import "core:c"
import "core:math"
import "core:mem"

// ============================================================================
// C-ABI exports.
//
// Handle-based API. Every exported proc has `proc "c"` calling convention and a
// stable `alphago_*` link_name so the Python ctypes wrapper can dlopen this .so
// and bind via a small surface.
//
// CONTEXT: Odin runtime needs a context for allocators/etc. We materialize a
// default context at the top of each export via `runtime.default_context()`.
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

// Writes up to max_n legal-move flat indices to out_buf; returns the count.
// (Always returns count even if it exceeds max_n — caller can resize and retry.)
@(export, link_name = "alphago_goboard_get_legal_moves_flat")
ffi_goboard_get_legal_moves_flat :: proc "c" (
	h: rawptr,
	out_buf: ^c.int,
	max_n: c.int,
) -> c.int {
	context = runtime.default_context()
	b := cast(^GoBoard)h
	moves := get_legal_moves_flat(b)
	defer delete(moves)
	if out_buf != nil && max_n > 0 {
		n := min(int(max_n), len(moves))
		dst := mem.slice_ptr(out_buf, n)
		for i in 0 ..< n {dst[i] = c.int(moves[i])}
	}
	return c.int(len(moves))
}

// Copies the board into out_buf (size*size i8 entries). Caller must pre-allocate.
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
	// Re-interpret c.char slice as i8 slice (both are 1-byte).
	scratch := make([]i8, n, context.temp_allocator)
	defer delete(scratch, context.temp_allocator)
	for i in 0 ..< n {scratch[i] = i8(src[i])}
	set_from_array(b, scratch, i8(to_play))
}

// -------------------- MCTSConfig --------------------
//
// Exposed as a plain heap struct that Python ctypes mirrors. We keep `pcr_*` as
// opaque slices owned by Odin; Python sets them via a dedicated setter.

@(export, link_name = "alphago_mcts_config_new")
ffi_config_new :: proc "c" () -> rawptr {
	context = runtime.default_context()
	c := new(MCTSConfig)
	c^ = default_mcts_config()
	return rawptr(c)
}

@(export, link_name = "alphago_mcts_config_free")
ffi_config_free :: proc "c" (h: rawptr) {
	context = runtime.default_context()
	if h == nil {return}
	c := cast(^MCTSConfig)h
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
	cfg := cast(^MCTSConfig)h
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
	cfg := cast(^MCTSConfig)h
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

@(export, link_name = "alphago_mcts_tree_new")
ffi_tree_new :: proc "c" (board: rawptr, config: rawptr, seed: u64) -> rawptr {
	context = runtime.default_context()
	b := cast(^GoBoard)board
	cfg := cast(^MCTSConfig)config
	t := new(MCTSTree)
	t^ = make_mcts_tree(b, cfg^, seed)
	return rawptr(t)
}

@(export, link_name = "alphago_mcts_tree_free")
ffi_tree_free :: proc "c" (h: rawptr) {
	context = runtime.default_context()
	if h == nil {return}
	t := cast(^MCTSTree)h
	destroy_mcts_tree(t)
	free(t)
}

@(export, link_name = "alphago_mcts_tree_size")
ffi_tree_size :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	t := cast(^MCTSTree)h; return c.int(tree_size(t))
}

@(export, link_name = "alphago_mcts_tree_root_visits")
ffi_tree_root_visits :: proc "c" (h: rawptr) -> c.int {
	context = runtime.default_context()
	t := cast(^MCTSTree)h; return c.int(get_root_visit_count(t))
}

@(export, link_name = "alphago_mcts_tree_root_q")
ffi_tree_root_q :: proc "c" (h: rawptr) -> c.float {
	context = runtime.default_context()
	t := cast(^MCTSTree)h; return c.float(get_root_q_value(t))
}

// C-ABI evaluator signature: caller fills out_actions[0..n_out), out_probs[0..n_out),
// and writes the leaf value into *out_value. max_n is the buffer capacity (typically
// size*size + 1). Returns number of (action, prob) pairs written.
CEvaluator :: #type proc "c" (
	board_data: ^c.char,
	to_play:    c.char,
	size:       c.int,
	out_actions: ^c.int,
	out_probs:   ^c.float,
	max_n:       c.int,
	out_value:   ^c.float,
	user_data:   rawptr,
) -> c.int

// Trampoline state for wrapping a C evaluator into the Odin EvaluatorFn.
@(private)
CallbackCtx :: struct {
	cb:        CEvaluator,
	user_data: rawptr,
}

@(private)
odin_evaluator_trampoline :: proc(state: ^GoBoard, user_data: rawptr) -> (map[int]f32, f32) {
	ctx := cast(^CallbackCtx)user_data
	n := state.size * state.size
	max_n := n + 1

	board_buf := make([]c.char, n, context.temp_allocator)
	defer delete(board_buf, context.temp_allocator)
	for i in 0 ..< n {board_buf[i] = c.char(state.board[i])}

	actions := make([]c.int, max_n, context.temp_allocator)
	defer delete(actions, context.temp_allocator)
	probs := make([]c.float, max_n, context.temp_allocator)
	defer delete(probs, context.temp_allocator)
	value: c.float

	written := ctx.cb(
		raw_data(board_buf),
		c.char(state.to_play),
		c.int(state.size),
		raw_data(actions),
		raw_data(probs),
		c.int(max_n),
		&value,
		ctx.user_data,
	)

	policy := make(map[int]f32, int(written))
	for i in 0 ..< int(written) {
		policy[int(actions[i])] = f32(probs[i])
	}
	return policy, f32(value)
}

@(export, link_name = "alphago_mcts_tree_run_simulations")
ffi_tree_run_simulations :: proc "c" (
	h: rawptr,
	num_simulations: c.int,
	cb: CEvaluator,
	user_data: rawptr,
) {
	context = runtime.default_context()
	t := cast(^MCTSTree)h
	cctx := CallbackCtx{cb = cb, user_data = user_data}
	run_simulations(t, int(num_simulations), odin_evaluator_trampoline, &cctx)
}

@(export, link_name = "alphago_mcts_tree_select_action")
ffi_tree_select_action :: proc "c" (h: rawptr, temperature: c.float) -> c.int {
	context = runtime.default_context()
	t := cast(^MCTSTree)h
	return c.int(select_action(t, f32(temperature)))
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
	t := cast(^MCTSTree)h
	cv := get_child_visit_counts(t)
	defer delete(cv)
	n := min(int(max_n), len(cv))
	if out_actions == nil || out_counts == nil {return c.int(len(cv))}
	a := mem.slice_ptr(out_actions, n)
	c_ := mem.slice_ptr(out_counts, n)
	i := 0
	for action, count in cv {
		if i >= n {break}
		a[i] = c.int(action); c_[i] = c.int(count); i += 1
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
	t := cast(^MCTSTree)h
	cq := get_child_q_values(t)
	defer delete(cq)
	n := min(int(max_n), len(cq))
	if out_actions == nil || out_q == nil {return c.int(len(cq))}
	a := mem.slice_ptr(out_actions, n)
	q := mem.slice_ptr(out_q, n)
	i := 0
	for action, qv in cq {
		if i >= n {break}
		a[i] = c.int(action); q[i] = c.float(qv); i += 1
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
	t := cast(^MCTSTree)h
	probs := get_action_probabilities(t, f32(temperature))
	defer delete(probs)
	n := min(int(max_n), len(probs))
	if out_actions == nil || out_probs == nil {return c.int(len(probs))}
	a := mem.slice_ptr(out_actions, n)
	p := mem.slice_ptr(out_probs, n)
	i := 0
	for action, pv in probs {
		if i >= n {break}
		a[i] = c.int(action); p[i] = c.float(pv); i += 1
	}
	return c.int(len(probs))
}

// Suppress "unused import" warning.
@(private)
_ :: math.PI
