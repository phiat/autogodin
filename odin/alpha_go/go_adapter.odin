package alpha_go

// Go vtable wrapping our GoBoard for the vendored mcts-odin package.
//
// Action space translation:
//   - MCTS sees actions in [0, size*size]. pass = size*size.
//   - Internally GoBoard uses PASS_ACTION = -1.
//   - Adapter does the translation in legal_actions / mcts_do_move.
//
// Players: BLACK -> 0, WHITE -> 1 (mcts convention).
//
// Move_Delta packing: mcts.Move_Delta has 3 slots (hash, flags, extra) which
// isn't enough for autogodin's MoveDelta (10+ fields). do_move allocates a
// fresh Adapter_Delta on context.allocator carrying the full MoveDelta plus
// the per-move captures buffer, stashes the pointer in .extra, and undo_move
// frees both. mcts-odin uses do_move/undo_move (never clone-on-descent here)
// so every delta is paired with an undo.
//
// This file is the integration bridge described in
// docs/design-mcts-odin-integration.md. mcts-odin's own games/go/game.odin
// was extracted from autogodin's GoBoard, so the bridge is mechanical.

import mcts "../vendor/mcts-odin/mcts"

@(private = "file")
Adapter_Delta :: struct {
	delta:    MoveDelta,
	captures: [dynamic]CaptureRecord,
}

// MCTS-facing pass id for a board of this size.
@(private = "file")
mcts_pass_id :: proc(b: ^GoBoard) -> int {
	return b.size * b.size
}

@(private = "file")
adapter_new_state :: proc(size: int = 9, komi: f32 = KOMI_DEFAULT, allocator := context.allocator) -> rawptr {
	b := new(GoBoard, allocator)
	b^ = make_go_board(size, komi, allocator)
	return rawptr(b)
}

@(private = "file")
adapter_free :: proc(state: rawptr) {
	if state == nil {return}
	b := cast(^GoBoard)state
	alloc := b.allocator
	destroy_go_board(b)
	free(b, alloc)
}

@(private = "file")
adapter_clone :: proc(state: rawptr) -> rawptr {
	src := cast(^GoBoard)state
	dst := new(GoBoard, src.allocator)
	dst^ = clone_go_board(src, src.allocator)
	return rawptr(dst)
}

@(private = "file")
adapter_is_terminal :: proc(state: rawptr) -> bool {
	return is_game_over(cast(^GoBoard)state)
}

// Terminal value from the CURRENT to_play's perspective. mcts convention:
//   draw -> 0.5;  winner == to_play -> 1.0;  else 0.0.
@(private = "file")
adapter_terminal_value :: proc(state: rawptr) -> f32 {
	b := cast(^GoBoard)state
	w := get_winner(b)
	if w == 0 {return 0.5}
	return 1.0 if w == b.to_play else 0.0
}

@(private = "file")
adapter_legal_actions :: proc(state: rawptr, out: ^[dynamic]int) {
	b := cast(^GoBoard)state
	if is_game_over(b) {return}
	n := b.size * b.size
	for i in 0 ..< n {
		if is_legal_flat(b, i) {
			append(out, i)
		}
	}
	append(out, n) // PASS action id = size*size
}

@(private = "file")
adapter_current_player :: proc(state: rawptr) -> i32 {
	b := cast(^GoBoard)state
	return 0 if b.to_play == BLACK else 1
}

// Allocator strategy. Adapter_Delta + its captures buffer are short-lived
// (one MCTS descent leg) and live behind mcts.Move_Delta.extra. We allocate
// them on context.temp_allocator — by contract mcts-odin never touches the
// caller's temp_allocator, so it's ours to use as a scratch arena. The C-ABI
// boundary (ffi_tree_run_simulations) is responsible for free_all-ing the
// temp_allocator at entry/exit so it stays bounded across calls.
//
// undo_move's delete/free calls are no-ops on the default arena temp_allocator
// but kept for correctness when callers swap a different allocator in.
@(private = "file")
adapter_do_move :: proc(state: rawptr, action: int) -> mcts.Move_Delta {
	b := cast(^GoBoard)state
	internal_action := PASS_ACTION if action == mcts_pass_id(b) else action

	ad := new(Adapter_Delta, context.temp_allocator)
	ad.captures = make([dynamic]CaptureRecord, 0, 4, context.temp_allocator)
	ad.delta = do_move(b, internal_action, &ad.captures)
	return mcts.Move_Delta {
		hash  = ad.delta.prev_current_hash,
		flags = 0,
		extra = rawptr(ad),
	}
}

@(private = "file")
adapter_undo_move :: proc(state: rawptr, delta: mcts.Move_Delta) {
	if delta.extra == nil {return}
	b := cast(^GoBoard)state
	ad := cast(^Adapter_Delta)delta.extra
	undo_move(b, ad.delta, &ad.captures)
	delete(ad.captures)
	free(ad, context.temp_allocator)
}

// Returns the Game vtable for Go. max_actions is sized to the supplied
// board size (default 9x9 + pass). The vtable is stateless; one instance
// can drive many trees.
go_game_vtable :: proc(size: int = 9) -> mcts.Game {
	return mcts.Game {
		clone          = adapter_clone,
		free           = adapter_free,
		do_move        = adapter_do_move,
		undo_move      = adapter_undo_move,
		is_terminal    = adapter_is_terminal,
		terminal_value = adapter_terminal_value,
		legal_actions  = adapter_legal_actions,
		current_player = adapter_current_player,
		max_actions    = size * size + 1,
	}
}

// Convenience helper for FFI callers and tests: spin up a new GoBoard as a
// rawptr suitable for handing to mcts.init alongside the vtable.
go_adapter_new_state :: proc(size: int = 9, komi: f32 = KOMI_DEFAULT, allocator := context.allocator) -> rawptr {
	return adapter_new_state(size, komi, allocator)
}
