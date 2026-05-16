package main

// autogodin 4ig — in-process Odin evaluator bench against autogodin's GoBoard.
//
// Isolates Python ctypes FFI cost. Mirrors experiments/2026-05-16_05-40-mcts-bench-
// cpp-vs-odin/bench.py exactly except the evaluator is an in-process Odin proc
// instead of a Python callback through the ctypes trampoline.
//
// Config (matches bench.py defaults):
//   size = 9, komi = 7.5
//   trials = 5, warmups = 1
//   sims_per_move = 1600, moves_per_trial = 32
//   c_puct = 1.0, lambda = 0.0, dirichlet_alpha/weight = 0.0, max_depth = 100
//   temperature = 1.0 (used at select_action; argmax via temperature = 0.0)
//   evaluator: uniform over legal + pass, value = 0.0
//
// The delta vs the Python bench's autogodin run (20,773 ± 132 sims/s on miniwini,
// post-FPU vendor) is the Python ctypes round-trip cost.
//
// Build:
//   odin build experiments/2026-05-16_12-50-4ig-inprocess-bench \
//       -o:speed -out:experiments/2026-05-16_12-50-4ig-inprocess-bench/bench
//
// Run: just execute the produced binary.

import "core:fmt"
import "core:math"
import "core:time"

import ag "../../odin/alpha_go"
import mcts "../../odin/vendor/mcts-odin/mcts"

uniform_evaluator :: proc(
	state:       rawptr,
	out_actions: []int,
	out_probs:   []f32,
	out_value:   ^f32,
	user_data:   rawptr,
) -> int {
	b := cast(^ag.GoBoard)state
	legal := ag.get_legal_moves_flat(b, context.temp_allocator)
	defer delete(legal) // [dynamic] uses its stored allocator (temp_allocator)

	pass_id := b.size * b.size
	n := len(legal) + 1
	uniform := f32(1.0) / f32(n)
	written := 0
	for m in legal {
		if written >= len(out_actions) {break}
		out_actions[written] = m
		out_probs[written]   = uniform
		written += 1
	}
	if written < len(out_actions) {
		out_actions[written] = pass_id
		out_probs[written]   = uniform
		written += 1
	}
	out_value^ = 0.0
	return written
}

run_trial :: proc(sims_per_move, moves_per_trial: int, seed: u64) -> (elapsed_ns: i64, total_sims: int) {
	// Match bench.py: fresh GoBoard per trial; fresh MCTSTree per move.
	cfg := mcts.default_config()
	cfg.c_puct = 1.0
	cfg.lambda = 0.0
	cfg.dirichlet_alpha = 0.0
	cfg.dirichlet_weight = 0.0
	cfg.temperature = 1.0
	cfg.max_depth = 100

	board := ag.make_go_board(9, 7.5)
	defer ag.destroy_go_board(&board)

	game := ag.go_game_vtable(9)
	start := time.tick_now()

	for move in 0 ..< moves_per_trial {
		if ag.is_game_over(&board) {break}

		// Hand mcts.init an owned clone of the working board so the tree
		// can free it on destroy without affecting our outer state.
		clone := new(ag.GoBoard)
		clone^ = ag.clone_go_board(&board)

		tree: mcts.Tree
		mcts.init(&tree, &game, rawptr(clone), cfg, seed + u64(move))
		mcts.run_simulations(&tree, sims_per_move, uniform_evaluator)
		action_mcts := mcts.select_action(&tree, 0.0) // argmax visits
		mcts.destroy(&tree)

		// Translate mcts action id (pass = size*size) to autogodin action id (pass = -1).
		action := ag.PASS_ACTION if action_mcts == 9 * 9 else action_mcts

		if action == ag.PASS_ACTION {
			ag.pass_move(&board)
		} else {
			_ = ag.play_flat(&board, action)
		}

		total_sims += sims_per_move
		// Bench discipline: drain temp_allocator between moves so the
		// arena doesn't grow without bound across the trial.
		free_all(context.temp_allocator)
	}

	end := time.tick_now()
	elapsed_ns = i64(time.duration_nanoseconds(time.tick_diff(start, end)))
	return
}

main :: proc() {
	sims_per_move   := 1600
	moves_per_trial := 32
	warmups         := 1
	trials          := 5

	fmt.println("autogodin 4ig — in-process Odin evaluator vs Python ctypes")
	fmt.println("9x9 Go, uniform evaluator, single-thread, post-FPU vendor")
	fmt.printf("config: %d sims/move x %d moves/trial = %d sims/trial\n",
		sims_per_move, moves_per_trial, sims_per_move * moves_per_trial)
	fmt.printf("warmup: %d  trials: %d\n\n", warmups, trials)

	for i in 0 ..< warmups {
		ns, sims := run_trial(sims_per_move, moves_per_trial, u64(42 + i))
		fmt.printf("warmup %d: %.3fs, %d sims, %.0f sims/s\n",
			i, f64(ns) / 1e9, sims, f64(sims) / (f64(ns) / 1e9))
	}

	rates: [dynamic]f64
	defer delete(rates)

	for i in 0 ..< trials {
		ns, sims := run_trial(sims_per_move, moves_per_trial, u64(100 + i))
		rate := f64(sims) / (f64(ns) / 1e9)
		append(&rates, rate)
		fmt.printf("trial %d: %.3fs, %d sims, %.0f sims/s\n",
			i, f64(ns) / 1e9, sims, rate)
	}

	mean := f64(0)
	for r in rates {mean += r}
	mean /= f64(len(rates))
	variance := f64(0)
	for r in rates {variance += (r - mean) * (r - mean)}
	std := math.sqrt_f64(variance / f64(len(rates)))
	ci95 := 1.96 * std / math.sqrt_f64(f64(len(rates)))

	fmt.println()
	fmt.printf("in-process odin: %.0f ± %.0f sims/sec (95%% CI, n=%d)\n",
		mean, ci95, len(rates))
	fmt.println()
	fmt.println("Reference (miniwini, same config):")
	fmt.println("  autogodin Python ctypes (post-FPU): 20,773 ± 132 sims/s")
	fmt.println("  alpha_go_cpp:                        8,655 ± 86  sims/s")
	if mean > 0 {
		fmt.printf("\nFFI cost (in-process vs Python ctypes): %.1f%% slowdown\n",
			(mean - 20773) / mean * 100.0)
	}
}
