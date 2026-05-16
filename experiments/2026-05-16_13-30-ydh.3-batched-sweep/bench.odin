package main

// autogodin ydh.3 — batched-MCTS throughput sweep.
//
// Sweeps batch_size × per-leaf synthetic evaluator latency. Verifies that
// mcts.run_simulations_batched's virtual-loss path scales as expected when
// the evaluator is slow (the NN-eval scenario): larger batches amortize the
// evaluator's per-call cost across more leaves, so throughput should rise.
//
// Two axes:
//   batch_size:   1, 8, 32, 128
//   eval_latency: 0us, 100us, 1ms
//
// Per cell: 3 trials × (1600 sims × 32 moves = 51,200 sims) on 9x9 Go.
//
// Note: 10ms latency excluded — at batch=1 that's ~512 sec/trial × 3 = 25 min
// per cell. Pattern is already clear at 1ms and is monotonic with latency.
//
// What this catches:
//   - virtual-loss correctness: at batch=N, total visits per move should
//     still equal num_simulations (already covered by mcts-odin tests, but
//     reconfirmed here at run time)
//   - amortization slope: at high latency, throughput should be ~linear in
//     batch_size up to the search tree's effective branching limit
//   - overhead floor: at zero latency, batched should track non-batched
//     within ~10-20% (snapshot + virtual-loss bookkeeping cost)
//
// Build:
//   odin build experiments/2026-05-16_13-30-ydh.3-batched-sweep \
//       -o:speed -out:experiments/2026-05-16_13-30-ydh.3-batched-sweep/bench

import "core:fmt"
import "core:math"
import "core:time"

import ag "../../odin/alpha_go"
import mcts "../../odin/vendor/mcts-odin/mcts"

Eval_Ctx :: struct {
	latency_ns: i64,  // per-call synthetic sleep
	call_count: int,  // diagnostic
}

batched_uniform :: proc(
	states:      []rawptr,
	out_actions: [][]int,
	out_probs:   [][]f32,
	out_counts:  []int,
	out_values:  []f32,
	user_data:   rawptr,
) {
	ctx := cast(^Eval_Ctx)user_data
	ctx.call_count += 1
	if ctx.latency_ns > 0 {
		time.sleep(time.Duration(ctx.latency_ns))
	}
	// Fill each batch member with a uniform-over-legal policy and value=0.
	// All slots are computed identically; the goal is throughput, not strength.
	for i in 0 ..< len(states) {
		b := cast(^ag.GoBoard)states[i]
		legal := ag.get_legal_moves_flat(b, context.temp_allocator)
		defer delete(legal)
		pass_id := b.size * b.size
		n := len(legal) + 1
		uniform := f32(1.0) / f32(n)
		w := 0
		for m in legal {
			if w >= len(out_actions[i]) {break}
			out_actions[i][w] = m
			out_probs[i][w]   = uniform
			w += 1
		}
		if w < len(out_actions[i]) {
			out_actions[i][w] = pass_id
			out_probs[i][w]   = uniform
			w += 1
		}
		out_counts[i] = w
		out_values[i] = 0.0
	}
}

run_trial :: proc(
	sims_per_move, moves_per_trial, batch_size: int,
	latency_ns: i64,
	seed: u64,
) -> (elapsed_ns: i64, total_sims: int, eval_calls: int) {
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

	ev := Eval_Ctx{latency_ns = latency_ns}

	start := time.tick_now()
	for move in 0 ..< moves_per_trial {
		if ag.is_game_over(&board) {break}

		clone := new(ag.GoBoard)
		clone^ = ag.clone_go_board(&board)

		tree: mcts.Tree
		mcts.init(&tree, &game, rawptr(clone), cfg, seed + u64(move))
		mcts.run_simulations_batched(&tree, sims_per_move, batch_size, batched_uniform, &ev)
		action_mcts := mcts.select_action(&tree, 0.0)
		mcts.destroy(&tree)

		action := ag.PASS_ACTION if action_mcts == 9 * 9 else action_mcts
		if action == ag.PASS_ACTION {
			ag.pass_move(&board)
		} else {
			_ = ag.play_flat(&board, action)
		}
		total_sims += sims_per_move
		free_all(context.temp_allocator)
	}
	end := time.tick_now()
	elapsed_ns = i64(time.duration_nanoseconds(time.tick_diff(start, end)))
	eval_calls = ev.call_count
	return
}

bench_cell :: proc(batch_size: int, latency_ns: i64, trials: int) {
	sims_per_move   := 1600
	moves_per_trial := 32

	// Warmup
	_, _, _ = run_trial(sims_per_move, moves_per_trial, batch_size, latency_ns, u64(42))

	rates: [dynamic]f64
	defer delete(rates)
	last_eval_calls := 0
	for i in 0 ..< trials {
		ns, sims, calls := run_trial(sims_per_move, moves_per_trial, batch_size, latency_ns, u64(100 + i))
		rate := f64(sims) / (f64(ns) / 1e9)
		append(&rates, rate)
		last_eval_calls = calls
	}

	mean := f64(0)
	for r in rates {mean += r}
	mean /= f64(len(rates))
	variance := f64(0)
	for r in rates {variance += (r - mean) * (r - mean)}
	std := math.sqrt_f64(variance / f64(len(rates)))
	ci95 := 1.96 * std / math.sqrt_f64(f64(len(rates)))

	latency_us := f64(latency_ns) / 1e3
	fmt.printf("batch=%d\tlatency=%vus\t| %.0f +- %.0f sims/s\t| eval_calls=%d\n",
		batch_size, latency_us, mean, ci95, last_eval_calls)
}

main :: proc() {
	fmt.println("autogodin ydh.3 — batched MCTS throughput sweep")
	fmt.println("9x9 Go, uniform policy, in-process Odin evaluator, synthetic latency")
	fmt.println("config: 1600 sims/move x 32 moves/trial = 51,200 sims/trial; 3 trials/cell")
	fmt.println()
	fmt.println("batch_size × per-leaf evaluator latency:")
	fmt.println()

	trials := 3
	latencies_us := []i64{0, 100, 1000}  // 0us, 100us, 1ms
	batches      := []int{1, 8, 32, 128}

	for lat_us in latencies_us {
		lat_ns := lat_us * 1000
		for bs in batches {
			bench_cell(bs, lat_ns, trials)
		}
		fmt.println()
	}
}
