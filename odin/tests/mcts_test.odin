package alpha_go_tests

// MCTS integration tests for the vendored mcts-odin package + go_adapter.
//
// The semantics are unchanged from the pre-vendor tests: same fixed seed,
// same uniform/biased priors, same visit-count + tree-size invariants. What
// moved is the *implementation* — autogodin's hand-rolled MCTS is gone; we
// now drive `mcts.Tree` (vendored at odin/vendor/mcts-odin) through the
// `go_adapter` Game vtable.
//
// Action convention: tests use the mcts-side id space (pass = size*size).
// The adapter handles the boundary to Python's PASS_ACTION = -1.

import "core:slice"
import "core:testing"
import ag "../alpha_go"
import mcts "../vendor/mcts-odin/mcts"

@(private = "file")
PASS_MCTS :: 9 * 9

@(private = "file")
uniform_evaluator :: proc(
	state:       rawptr,
	out_actions: []int,
	out_probs:   []f32,
	out_value:   ^f32,
	user_data:   rawptr,
) -> int {
	b := cast(^ag.GoBoard)state
	moves := ag.get_legal_moves_flat(b)
	defer delete(moves)
	pass_id := b.size * b.size
	n := len(moves) + 1
	uniform := f32(1.0) / f32(n)
	written := 0
	for m in moves {
		if written >= len(out_actions) {break}
		out_actions[written] = m
		out_probs[written] = uniform
		written += 1
	}
	if written < len(out_actions) {
		out_actions[written] = pass_id
		out_probs[written] = uniform
		written += 1
	}
	out_value^ = 0.5
	return written
}

@(private = "file")
biased_evaluator :: proc(
	state:       rawptr,
	out_actions: []int,
	out_probs:   []f32,
	out_value:   ^f32,
	user_data:   rawptr,
) -> int {
	b := cast(^ag.GoBoard)state
	moves := ag.get_legal_moves_flat(b)
	defer delete(moves)
	pass_id := b.size * b.size
	total := f32(0)
	written := 0
	for m in moves {
		if written >= len(out_actions) {break}
		w := f32(1.0) / f32(m + 2)
		out_actions[written] = m
		out_probs[written] = w
		total += w
		written += 1
	}
	if written < len(out_actions) {
		// Mirror the prior test: pass weight uses (PASS_ACTION + 2) = (-1 + 2) = 1.
		w := f32(1.0)
		out_actions[written] = pass_id
		out_probs[written] = w
		total += w
		written += 1
	}
	for i in 0 ..< written {out_probs[i] /= total}
	out_value^ = 0.6
	return written
}

// Local helper that initialises the (tree, game) pair in-place at the caller's
// addresses. Tree is NOT safe to return by value — mcts.Tree.allocator embeds
// &arena, and the game vtable pointer stored in t.game must stay live for the
// tree's lifetime. So every test owns its own locals and calls this from there.
@(private = "file")
setup :: proc(tree: ^mcts.Tree, game: ^mcts.Game, cfg: mcts.Config, size: int = 9) {
	game^ = ag.go_game_vtable(size)
	state := ag.go_adapter_new_state(size)
	mcts.init(tree, game, state, cfg, 1)
}

@(test)
mcts_construction :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	testing.expect_value(t, mcts.tree_size(&tree), 1)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 0)
}

@(test)
mcts_single_simulation :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 1, uniform_evaluator)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 1)
	testing.expect(t, mcts.tree_size(&tree) >= 1)
}

@(test)
mcts_multiple_simulations :: proc(t: ^testing.T) {
	cfg := mcts.default_config(); cfg.c_puct = 1.0
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, cfg)
	defer mcts.destroy(&tree)

	mcts.run_simulations(&tree, 100, uniform_evaluator)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 100)
	testing.expect(t, mcts.tree_size(&tree) > 1)

	cv := mcts.get_child_visit_counts(&tree)
	defer delete(cv)
	testing.expect(t, len(cv) > 0)
	total: int
	for _, n in cv {total += n}
	testing.expect_value(t, total, 99)
}

@(test)
mcts_action_probs_temperature_1 :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 100, uniform_evaluator)
	probs := mcts.get_action_probabilities(&tree, 1.0)
	defer delete(probs)
	sum := f32(0)
	for _, p in probs {
		sum += p
		testing.expect(t, p >= 0)
		testing.expect(t, p <= 1)
	}
	testing.expectf(t, abs(sum - 1.0) < 0.01, "sum=%f", sum)
}

@(test)
mcts_action_probs_temperature_0 :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 100, biased_evaluator)
	probs := mcts.get_action_probabilities(&tree, 0.0)
	defer delete(probs)
	ones := 0
	for _, p in probs {if p == 1.0 {ones += 1}}
	testing.expect_value(t, ones, 1)
}

@(test)
mcts_select_action :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 100, uniform_evaluator)
	action := mcts.select_action(&tree, 1.0)

	b := cast(^ag.GoBoard)tree.working_state
	legal := ag.get_legal_moves_flat(b)
	defer delete(legal)
	is_valid := action == PASS_MCTS || slice.contains(legal[:], action)
	testing.expect(t, is_valid)
}

@(test)
mcts_deterministic_at_temp_0 :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 100, biased_evaluator)
	first := mcts.select_action(&tree, 0.0)
	for _ in 0 ..< 10 {
		testing.expect_value(t, mcts.select_action(&tree, 0.0), first)
	}
}

@(test)
mcts_respects_high_prior :: proc(t: ^testing.T) {
	cfg := mcts.default_config(); cfg.c_puct = 1.0
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, cfg)
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 200, biased_evaluator)
	visits := mcts.get_child_visit_counts(&tree)
	defer delete(visits)
	if len(visits) > 1 {
		max_action := -1
		max_visits := 0
		for action, v in visits {
			if v > max_visits {max_visits = v; max_action = action}
		}
		testing.expect(t, max_action < 10 || max_action == PASS_MCTS)
	}
}

@(test)
mcts_dirichlet_noise :: proc(t: ^testing.T) {
	cfg := mcts.default_config(); cfg.dirichlet_alpha = 0.3; cfg.dirichlet_weight = 0.25
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, cfg)
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 50, uniform_evaluator)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 50)
}

@(test)
mcts_q_in_bounds :: proc(t: ^testing.T) {
	cfg := mcts.default_config(); cfg.c_puct = 1.0
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, cfg)
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 100, uniform_evaluator)
	rq := mcts.get_root_q_value(&tree)
	testing.expect(t, rq >= 0)
	testing.expect(t, rq <= 1)
	cq := mcts.get_child_q_values(&tree)
	defer delete(cq)
	for _, q in cq {
		testing.expect(t, q >= 0)
		testing.expect(t, q <= 1)
	}
}

@(test)
mcts_handles_terminal_state :: proc(t: ^testing.T) {
	// Hand-build a terminal state (two consecutive passes) before mcts.init,
	// since mcts takes ownership of the state pointer.
	state_raw := ag.go_adapter_new_state(9)
	b := cast(^ag.GoBoard)state_raw
	ag.pass_move(b)
	ag.pass_move(b)
	testing.expect(t, ag.is_game_over(b))

	game := ag.go_game_vtable(9)
	tree: mcts.Tree
	mcts.init(&tree, &game, state_raw, mcts.default_config(), 1)
	defer mcts.destroy(&tree)
	mcts.run_simulations(&tree, 10, uniform_evaluator)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 10)
}

@(private = "file")
batched_uniform :: proc(
	states:      []rawptr,
	out_actions: [][]int,
	out_probs:   [][]f32,
	out_counts:  []int,
	out_values:  []f32,
	user_data:   rawptr,
) {
	for i in 0 ..< len(states) {
		v: f32
		out_counts[i] = uniform_evaluator(states[i], out_actions[i], out_probs[i], &v, nil)
		out_values[i] = v
	}
}

@(test)
mcts_batched_smoke :: proc(t: ^testing.T) {
	tree: mcts.Tree; game: mcts.Game
	setup(&tree, &game, mcts.default_config())
	defer mcts.destroy(&tree)
	mcts.run_simulations_batched(&tree, 50, 8, batched_uniform)
	testing.expect_value(t, mcts.get_root_visit_count(&tree), 50)
}
