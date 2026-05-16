package alpha_go_tests

import "core:slice"
import "core:testing"
import ag "../alpha_go"

@(private = "file")
uniform_evaluator :: proc(state: ^ag.GoBoard, user_data: rawptr) -> (policy: map[int]f32, value: f32) {
	moves := ag.get_legal_moves_flat(state)
	defer delete(moves)
	policy = make(map[int]f32, len(moves) + 1)
	uniform := 1.0 / f32(len(moves) + 1)
	for m in moves {policy[m] = uniform}
	policy[ag.PASS_ACTION] = uniform
	return policy, 0.5
}

@(private = "file")
biased_evaluator :: proc(state: ^ag.GoBoard, user_data: rawptr) -> (policy: map[int]f32, value: f32) {
	moves := ag.get_legal_moves_flat(state)
	defer delete(moves)
	policy = make(map[int]f32, len(moves) + 1)
	total := f32(0)
	for m in moves {
		w := 1.0 / f32(m + 2)
		policy[m] = w
		total += w
	}
	w := 1.0 / f32(ag.PASS_ACTION + 2)
	// PASS_ACTION = -1 so w = 1.0/1 = 1.0
	policy[ag.PASS_ACTION] = w
	total += w
	for a, _ in policy {
		policy[a] /= total
	}
	return policy, 0.6
}

@(test)
mcts_construction :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	testing.expect_value(t, ag.tree_size(&tree), 1)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 0)
}

@(test)
mcts_single_simulation :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 1, uniform_evaluator)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 1)
	testing.expect(t, ag.tree_size(&tree) >= 1)
}

@(test)
mcts_multiple_simulations :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	cfg.c_puct = 1.0
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)

	ag.run_simulations(&tree, 100, uniform_evaluator)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 100)
	testing.expect(t, ag.tree_size(&tree) > 1)

	cv := ag.get_child_visit_counts(&tree)
	defer delete(cv)
	testing.expect(t, len(cv) > 0)
	total: int
	for _, n in cv {total += n}
	testing.expect_value(t, total, 99)
}

@(test)
mcts_action_probs_temperature_1 :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 100, uniform_evaluator)
	probs := ag.get_action_probabilities(&tree, 1.0)
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
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 100, biased_evaluator)
	probs := ag.get_action_probabilities(&tree, 0.0)
	defer delete(probs)
	ones := 0
	for _, p in probs {if p == 1.0 {ones += 1}}
	testing.expect_value(t, ones, 1)
}

@(test)
mcts_select_action :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 100, uniform_evaluator)
	action := ag.select_action(&tree, 1.0)
	legal := ag.get_legal_moves_flat(&b)
	defer delete(legal)
	is_valid := action == ag.PASS_ACTION || slice.contains(legal[:], action)
	testing.expect(t, is_valid)
}

@(test)
mcts_deterministic_at_temp_0 :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 100, biased_evaluator)
	first := ag.select_action(&tree, 0.0)
	for _ in 0 ..< 10 {
		testing.expect_value(t, ag.select_action(&tree, 0.0), first)
	}
}

@(test)
mcts_respects_high_prior :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	cfg.c_puct = 1.0
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 200, biased_evaluator)
	visits := ag.get_child_visit_counts(&tree)
	defer delete(visits)
	if len(visits) > 1 {
		max_action := -1
		max_visits := 0
		for action, v in visits {
			if v > max_visits {max_visits = v; max_action = action}
		}
		testing.expect(t, max_action < 10 || max_action == ag.PASS_ACTION)
	}
}

@(test)
mcts_dirichlet_noise :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	cfg.dirichlet_alpha = 0.3
	cfg.dirichlet_weight = 0.25
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 50, uniform_evaluator)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 50)
}

@(test)
mcts_q_in_bounds :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	cfg.c_puct = 1.0
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 100, uniform_evaluator)
	rq := ag.get_root_q_value(&tree)
	testing.expect(t, rq >= 0)
	testing.expect(t, rq <= 1)
	cq := ag.get_child_q_values(&tree)
	defer delete(cq)
	for _, q in cq {
		testing.expect(t, q >= 0)
		testing.expect(t, q <= 1)
	}
}

@(test)
mcts_handles_terminal_state :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	ag.pass_move(&b)
	ag.pass_move(&b)
	testing.expect(t, ag.is_game_over(&b))

	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)
	ag.run_simulations(&tree, 10, uniform_evaluator)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 10)
}

@(test)
mcts_batched_smoke :: proc(t: ^testing.T) {
	b := ag.make_go_board(9); defer ag.destroy_go_board(&b)
	cfg := ag.default_mcts_config()
	tree := ag.make_mcts_tree(&b, cfg, 1)
	defer ag.destroy_mcts_tree(&tree)

	batched := proc(states: []^ag.GoBoard, out_policies: []map[int]f32, out_values: []f32, user_data: rawptr) {
		for i in 0 ..< len(states) {
			p, v := uniform_evaluator(states[i], nil)
			out_policies[i] = p
			out_values[i] = v
		}
	}

	ag.run_simulations_batched(&tree, 50, 8, batched)
	testing.expect_value(t, ag.get_root_visit_count(&tree), 50)
}
