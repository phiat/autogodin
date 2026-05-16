package alpha_go

import "base:runtime"
import "core:math"
import "core:math/rand"
import "core:mem/virtual"

PASS_ACTION :: -1

MCTSConfig :: struct {
	c_puct:              f32,
	lambda:              f32,   // 0 = pure value, 1 = pure rollout
	dirichlet_alpha:     f32,   // 0 = no noise
	dirichlet_weight:    f32,
	temperature:         f32,
	max_depth:           int,   // tree + rollout combined budget
	rollout_temperature: f32,

	pcr_sims:  []int,
	pcr_probs: []f32,
}

default_mcts_config :: proc() -> MCTSConfig {
	return MCTSConfig{
		c_puct              = 1.0,
		lambda              = 0.0,
		dirichlet_alpha     = 0.0,
		dirichlet_weight    = 0.25,
		temperature         = 1.0,
		max_depth           = 100,
		rollout_temperature = 1.0,
	}
}

MCTSNode :: struct {
	N:                  int,
	N_virt:             int,
	Q:                  f32,
	first_eval_value:   f32,
	has_eval:           bool,
	parent_idx:         int,         // -1 for root
	player_at_parent:   i8,          // 0=BLACK, 1=WHITE (MCTS convention; differs from board)
	depth:              int,
	move_played:        int,         // -1 for root; PASS_ACTION for pass; else flat board index
	children:           map[int]int, // action -> child node_idx
	logP_A:             map[int]f32, // action -> log prior
}

MCTSTree :: struct {
	nodes:     [dynamic]MCTSNode,
	config:    MCTSConfig,
	rng_state: rand.Default_Random_State,
	// Per-tree growing arena. All tree-owned allocations (nodes, per-node
	// children/logP_A maps, working_board's seen_hashes + board slice, capture
	// stack) live here; destroy_mcts_tree frees the whole arena in one shot.
	arena:     virtual.Arena,
	allocator: runtime.Allocator,

	// Single board mutated during traversal: do_move down, undo_move up. This
	// replaces the per-node inline GoBoard that piece 3 of autogodin-4rw
	// removed. INVARIANT: working_board is at the root's state whenever no
	// traversal is in progress.
	working_board: GoBoard,
	capture_stack: [dynamic]CaptureRecord,
}

// Bind t's RNG state into the current context. Call once at the start of any
// public proc that needs randomness; transitive callees pick it up via context.
@(private)
use_tree_rng :: proc(t: ^MCTSTree) {
	context.random_generator = rand.default_random_generator(&t.rng_state)
}

// Evaluator: state -> (policy[action]=prob, value in [0,1] from state.to_play perspective)
EvaluatorFn :: #type proc(state: ^GoBoard, user_data: rawptr) -> (policy: map[int]f32, value: f32)

BatchedEvaluatorFn :: #type proc(
	states:        []^GoBoard,
	out_policies:  []map[int]f32, // caller-allocated outer slice; callback fills maps
	out_values:    []f32,
	user_data:     rawptr,
)

// MCTS-perspective player from board color: BLACK -> 0, WHITE -> 1.
@(private = "file")
mcts_player :: proc(c: i8) -> i8 {
	return 0 if c == BLACK else 1
}

@(private = "file")
mcts_color :: proc(p: i8) -> i8 {
	return BLACK if p == 0 else WHITE
}

// Initializes `t` in-place. MCTSTree MUST be init'd at its final address
// (not returned by value) because t.allocator holds a pointer to t.arena;
// any move/copy of the struct would dangle the arena pointer.
init_mcts_tree :: proc(t: ^MCTSTree, root_state: ^GoBoard, config: MCTSConfig, seed: u64 = 0) {
	t^ = {}
	t.config = config
	_ = virtual.arena_init_growing(&t.arena, 8 << 20)
	t.allocator = virtual.arena_allocator(&t.arena)

	t.nodes = make([dynamic]MCTSNode, 0, 64, t.allocator)
	t.rng_state = rand.create(seed if seed != 0 else 0xC0FFEE_DECADE)
	t.working_board = clone_go_board(root_state, t.allocator)
	t.capture_stack = make([dynamic]CaptureRecord, 0, 64, t.allocator)

	root := MCTSNode{
		parent_idx       = -1,
		player_at_parent = 1 if root_state.to_play == BLACK else 0,
		depth            = 0,
		move_played      = -1,
		children         = make(map[int]int, 8, t.allocator),
		logP_A           = make(map[int]f32, 8, t.allocator),
	}
	append(&t.nodes, root)
}

destroy_mcts_tree :: proc(t: ^MCTSTree) {
	// One wholesale free of every tree-internal allocation. No need to walk nodes.
	virtual.arena_destroy(&t.arena)
	t^ = {}
}

create_node :: proc(t: ^MCTSTree, move_played: int, parent_idx: int, player_at_parent: i8) -> int {
	idx := len(t.nodes)
	depth := t.nodes[parent_idx].depth + 1 if parent_idx >= 0 else 0
	n := MCTSNode{
		parent_idx       = parent_idx,
		player_at_parent = player_at_parent,
		depth            = depth,
		move_played      = move_played,
		// Pre-create the per-node maps in the tree arena so any subsequent
		// insert lands there, not in the caller's context.allocator.
		children         = make(map[int]int, 8, t.allocator),
		logP_A           = make(map[int]f32, 8, t.allocator),
	}
	append(&t.nodes, n)
	return idx
}

@(private = "file")
log_safe :: proc(x: f32) -> f32 {
	return math.ln(x + 1e-8)
}

compute_puct_scores :: proc(t: ^MCTSTree, node_idx: int) -> map[int]f32 {
	node := &t.nodes[node_idx]
	total_visits := 0
	for action, _ in node.logP_A {
		if child_idx, ok := node.children[action]; ok {
			total_visits += t.nodes[child_idx].N
		}
	}
	sqrt_total := math.sqrt(f32(total_visits) + 1.0)

	scores := make(map[int]f32, len(node.logP_A))
	for action, log_prior in node.logP_A {
		prior := math.exp(log_prior)
		q_value := f32(0)
		n_visits := 0
		if child_idx, ok := node.children[action]; ok {
			q_value = t.nodes[child_idx].Q
			n_visits = t.nodes[child_idx].N
		}
		u_value := t.config.c_puct * prior * sqrt_total / (1.0 + f32(n_visits))
		scores[action] = q_value + u_value
	}
	return scores
}

select_action_puct :: proc(t: ^MCTSTree, node_idx: int) -> int {
	scores := compute_puct_scores(t, node_idx)
	defer delete(scores)
	best_action := -1
	best_score := f32(min(f32))
	// Tie-break: lowest action wins (deterministic). Iterate sorted not needed here
	// since the C++ uses unordered_map and ties are also non-deterministic.
	for action, score in scores {
		if score > best_score {
			best_score = score
			best_action = action
		}
	}
	return best_action
}

// Marsaglia & Tsang gamma sampler (shape >= 1). For 0 < alpha < 1 use the
// boost: sample G(alpha+1) then multiply by U^(1/alpha). Uses context RNG.
@(private = "file")
gamma_sample :: proc(alpha: f32) -> f32 {
	if alpha < 1.0 {
		g := gamma_sample(alpha + 1.0)
		u := rand.float32()
		return g * math.pow(u, 1.0 / alpha)
	}
	d := alpha - 1.0 / 3.0
	c := 1.0 / math.sqrt(9.0 * d)
	for {
		x := rand.float32_normal(0, 1)
		v := 1.0 + c * x
		if v <= 0.0 {continue}
		v = v * v * v
		u := rand.float32()
		if u < 1.0 - 0.0331 * x * x * x * x {return d * v}
		if math.ln(u) < 0.5 * x * x + d * (1.0 - v + math.ln(v)) {return d * v}
	}
}

add_dirichlet_noise :: proc(t: ^MCTSTree, alpha, weight: f32) {
	root := &t.nodes[0]
	if len(root.logP_A) == 0 {return}

	actions := make([dynamic]int, 0, len(root.logP_A), context.temp_allocator)
	defer delete(actions)
	for action, _ in root.logP_A {
		append(&actions, action)
	}

	noise := make([]f32, len(actions), context.temp_allocator)
	defer delete(noise, context.temp_allocator)
	sum := f32(0)
	for i in 0 ..< len(actions) {
		noise[i] = gamma_sample(alpha)
		sum += noise[i]
	}
	for i in 0 ..< len(noise) {
		noise[i] /= sum
	}

	for i in 0 ..< len(actions) {
		action := actions[i]
		log_prior := root.logP_A[action]
		prior := math.exp(log_prior)
		noisy := (1.0 - weight) * prior + weight * noise[i]
		root.logP_A[action] = log_safe(noisy)
	}
}

sample_action_from_policy :: proc(
	t: ^MCTSTree,
	policy: map[int]f32,
	temperature: f32,
) -> int {
	if len(policy) == 0 {return PASS_ACTION}

	actions := make([]int, len(policy), context.temp_allocator)
	defer delete(actions, context.temp_allocator)
	probs := make([]f32, len(policy), context.temp_allocator)
	defer delete(probs, context.temp_allocator)

	i := 0
	for action, prob in policy {
		actions[i] = action
		probs[i] = prob
		i += 1
	}

	if temperature != 1.0 && temperature > 0 {
		max_logit := f32(min(f32))
		logits := make([]f32, len(probs), context.temp_allocator)
		defer delete(logits, context.temp_allocator)
		for k in 0 ..< len(probs) {
			logits[k] = math.ln(probs[k] + 1e-8) / temperature
			if logits[k] > max_logit {max_logit = logits[k]}
		}
		sum := f32(0)
		for k in 0 ..< len(probs) {
			probs[k] = math.exp(logits[k] - max_logit)
			sum += probs[k]
		}
		for k in 0 ..< len(probs) {probs[k] /= sum}
	}

	r := rand.float32()
	cum := f32(0)
	for k in 0 ..< len(probs) {
		cum += probs[k]
		if r < cum {return actions[k]}
	}
	return actions[len(actions) - 1]
}

fast_rollout :: proc(
	t: ^MCTSTree,
	start_state: ^GoBoard,
	player_perspective: i8,
	remaining_depth: int,
	evaluator: EvaluatorFn,
	user_data: rawptr,
) -> f32 {
	current := clone_go_board(start_state, context.temp_allocator)
	defer destroy_go_board(&current)

	depth := 0
	for !is_game_over(&current) && depth < remaining_depth {
		policy, _ := evaluator(&current, user_data)
		action: int
		if len(policy) == 0 {
			pass_move(&current)
			action = PASS_ACTION
		} else {
			action = sample_action_from_policy(t, policy, t.config.rollout_temperature)
			if action == PASS_ACTION {
				pass_move(&current)
			} else {
				row, col := row_col(&current, action)
				if !play(&current, row, col) {
					pass_move(&current)
					action = PASS_ACTION
				}
			}
		}
		delete(policy)
		depth += 1
	}

	if is_game_over(&current) {
		winner := get_winner(&current)
		if winner == 0 {return 0.5}
		pc := mcts_color(player_perspective)
		return 1.0 if winner == pc else 0.0
	}
	_, v := evaluator(&current, user_data)
	cp := mcts_player(current.to_play)
	if cp != player_perspective {v = 1.0 - v}
	return v
}

// Single-path simulation. Returns U from node_idx.player_at_parent perspective.
//
// INVARIANT: caller must have t.working_board positioned at node_idx's state
// before entering. On return, working_board is restored to that same state.
perform_playout :: proc(t: ^MCTSTree, node_idx: int, evaluator: EvaluatorFn, user_data: rawptr) -> f32 {
	// Don't store references — t.nodes may reallocate. Index-based throughout.
	player_perspective := t.nodes[node_idx].player_at_parent

	U: f32

	if is_game_over(&t.working_board) {
		winner := get_winner(&t.working_board)
		if winner == 0 {
			U = 0.5
		} else {
			pc := mcts_color(player_perspective)
			U = 1.0 if winner == pc else 0.0
		}
	} else if t.nodes[node_idx].N == 0 {
		policy, v_theta := evaluator(&t.working_board, user_data)
		for action, prob in policy {
			t.nodes[node_idx].logP_A[action] = log_safe(prob)
		}
		delete(policy)

		current_player := mcts_player(t.working_board.to_play)
		if current_player != player_perspective {v_theta = 1.0 - v_theta}

		t.nodes[node_idx].first_eval_value = v_theta
		t.nodes[node_idx].has_eval = true

		if t.config.lambda > 0 {
			current_depth := t.nodes[node_idx].depth
			remaining := t.config.max_depth - current_depth
			if remaining > 0 {
				z_L := fast_rollout(t, &t.working_board, player_perspective, remaining, evaluator, user_data)
				U = (1.0 - t.config.lambda) * v_theta + t.config.lambda * z_L
			} else {
				U = v_theta
			}
		} else {
			U = v_theta
		}
	} else {
		action := select_action_puct(t, node_idx)

		// cp captures parent's mover BEFORE we mutate working_board.
		cp := mcts_player(t.working_board.to_play)

		// Descend: mutate working_board, recurse, then undo.
		delta := do_move(&t.working_board, action, &t.capture_stack)

		if _, ok := t.nodes[node_idx].children[action]; !ok {
			child_idx := create_node(t, action, node_idx, cp)
			// After create_node, t.nodes may have reallocated. Re-fetch.
			t.nodes[node_idx].children[action] = child_idx
		}
		child_idx := t.nodes[node_idx].children[action]
		child_value := perform_playout(t, child_idx, evaluator, user_data)
		U = 1.0 - child_value

		undo_move(&t.working_board, delta, &t.capture_stack)
	}

	t.nodes[node_idx].N += 1
	t.nodes[node_idx].Q = t.nodes[node_idx].Q + (U - t.nodes[node_idx].Q) / f32(t.nodes[node_idx].N)
	return U
}

run_simulations :: proc(t: ^MCTSTree, num_simulations: int, evaluator: EvaluatorFn, user_data: rawptr = nil) {
	use_tree_rng(t)
	// Reset thread-local temp_allocator on entry and exit. Many helpers (do_move's
	// get_group_and_liberties, is_legal_flat's clone_for_sim, etc.) allocate to
	// temp_allocator and rely on someone above to wipe it. The default
	// runtime.default_temp_allocator is a growing arena — it never auto-resets,
	// so without this every batch of sims leaks ~30 MB.
	free_all(context.temp_allocator)
	defer free_all(context.temp_allocator)
	n_sims := num_simulations
	if len(t.config.pcr_sims) > 0 {
		// Categorical sample from pcr_probs
		r := rand.float32()
		cum := f32(0)
		pick := len(t.config.pcr_sims) - 1
		for i in 0 ..< len(t.config.pcr_probs) {
			cum += t.config.pcr_probs[i]
			if r < cum {pick = i; break}
		}
		n_sims = t.config.pcr_sims[pick]
	}

	// working_board is at root (init_mcts_tree's invariant). Eval the root.
	policy, _ := evaluator(&t.working_board, user_data)
	for action, prob in policy {
		t.nodes[0].logP_A[action] = log_safe(prob)
	}
	delete(policy)

	if t.config.dirichlet_alpha > 0 {
		add_dirichlet_noise(t, t.config.dirichlet_alpha, t.config.dirichlet_weight)
	}

	for _ in 0 ..< n_sims {
		perform_playout(t, 0, evaluator, user_data)
		// perform_playout preserves working_board, so we stay at root here.
	}
}

// -------- Leaf-parallel MCTS with virtual loss --------
//
// Post foundation-piece-3, MCTSNode no longer stores per-node GoBoard, so the
// batched gather walks t.working_board with do_move/undo. Each evaluated leaf
// requires a state snapshot (the batch needs all states simultaneously); we
// keep those in a pre-reserved [dynamic]GoBoard sized to leaf_batch_size so
// pointers into it stay stable for the eval call.

@(private = "file")
PendingLeaf :: struct {
	path:           [dynamic]int,
	leaf_idx:       int,
	leaf_to_play:   i8,   // working_board.to_play at the leaf, captured at gather time
	is_terminal:    bool,
	terminal_U:     f32,
	eval_slot:      int,  // -1 if terminal
}

run_simulations_batched :: proc(
	t: ^MCTSTree,
	num_simulations: int,
	leaf_batch_size: int,
	evaluator: BatchedEvaluatorFn,
	user_data: rawptr = nil,
) {
	use_tree_rng(t)
	// See run_simulations for rationale. Per-batch temp_allocator pressure here
	// is even larger (eval state snapshots), so we wipe between batches too.
	free_all(context.temp_allocator)
	defer free_all(context.temp_allocator)
	n_sims := num_simulations
	if len(t.config.pcr_sims) > 0 {
		r := rand.float32()
		cum := f32(0)
		pick := len(t.config.pcr_sims) - 1
		for i in 0 ..< len(t.config.pcr_probs) {
			cum += t.config.pcr_probs[i]
			if r < cum {pick = i; break}
		}
		n_sims = t.config.pcr_sims[pick]
	}

	// Evaluate root if not yet populated. working_board is at root.
	if len(t.nodes[0].logP_A) == 0 {
		states := []^GoBoard{&t.working_board}
		policies := make([]map[int]f32, 1, context.temp_allocator)
		defer delete(policies, context.temp_allocator)
		values := make([]f32, 1, context.temp_allocator)
		defer delete(values, context.temp_allocator)
		evaluator(states, policies, values, user_data)
		for action, prob in policies[0] {
			t.nodes[0].logP_A[action] = log_safe(prob)
		}
		delete(policies[0])
	}
	if t.config.dirichlet_alpha > 0 {
		add_dirichlet_noise(t, t.config.dirichlet_alpha, t.config.dirichlet_weight)
	}

	completed := 0
	for completed < n_sims {
		// Per-batch wipe of temp_allocator so peak stays bounded by one batch's
		// worth of temp churn rather than the whole run. Declared FIRST so it
		// fires LAST (LIFO), after the other in-scope cleanup defers below have
		// run on data that lives in temp.
		defer free_all(context.temp_allocator)

		target := min(leaf_batch_size, n_sims - completed)
		pending := make([dynamic]PendingLeaf, 0, target, context.temp_allocator)
		// Pre-reserve eval_state_storage so pointers into it stay stable across
		// appends within this batch.
		eval_state_storage := make([dynamic]GoBoard, 0, target, context.temp_allocator)
		eval_states := make([dynamic]^GoBoard, 0, target, context.temp_allocator)
		defer {
			for &p in pending {delete(p.path)}
			delete(pending)
			for i in 0 ..< len(eval_state_storage) {
				destroy_go_board(&eval_state_storage[i])
			}
			delete(eval_state_storage)
			delete(eval_states)
		}

		for _ in 0 ..< target {
			path := make([dynamic]int, 0, 8)
			deltas := make([dynamic]MoveDelta, 0, 8, context.temp_allocator)
			defer delete(deltas)

			node_idx := 0
			append(&path, node_idx)

			pl: PendingLeaf
			pl.eval_slot = -1

			for {
				if is_game_over(&t.working_board) {
					winner := get_winner(&t.working_board)
					persp := t.nodes[node_idx].player_at_parent
					U: f32
					if winner == 0 {
						U = 0.5
					} else {
						pc := mcts_color(persp)
						U = 1.0 if winner == pc else 0.0
					}
					pl.is_terminal = true
					pl.terminal_U = U
					pl.leaf_idx = node_idx
					pl.leaf_to_play = t.working_board.to_play
					break
				}
				if len(t.nodes[node_idx].logP_A) == 0 {
					pl.leaf_idx = node_idx
					pl.leaf_to_play = t.working_board.to_play
					pl.eval_slot = len(eval_states)
					append(&eval_state_storage, clone_go_board(&t.working_board, context.temp_allocator))
					append(&eval_states, &eval_state_storage[len(eval_state_storage)-1])
					break
				}

				// PUCT-with-virtual-loss
				node := &t.nodes[node_idx]
				total_visits := 0
				for action, _ in node.logP_A {
					if ci, ok := node.children[action]; ok {
						total_visits += t.nodes[ci].N + t.nodes[ci].N_virt
					}
				}
				sqrt_total := math.sqrt(f32(total_visits) + 1.0)
				best_action := -1
				best_score := f32(min(f32))
				for action, log_prior in node.logP_A {
					prior := math.exp(log_prior)
					q := f32(0); n := 0; nv := 0
					if ci, ok := node.children[action]; ok {
						q = t.nodes[ci].Q
						n = t.nodes[ci].N
						nv = t.nodes[ci].N_virt
					}
					n_eff := f32(n + nv)
					q_eff := (q * f32(n)) / n_eff if n_eff > 0 else 0.0
					u := t.config.c_puct * prior * sqrt_total / (1.0 + n_eff)
					score := q_eff + u
					if score > best_score {best_score = score; best_action = action}
				}

				action := best_action
				cp := mcts_player(t.working_board.to_play)
				delta := do_move(&t.working_board, action, &t.capture_stack)
				append(&deltas, delta)

				if _, ok := t.nodes[node_idx].children[action]; !ok {
					child_idx := create_node(t, action, node_idx, cp)
					t.nodes[node_idx].children[action] = child_idx
				}
				child_idx := t.nodes[node_idx].children[action]
				append(&path, child_idx)
				node_idx = child_idx
			}

			pl.path = path
			append(&pending, pl)
			for idx in pl.path {t.nodes[idx].N_virt += 1}

			// Undo all moves to return working_board to the root state for the
			// next gather iteration (and for the post-batch invariant).
			#reverse for d in deltas {
				undo_move(&t.working_board, d, &t.capture_stack)
			}
		}

		eval_policies: []map[int]f32
		eval_values: []f32
		if len(eval_states) > 0 {
			eval_policies = make([]map[int]f32, len(eval_states), context.temp_allocator)
			eval_values = make([]f32, len(eval_states), context.temp_allocator)
			evaluator(eval_states[:], eval_policies, eval_values, user_data)
		}
		defer if len(eval_states) > 0 {
			for &p in eval_policies {delete(p)}
			delete(eval_policies, context.temp_allocator)
			delete(eval_values, context.temp_allocator)
		}

		for &pl in pending {
			U: f32
			if pl.is_terminal {
				U = pl.terminal_U
			} else {
				policy := eval_policies[pl.eval_slot]
				v_theta := eval_values[pl.eval_slot]
				if len(t.nodes[pl.leaf_idx].logP_A) == 0 {
					for action, prob in policy {
						t.nodes[pl.leaf_idx].logP_A[action] = log_safe(prob)
					}
				}
				persp := t.nodes[pl.leaf_idx].player_at_parent
				curp := mcts_player(pl.leaf_to_play)
				U = (1.0 - v_theta) if curp != persp else v_theta
				if !t.nodes[pl.leaf_idx].has_eval {
					t.nodes[pl.leaf_idx].first_eval_value = U
					t.nodes[pl.leaf_idx].has_eval = true
				}
			}
			// Backup
			#reverse for idx in pl.path {
				t.nodes[idx].N_virt -= 1
				t.nodes[idx].N += 1
				t.nodes[idx].Q += (U - t.nodes[idx].Q) / f32(t.nodes[idx].N)
				U = 1.0 - U
			}
			completed += 1
		}
	}
}

// -------- Action selection / readout helpers --------

get_action_probabilities :: proc(t: ^MCTSTree, temperature: f32 = 1.0) -> map[int]f32 {
	root := &t.nodes[0]

	if len(root.children) == 0 {
		probs := make(map[int]f32, len(root.logP_A))
		uniform := 1.0 / f32(len(root.logP_A)) if len(root.logP_A) > 0 else 0
		for action, _ in root.logP_A {
			probs[action] = uniform
		}
		return probs
	}

	visit_counts := make(map[int]int, len(root.children), context.temp_allocator)
	defer delete(visit_counts)
	for action, child_idx in root.children {
		visit_counts[action] = t.nodes[child_idx].N
	}

	if temperature == 0 {
		best_action := -1
		best_visits := -1
		for action, v in visit_counts {
			if v > best_visits {best_visits = v; best_action = action}
		}
		probs := make(map[int]f32, len(visit_counts))
		for action in visit_counts {
			probs[action] = 1.0 if action == best_action else 0.0
		}
		return probs
	}

	out := make(map[int]f32, len(visit_counts))
	total := f32(0)
	for action, v in visit_counts {
		val := math.pow(f32(v), 1.0 / temperature)
		out[action] = val
		total += val
	}
	if total == 0 {
		uniform := 1.0 / f32(len(out))
		for action in out {out[action] = uniform}
		return out
	}
	for action, v in out {out[action] = v / total}
	return out
}

select_action :: proc(t: ^MCTSTree, temperature: f32 = 1.0) -> int {
	use_tree_rng(t)
	probs := get_action_probabilities(t, temperature)
	defer delete(probs)

	if temperature == 0 {
		best_action := -1
		best_p := f32(-1)
		for action, p in probs {
			if p > best_p {best_p = p; best_action = action}
		}
		return best_action
	}

	r := rand.float32()
	cum := f32(0)
	last_action := -1
	for action, p in probs {
		last_action = action
		cum += p
		if r < cum {return action}
	}
	return last_action
}

tree_size :: proc(t: ^MCTSTree) -> int {return len(t.nodes)}

get_root_visit_count :: proc(t: ^MCTSTree) -> int {return t.nodes[0].N}

get_root_q_value :: proc(t: ^MCTSTree) -> f32 {return t.nodes[0].Q}

get_child_visit_counts :: proc(t: ^MCTSTree, allocator := context.allocator) -> map[int]int {
	out := make(map[int]int, len(t.nodes[0].children), allocator)
	for action, ci in t.nodes[0].children {
		out[action] = t.nodes[ci].N
	}
	return out
}

get_child_q_values :: proc(t: ^MCTSTree, allocator := context.allocator) -> map[int]f32 {
	out := make(map[int]f32, len(t.nodes[0].children), allocator)
	for action, ci in t.nodes[0].children {
		out[action] = t.nodes[ci].Q
	}
	return out
}

get_child_first_eval_values :: proc(t: ^MCTSTree, allocator := context.allocator) -> map[int]f32 {
	out := make(map[int]f32, len(t.nodes[0].children), allocator)
	for action, ci in t.nodes[0].children {
		if t.nodes[ci].has_eval {
			out[action] = t.nodes[ci].first_eval_value
		}
	}
	return out
}

get_root_policy_priors :: proc(t: ^MCTSTree, allocator := context.allocator) -> map[int]f32 {
	out := make(map[int]f32, len(t.nodes[0].logP_A), allocator)
	for action, log_prior in t.nodes[0].logP_A {
		out[action] = math.exp(log_prior)
	}
	return out
}

get_child_max_subtree_depths :: proc(t: ^MCTSTree, allocator := context.allocator) -> map[int]int {
	out := make(map[int]int, len(t.nodes[0].children), allocator)
	stack := make([dynamic]int, 0, 64, context.temp_allocator)
	defer delete(stack)
	for action, ci in t.nodes[0].children {
		max_depth := t.nodes[ci].depth
		clear(&stack)
		append(&stack, ci)
		for len(stack) > 0 {
			cur := pop(&stack)
			if t.nodes[cur].depth > max_depth {max_depth = t.nodes[cur].depth}
			for _, c in t.nodes[cur].children {
				append(&stack, c)
			}
		}
		out[action] = max_depth
	}
	return out
}
