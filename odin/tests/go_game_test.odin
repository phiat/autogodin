package alpha_go_tests

import "core:testing"
import ag "../alpha_go"

@(test)
goboard_construction :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	testing.expect_value(t, b.size, 9)
	testing.expect_value(t, b.to_play, ag.BLACK)
	testing.expect_value(t, b.consecutive_passes, 0)
	testing.expect_value(t, b.move_count, 0)
	testing.expect(t, !ag.is_game_over(&b))
	for i in 0 ..< 81 {
		testing.expect_value(t, ag.at_flat(&b, i), ag.EMPTY)
	}
}

@(test)
basic_stone_placement :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	testing.expect(t, ag.play(&b, 4, 4))
	testing.expect_value(t, ag.at(&b, 4, 4), ag.BLACK)
	testing.expect_value(t, b.to_play, ag.WHITE)
	testing.expect_value(t, b.move_count, 1)

	testing.expect(t, ag.play(&b, 4, 5))
	testing.expect_value(t, ag.at(&b, 4, 5), ag.WHITE)
	testing.expect_value(t, b.to_play, ag.BLACK)
	testing.expect_value(t, b.move_count, 2)

	testing.expect(t, !ag.is_legal(&b, 4, 4))
	testing.expect(t, !ag.play(&b, 4, 4))
}

@(test)
single_stone_capture :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	ag.play(&b, 0, 1) // Black
	ag.play(&b, 0, 0) // White (to be captured)
	ag.play(&b, 1, 0) // Black (completes capture)
	testing.expect_value(t, ag.at(&b, 0, 0), ag.EMPTY)
}

@(test)
group_capture :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	ag.play(&b, 0, 0) // Black
	ag.play(&b, 1, 0) // White
	ag.play(&b, 0, 1) // Black
	ag.play(&b, 1, 1) // White
	ag.play(&b, 2, 0) // Black
	ag.play(&b, 8, 8) // White elsewhere
	ag.play(&b, 2, 1) // Black
	ag.pass_move(&b) // White passes
	ag.play(&b, 1, 2) // Black — completes capture
	testing.expect_value(t, ag.at(&b, 1, 0), ag.EMPTY)
	testing.expect_value(t, ag.at(&b, 1, 1), ag.EMPTY)
}

@(test)
ko_rule :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	ag.play(&b, 0, 1) // Black
	ag.play(&b, 0, 2) // White
	ag.play(&b, 1, 0) // Black
	ag.play(&b, 1, 3) // White
	ag.play(&b, 1, 2) // Black
	ag.play(&b, 2, 2) // White
	ag.play(&b, 2, 1) // Black
	ag.play(&b, 1, 1) // White captures Black at (1,2)

	testing.expect_value(t, ag.at(&b, 1, 2), ag.EMPTY)
	testing.expect(t, b.ko_point != ag.NO_KO)
	testing.expect(t, !ag.is_legal(&b, 1, 2))
}

@(test)
positional_superko :: proc(t: ^testing.T) {
	b := ag.make_go_board(5)
	defer ag.destroy_go_board(&b)

	arr: [25]i8
	arr[0 * 5 + 1] = ag.EMPTY
	arr[1 * 5 + 0] = ag.BLACK
	arr[2 * 5 + 1] = ag.BLACK
	arr[1 * 5 + 2] = ag.BLACK
	arr[1 * 5 + 1] = ag.WHITE
	ag.set_from_array(&b, arr[:], ag.BLACK)

	testing.expect(t, ag.play(&b, 0, 1))
	testing.expect_value(t, ag.at(&b, 1, 1), ag.EMPTY)
	testing.expect(t, !ag.is_legal(&b, 1, 1))
}

@(test)
single_stone_suicide_illegal :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	ag.play(&b, 0, 1)
	ag.play(&b, 8, 8)
	ag.play(&b, 1, 0)
	testing.expect(t, !ag.is_legal(&b, 0, 0))
}

@(test)
multi_stone_suicide_illegal :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	ag.play(&b, 0, 2); ag.play(&b, 8, 8)
	ag.play(&b, 1, 1); ag.play(&b, 8, 7)
	ag.play(&b, 1, 3); ag.play(&b, 7, 8)
	ag.play(&b, 2, 1); ag.play(&b, 7, 7)
	ag.play(&b, 2, 3); ag.play(&b, 6, 8)
	ag.play(&b, 3, 2)
	testing.expect(t, ag.play(&b, 1, 2))
	testing.expect(t, ag.play(&b, 8, 0))
	testing.expect(t, !ag.is_legal(&b, 2, 2))
	testing.expect(t, !ag.play(&b, 2, 2))
	testing.expect_value(t, ag.at(&b, 1, 2), ag.WHITE)
	testing.expect_value(t, ag.at(&b, 2, 2), ag.EMPTY)
}

@(test)
capture_is_not_suicide :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)
	ag.play(&b, 0, 1) // Black
	ag.play(&b, 0, 0) // White (will be captured)
	ag.play(&b, 1, 0) // Black captures
	testing.expect_value(t, ag.at(&b, 0, 0), ag.EMPTY)
}

@(test)
pass_and_game_end :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	testing.expect_value(t, b.consecutive_passes, 0)
	testing.expect(t, !ag.is_game_over(&b))

	ag.pass_move(&b)
	testing.expect_value(t, b.consecutive_passes, 1)
	testing.expect_value(t, b.to_play, ag.WHITE)
	testing.expect(t, !ag.is_game_over(&b))

	ag.pass_move(&b)
	testing.expect_value(t, b.consecutive_passes, 2)
	testing.expect(t, ag.is_game_over(&b))
}

@(test)
non_consecutive_passes_do_not_end :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)
	ag.pass_move(&b)
	testing.expect_value(t, b.consecutive_passes, 1)
	testing.expect(t, !ag.is_game_over(&b))

	testing.expect(t, ag.play(&b, 4, 4))
	testing.expect_value(t, b.consecutive_passes, 0)
	testing.expect(t, !ag.is_game_over(&b))

	ag.pass_move(&b)
	testing.expect_value(t, b.consecutive_passes, 1)
	testing.expect(t, !ag.is_game_over(&b))

	ag.pass_move(&b)
	testing.expect_value(t, b.consecutive_passes, 2)
	testing.expect(t, ag.is_game_over(&b))
}

@(test)
legal_moves_generation :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	moves := ag.get_legal_moves_flat(&b)
	defer delete(moves)
	testing.expect_value(t, len(moves), 81)

	ag.play(&b, 4, 4)
	moves2 := ag.get_legal_moves_flat(&b)
	defer delete(moves2)
	testing.expect_value(t, len(moves2), 80)
}

@(test)
scoring_empty_board :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)

	s := ag.score(&b)
	testing.expectf(t, abs(s - (-7.5)) < 0.01, "expected -7.5, got %f", s)
	testing.expect_value(t, ag.get_winner(&b), ag.WHITE)
}

@(test)
custom_komi :: proc(t: ^testing.T) {
	b := ag.make_go_board(9, 5.5)
	defer ag.destroy_go_board(&b)
	testing.expect_value(t, b.komi, f32(5.5))
	s := ag.score(&b)
	testing.expectf(t, abs(s - (-5.5)) < 0.01, "expected -5.5, got %f", s)
}

@(test)
default_komi :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)
	testing.expect_value(t, b.komi, f32(7.5))
}

@(test)
komi_preserved_on_copy :: proc(t: ^testing.T) {
	b := ag.make_go_board(9, 5.5)
	defer ag.destroy_go_board(&b)
	c := ag.clone_go_board(&b)
	defer ag.destroy_go_board(&c)
	testing.expect_value(t, c.komi, f32(5.5))
}

@(test)
board_copy :: proc(t: ^testing.T) {
	b := ag.make_go_board(9)
	defer ag.destroy_go_board(&b)
	ag.play(&b, 4, 4)
	ag.play(&b, 4, 5)

	c := ag.clone_go_board(&b)
	defer ag.destroy_go_board(&c)

	testing.expect_value(t, ag.at(&c, 4, 4), ag.BLACK)
	testing.expect_value(t, ag.at(&c, 4, 5), ag.WHITE)
	testing.expect_value(t, c.to_play, b.to_play)
	testing.expect_value(t, c.move_count, b.move_count)

	ag.play(&c, 0, 0)
	testing.expect_value(t, ag.at(&b, 0, 0), ag.EMPTY)
	testing.expect_value(t, ag.at(&c, 0, 0), ag.BLACK)
}
