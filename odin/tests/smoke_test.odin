package alpha_go_tests

import "core:testing"
import alpha_go "../alpha_go"

@(test)
version_nonempty :: proc(t: ^testing.T) {
	testing.expect(t, len(alpha_go.VERSION) > 0)
}

@(test)
ffi_smoke_returns_42 :: proc(t: ^testing.T) {
	testing.expect_value(t, alpha_go.ffi_smoke(), i32(42))
}
