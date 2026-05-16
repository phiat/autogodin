package alpha_go

VERSION :: "0.2.0"

@(export, link_name="alphago_odin_version")
ffi_version :: proc "c" () -> cstring {
	return cstring(VERSION)
}

@(export, link_name="alphago_odin_smoke")
ffi_smoke :: proc "c" () -> i32 {
	return 42
}
