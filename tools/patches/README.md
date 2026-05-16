# Upstream patches

Small patches we keep against upstream autogo until they merge or are no
longer needed. Apply by cd'ing into a fresh autogo clone and running the
relevant script.

## upstream_build_cpp_fix.py

Drops the hardcoded `libpython3.10.so` from `scripts/build_cpp.sh`. Without
this, `bash scripts/build_cpp.sh` fails on any host whose `uv`-resolved
Python isn't exactly 3.10 (uv currently defaults to 3.11.15).

```bash
cd autogo
python3 ../autogodin/tools/patches/upstream_build_cpp_fix.py
bash scripts/build_cpp.sh
```

Upstream PR: https://github.com/ericjang/autogo/pull/5
Close [[autogodin-ydh.9]] when merged.
