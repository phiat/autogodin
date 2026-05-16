#!/usr/bin/env python3
"""Patch autogo's scripts/build_cpp.sh to drop the hardcoded libpython3.10.so."""
import pathlib

p = pathlib.Path("scripts/build_cpp.sh")
s = p.read_text()

to_remove = [
    "PYTHON_INCLUDE=$($VENV_PYTHON -c \"import sysconfig; print(sysconfig.get_path('include'))\")\n",
    "PYTHON_LIBRARY=$($VENV_PYTHON -c \"import sysconfig, os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), 'libpython3.10.so'))\")\n",
    '    -DPython3_INCLUDE_DIR="$PYTHON_INCLUDE" \\\n',
    '    -DPython3_LIBRARY="$PYTHON_LIBRARY" \\\n',
]
for t in to_remove:
    assert t in s, f"not found: {t!r}"
    s = s.replace(t, "")

s = s.replace(
    "# Detect Python paths\n",
    "# Detect Python root (FindPython3 + pybind11 auto-detect the rest)\n",
)
p.write_text(s)
print("patched OK")
