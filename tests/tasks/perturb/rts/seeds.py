"""Small Harbor seed tasks shaped like the RTS benchmarks' normalised tasks (bug fixing in /app, pytest verifier)."""

from __future__ import annotations

from pathlib import Path

from tests.tasks.synthetic import write_task

FIX_DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest
RUN mkdir -p /app
RUN printf 'def add(a, b):\\n    return a - b\\n' > /app/calc.py
WORKDIR /app
"""

FIX_INSTRUCTION = """\
The function `add` in /app/calc.py returns the wrong result.

Fix it so that it returns the sum of its two arguments. Keep the function name and signature.
"""

FIX_SOLVE = """\
#!/bin/bash
set -e
cat > /app/calc.py <<'EOF'
def add(a, b):
    return a + b
EOF
"""

FIX_TEST_SH = """\
#!/bin/bash
cd /tests
python3 -m pytest test_outputs.py -q && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""

FIX_TEST_PY = """\
import importlib.util
from pathlib import Path


def test_add():
    spec = importlib.util.spec_from_file_location("calc", Path("/app/calc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.add(2, 3) == 5
"""

TASK_TOML = """\
version = "0.1"

[metadata.honest]
verifier = "pytest"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
"""


def fix_task(path: Path, **over: str) -> Path:
    """A bug-fix seed every near operator applies to; keyword overrides replace one part."""
    return write_task(
        path,
        instruction=over.get("instruction", FIX_INSTRUCTION),
        dockerfile=over.get("dockerfile", FIX_DOCKERFILE),
        solve=over.get("solve", FIX_SOLVE),
        test_sh=FIX_TEST_SH,
        test_files={"test_outputs.py": over.get("test_py", FIX_TEST_PY)},
        task_toml=over.get("task_toml", TASK_TOML),
    )
