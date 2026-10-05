"""Task builders shaped like the real Endless Terminals layout (measured on 40 tasks, 2026-10-04), for the hermetic
conversion and validation tests. No network, no Docker."""

from __future__ import annotations

from pathlib import Path

from tests.tasks.synthetic import TASK_TOML, write_task

# Verbatim `tests/test.sh` shared by every sampled Endless Terminals task: it needs the internet on every grading.
ET_TEST_SH = """\
#!/bin/bash

apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
# Run pytest tests
cd /home/user
uvx   --python 3.12   --with pytest==8.4.1   pytest /tests/test_final_state.py -v
# Check exit code and write reward
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

ET_DOCKERFILE = """\
FROM ubuntu:22.04

RUN apt-get update -y && \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip && \\
    pip3 install --no-cache-dir pytest && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/user/data && chmod 0777 /home/user
RUN printf 'alpha\\n' > /home/user/data/input.txt
"""

ET_INSTRUCTION = "Copy /home/user/data/input.txt to /home/user/out.txt.\n"
ET_SOLVE = "#!/bin/bash\nset -e\ncd /home/user\ncp /home/user/data/input.txt /home/user/out.txt\n"
ET_TEST_PY = """\
import os
from pathlib import Path

import pytest

OUT = Path("/home/user/out.txt")


def test_exists():
    assert OUT.exists()


def test_content():
    assert OUT.read_text() == "alpha\\n"


@pytest.mark.parametrize("mode", [0o644])
def test_mode(mode):
    assert (os.stat(OUT).st_mode & 0o777) == mode
"""


def et_task(
    path: Path,
    *,
    test_sh: str = ET_TEST_SH,
    test_py: str | dict[str, str] = ET_TEST_PY,
    dockerfile: str | None = ET_DOCKERFILE,
    task_toml: str = TASK_TOML,
    solve: str | None = ET_SOLVE,
) -> Path:
    """An Endless-Terminals-shaped task: one `tests/test_final_state.py`, the network-hungry `test.sh`."""
    files = test_py if isinstance(test_py, dict) else {"test_final_state.py": test_py}
    return write_task(
        path,
        instruction=ET_INSTRUCTION,
        dockerfile=dockerfile,
        solve=solve,
        test_sh=test_sh,
        test_files=files,
        task_toml=task_toml,
    )
