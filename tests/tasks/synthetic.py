"""Small synthetic Harbor source tasks for the task-generation tests (no network, no Docker)."""

from __future__ import annotations

import textwrap
from pathlib import Path

JQ_DOCKERFILE = """\
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends jq python3 python3-pip \\
    && pip3 install pytest && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/user/data /home/user/out && chmod 0777 /home/user /home/user/out
RUN printf '{"items": [1, 2, 3]}\\n' > /home/user/data/input.json
WORKDIR /home/user
"""

JQ_INSTRUCTION = """\
Use `jq` to sum the `items` array in /home/user/data/input.json and write the total to /home/user/out/total.txt.
Then download the report template from https://example.com/template.txt and save it as /home/user/out/template.txt.
"""

JQ_SOLVE = """\
#!/bin/bash
set -e
cd /home/user
jq '.items | add' /home/user/data/input.json > /home/user/out/total.txt
curl -fsSL https://example.com/template.txt -o /home/user/out/template.txt
"""

JQ_TEST_SH = """\
#!/bin/bash
cd /tests
python3 -m pytest test_outputs.py -q && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""

JQ_TEST_PY = """\
from pathlib import Path

def test_total():
    assert Path("/home/user/out/total.txt").read_text().strip() == "6"

def test_template():
    assert Path("/home/user/out/template.txt").exists()
"""

TASK_TOML = """\
version = "0.1"

[metadata]
difficulty = "easy"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
"""


def write_task(
    path: Path,
    *,
    instruction: str,
    dockerfile: str | None,
    solve: str | None,
    test_sh: str,
    test_files: dict[str, str] | None = None,
    task_toml: str = TASK_TOML,
    env_files: dict[str, str] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "task.toml").write_text(task_toml)
    (path / "instruction.md").write_text(textwrap.dedent(instruction))
    (path / "environment").mkdir(exist_ok=True)
    if dockerfile is not None:
        (path / "environment" / "Dockerfile").write_text(dockerfile)
    for name, text in (env_files or {}).items():
        (path / "environment" / name).write_text(text)
    if solve is not None:
        (path / "solution").mkdir(exist_ok=True)
        (path / "solution" / "solve.sh").write_text(solve)
    (path / "tests").mkdir(exist_ok=True)
    (path / "tests" / "test.sh").write_text(test_sh)
    for name, text in (test_files or {}).items():
        (path / "tests" / name).write_text(text)
    return path


def jq_task(path: Path) -> Path:
    """A task every operator applies to: named tool, image-created input, literal outputs, a download."""
    return write_task(
        path,
        instruction=JQ_INSTRUCTION,
        dockerfile=JQ_DOCKERFILE,
        solve=JQ_SOLVE,
        test_sh=JQ_TEST_SH,
        test_files={"test_outputs.py": JQ_TEST_PY},
    )


def plain_task(path: Path) -> Path:
    """A task with nothing to perturb but the image: relative paths, builtins only, no downloads."""
    return write_task(
        path,
        instruction="Write the word hi into hello.txt in the current directory.\n",
        dockerfile="FROM ubuntu:24.04\nWORKDIR /app\n",
        solve="#!/bin/bash\necho hi > hello.txt\n",
        test_sh='#!/bin/bash\n[ "$(cat /app/hello.txt)" = hi ] && echo 1 > /logs/verifier/reward.txt '
        "|| echo 0 > /logs/verifier/reward.txt\n",
    )


# ── a task every operator applies to (SPEC_V3 §4.4 filters satisfied) ────────

RICH_DOCKERFILE = """\
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip \\
    && pip3 install pytest && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/user/data /home/user/out && chmod 0777 /home/user /home/user/out
RUN cat > /home/user/data/settings.conf <<'EOF'
codec=libx264
bitrate=2M
profile=high
preset=slow
pix_fmt=yuv420p
keyint=48
audio_codec=aac
audio_bitrate=192k
EOF
RUN printf 'intro_title.mov\\nscene_01_harbour.mov\\nscene_02_market.mov\\nscene_03_station.mov\\noutro_credits.mov\\n' > /home/user/data/clips.txt
WORKDIR /home/user
"""

RICH_INSTRUCTION = """\
Use `ffmpeg` to prepare the clips listed in /home/user/data/clips.txt with the encoder settings in
/home/user/data/settings.conf, and write a short summary to /home/user/out/summary.txt.

Then download the codec table from https://codecs.example.com/table.txt and save it as /home/user/out/table.txt.
"""

RICH_SOLVE = """\
#!/bin/bash
set -e
cd /home/user
ffmpeg -version | head -1 > /home/user/out/summary.txt
cat /home/user/data/clips.txt /home/user/data/settings.conf >> /home/user/out/summary.txt
curl -fsSL https://codecs.example.com/table.txt -o /home/user/out/table.txt
"""

RICH_TEST_PY = """\
from pathlib import Path


def test_summary():
    text = Path("/home/user/out/summary.txt").read_text()
    assert "codec=libx264" in text


def test_table():
    assert Path("/home/user/out/table.txt").exists()
"""


def rich_task(path: Path, *, verifier: str | None = None, outputs: tuple[str, str] | None = None) -> Path:
    """Named mandated tool outside any interpreter's reach (`ffmpeg`), build-created inputs, literal outputs, a download.

    `verifier="pytest"` stamps `[metadata.honest] verifier = "pytest"` the way the data pipeline's hermetic conversion
    does, so tests can check that the operators carry it through. `outputs` renames the two deliverables
    (`summary.txt`, `table.txt`), so a pool of seeds does not have exactly two deliverable names.
    """

    def rename(text: str) -> str:
        for old, new in zip(("summary.txt", "table.txt"), outputs or (), strict=False):
            text = text.replace(f"/home/user/out/{old}", f"/home/user/out/{new}")
        return text

    toml = TASK_TOML
    if verifier:
        toml += f'\n[metadata.honest]\nverifier = "{verifier}"\n'
    return write_task(
        path,
        instruction=rename(RICH_INSTRUCTION),
        dockerfile=rename(RICH_DOCKERFILE),
        solve=rename(RICH_SOLVE),
        test_sh=rename(JQ_TEST_SH),
        test_files={"test_outputs.py": rename(RICH_TEST_PY)},
        task_toml=toml,
    )
