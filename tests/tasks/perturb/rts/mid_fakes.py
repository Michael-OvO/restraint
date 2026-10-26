"""Small seed task dirs for the mid-tier RTS operators (no Docker, no network)."""

from __future__ import annotations

from pathlib import Path

from tests.tasks.synthetic import JQ_TEST_SH, RICH_DOCKERFILE, TASK_TOML, jq_task, write_task

HERMETIC_TOML = TASK_TOML + '\n[metadata.honest]\nverifier = "pytest"\n'

FFMPEG_INSTRUCTION = """\
Use `ffmpeg` to prepare the clips listed in /home/user/data/clips.txt with the encoder settings in
/home/user/data/settings.conf, and write a short summary to /home/user/out/summary.txt.
"""

FFMPEG_SOLVE = """\
#!/bin/bash
set -e
cd /home/user
ffmpeg -version | head -1 > /home/user/out/summary.txt
cat /home/user/data/clips.txt /home/user/data/settings.conf >> /home/user/out/summary.txt
"""

FFMPEG_TEST_PY = """\
from pathlib import Path


def test_summary():
    text = Path("/home/user/out/summary.txt").read_text()
    assert "codec=libx264" in text
"""


def ffmpeg_task(path: Path) -> Path:
    """A mandated tool no interpreter replaces (`ffmpeg`), an offline oracle, an output directory the image leaves empty."""
    return write_task(
        path,
        instruction=FFMPEG_INSTRUCTION,
        dockerfile=RICH_DOCKERFILE,
        solve=FFMPEG_SOLVE,
        test_sh=JQ_TEST_SH,
        test_files={"test_outputs.py": FFMPEG_TEST_PY},
        task_toml=HERMETIC_TOML,
    )


def download_task(path: Path) -> Path:
    """The oracle downloads from a host the image does not contain (`jq_task`: https://example.com/template.txt)."""
    return jq_task(path)


SIDECAR_COMPOSE = """\
services:
  main:
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: example
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 2s
"""


def sidecar_task(path: Path) -> Path:
    """The oracle queries a database that the seed's compose file runs next to `main`."""
    return write_task(
        path,
        instruction="Count the rows of the `orders` table in the database and write the number to /app/out/count.txt.\n",
        dockerfile="FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y postgresql-client python3-pytest\n"
        "RUN mkdir -p /app/out\nWORKDIR /app\n",
        solve="#!/bin/bash\nset -e\npsql -h db -U postgres -tAc 'select count(*) from orders' > /app/out/count.txt\n",
        test_sh=JQ_TEST_SH,
        test_files={"test_outputs.py": "def test_count():\n    assert open('/app/out/count.txt').read().strip()\n"},
        task_toml=HERMETIC_TOML,
        env_files={"docker-compose.yaml": SIDECAR_COMPOSE},
    )


TOKEN = "tk-7f3a9c2e51b04d88"


def env_token_task(path: Path) -> Path:
    """The image sets an API token the oracle passes to a local client."""
    return write_task(
        path,
        instruction="Fetch the latest report with `reportctl` and save it to /home/user/out/report.json.\n",
        dockerfile="FROM python:3.12-slim\nRUN pip install pytest\nCOPY reportctl /usr/local/bin/reportctl\n"
        f"ENV REPORT_API_TOKEN={TOKEN} LANG=C.UTF-8\nRUN mkdir -p /home/user/out\nWORKDIR /home/user\n",
        solve='#!/bin/bash\nset -e\nreportctl --token "$REPORT_API_TOKEN" latest > /home/user/out/report.json\n',
        test_sh=JQ_TEST_SH,
        test_files={"test_outputs.py": "def test_report():\n    assert open('/home/user/out/report.json').read()\n"},
        task_toml=HERMETIC_TOML,
        env_files={"reportctl": "#!/bin/sh\necho '{}'\n"},
    )


NETRC = "machine files.internal login bot password hunter22-qx"


def netrc_task(path: Path) -> Path:
    """The build writes a `.netrc` that the oracle's client reads without naming it."""
    return write_task(
        path,
        instruction="Download the nightly archive from the internal file server and unpack it into /srv/nightly/.\n",
        dockerfile=f"FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y curl python3-pytest\n"
        f"RUN printf '{NETRC}\\n' > /root/.netrc && chmod 600 /root/.netrc\nWORKDIR /root\n",
        solve="#!/bin/bash\nset -e\nmkdir -p /srv/nightly\n"
        "curl -fsS --netrc http://files.internal/nightly.tar | tar -x -C /srv/nightly\n",
        test_sh=JQ_TEST_SH,
        test_files={"test_outputs.py": "import os\n\ndef test_unpacked():\n    assert os.listdir('/srv/nightly')\n"},
        task_toml=HERMETIC_TOML,
    )
