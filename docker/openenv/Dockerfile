# OpenEnv server for honest-agents. Build from the REPO ROOT:
#
#   docker build -f docker/openenv/Dockerfile -t honest-agents-openenv .
#
# Not based on ghcr.io/huggingface/openenv-base: it ships Python 3.11 and Harbor 0.24 needs >= 3.12.
# The server starts Harbor sandboxes, so it needs a Docker daemon. See "Running in a container" in
# src/honest_agents/openenv_env/README.md for the docker.sock caveats. In short:
#
#   docker run --rm -p 8000:8000 \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     -v /srv/honest/tasks:/srv/honest/tasks:ro \
#     -v /srv/honest/trials:/srv/honest/trials \
#     -e HONEST_TASKS_DIR=/srv/honest/tasks -e HONEST_TRIALS_DIR=/srv/honest/trials \
#     honest-agents-openenv
#
# (task and trial dirs mounted at the SAME path as on the host.)
#
# The hermetic verifier (docs/design.md section 6) needs the vendored pure-Python pytest bundle. Build it once on the host with
# `scripts/build_verifier_bundle.sh` (the command `honest_agents.verifier.ensure_bundle` names), then pass it as a named
# context:
#
#   docker build -f docker/openenv/Dockerfile --build-context verifier_bundle=$HONEST_VERIFIER_BUNDLE \
#     -t honest-agents-openenv .

FROM python:3.12-slim-bookworm

# Harbor's docker backend shells out to `docker` and `docker compose`: only the clients are needed. The daemon is the
# host's (socket mount) or a sibling dind service. buildx: recent `docker compose build` refuses to run without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
      > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin docker-buildx-plugin \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Dependencies first (cached layer). `--extra openenv` and the training extra conflict (vllm vs openenv `mcp` pins);
# this image only ever carries the openenv side.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra openenv --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra openenv --no-editable \
 && /app/.venv/bin/python -c "import honest_agents.openenv_env.server.app"

# The bundle is checked at build time, so an image that cannot grade a hermetic task is never produced.
COPY --from=verifier_bundle / /opt/honest/verifier-bundle
RUN /app/.venv/bin/python -c "from pathlib import Path; from honest_agents.verifier import ensure_bundle; ensure_bundle(Path('/opt/honest/verifier-bundle'))"

COPY docker/openenv/openenv.yaml /app/openenv.yaml

ENV PATH="/app/.venv/bin:$PATH" \
    HONEST_TASKS_DIR=/srv/honest/tasks \
    HONEST_TRIALS_DIR=/srv/honest/trials \
    HONEST_VERIFIER_BUNDLE=/opt/honest/verifier-bundle \
    HONEST_MAX_CONCURRENT_ENVS=64 \
    HONEST_MAX_CONCURRENT_BOOTS=16

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "honest_agents.openenv_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "20", "--ws-ping-timeout", "60"]
