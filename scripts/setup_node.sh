#!/usr/bin/env bash
# One-time setup of an Ubuntu GPU VM (bare metal / VM, not a container pod: Harbor needs a real Docker daemon + nftables).
#   scripts/setup_node.sh [--install-docker] [--pull-images TASKS_DIR]
source "$(dirname "$0")/_common.sh"

INSTALL_DOCKER=0; TASKS_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-docker) INSTALL_DOCKER=1; shift ;;
    --pull-images) TASKS_DIR="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if ! command -v docker >/dev/null; then
  if [[ $INSTALL_DOCKER == 1 ]]; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "Added $USER to the docker group; log out/in (or 'newgrp docker') before training."
  else
    echo "docker not found. Re-run with --install-docker or install Docker Engine + compose plugin." >&2; exit 1
  fi
fi
docker compose version >/dev/null 2>&1 || { echo "docker compose v2 plugin missing (apt install docker-compose-plugin)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "cannot talk to the docker daemon (permissions? is it running?)" >&2; exit 1; }

# Unattended OS upgrades replaced the NVIDIA userspace libraries under a loaded driver mid-run on the first node (580.178
# libraries vs 580.95 kernel module: every CUDA call failed until the module was reloaded). A rented training box
# should not change its own drivers.
if systemctl list-unit-files 2>/dev/null | grep -q unattended-upgrades; then
  systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true
  echo "disabled unattended upgrades (driver stability)"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen --extra train --extra eval

# The hermetic verifier's vendored pytest bundle (cache dir outside the venv; needs network once).
"$REPO_ROOT/scripts/build_verifier_bundle.sh"

# Qwen3.5 Gated DeltaNet kernels (imported as `fla`) come with the train extra; Qwen3.5 is ~20x slower without them.
uv run --frozen --extra train python -c "import fla" || { echo "flash-linear-attention failed to import" >&2; exit 1; }

if ! uv run --frozen --extra train python -c "from huggingface_hub import get_token; import sys; sys.exit(get_token() is None)"; then
  echo "NOTE: no Hugging Face token found; run 'uv run hf auth login' if model/dataset downloads are gated or rate limited."
fi

# Pre-download model weights so vLLM and the trainer don't race on the hub.
for m in Qwen/Qwen3-4B Qwen/Qwen3.5-4B; do
  rev="$(pinned_revision "$m")"  # the pinned snapshot (DEF-51)
  uv run --frozen --extra train hf download "$m" ${rev:+--revision "$rev"} >/dev/null && echo "cached $m@${rev:-main}"
done

# Pre-pull the exact (digest-pinned) images Harbor uses: the egress-sidecar base and the kernel probe image. Harbor's
# probe runs with a 30 s timeout and treats *any* failure (including a slow first pull) as "no kernel support",
# silently disabling egress control. Pulling by tag alone would not warm the pinned digest.
"${UV_RUN[@]}" python - <<'PY' | while read -r img; do echo "pull $img"; docker pull "$img" >/dev/null || echo "  FAILED $img"; done
import pathlib, re
from harbor.environments.docker import docker as d
print(d.DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE)
df = pathlib.Path(d.__file__).parent / "harbor-docker-egress-control-sidecar" / "Dockerfile"
for m in re.finditer(r"^FROM\s+(\S+)", df.read_text(), re.M):
    print(m.group(1))
PY

if [[ -n "$TASKS_DIR" ]]; then
  # Pull every prebuilt task image referenced via task.toml `docker_image`, and build the rest once so the
  # first training step does not pay for docker builds.
  grep -rhoE '^\s*docker_image\s*=\s*"[^"]+"' "$TASKS_DIR" --include task.toml | sed -E 's/.*"([^"]+)"/\1/' | sort -u |
    while read -r img; do echo "pull $img"; docker pull "$img" >/dev/null || echo "  FAILED $img"; done
  echo "Tasks without docker_image are built on first boot; prebuild them with the dataset builder's image step."
fi

"$REPO_ROOT/scripts/probe_node.sh"
