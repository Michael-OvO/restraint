# Sourced by scripts/eval_*.sh. Repo root, the uv command for our venv, and helpers for external benchmark checkouts.
# External benchmarks get their OWN venvs under $HONEST_EXT_DIR; nothing is ever installed into this repo's .venv.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
UV_RUN=(env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval)
EXT_DIR="${HONEST_EXT_DIR:-$HOME/.cache/honest_agents/ext}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"   # policy endpoint (EVAL=1 scripts/serve_vllm.sh ...)
export TRL_EXPERIMENTAL_SILENCE=1
mkdir -p "$EXT_DIR"

# Pinned upstream commits (checked 2026-10-04).
AGENTABSTAIN_REPO=https://github.com/AntiQuality/agentabstain
AGENTABSTAIN_COMMIT=cfc3faf7ab1cfd4892cde1158d6e43b2f312ddc3
AGENTIC_ABSTENTION_REPO=https://github.com/lhannnn/agentic-abstention
AGENTIC_ABSTENTION_COMMIT=ebbf1f51671f8e6d5da42a4188ff8c9254ee978e
ABSTENTIONBENCH_REPO=https://github.com/facebookresearch/AbstentionBench
ABSTENTIONBENCH_COMMIT=e29184174c69bc95139b8c33dcca09aacab9442a

# clone_pinned <url> <commit> <dir>: idempotent shallow-ish checkout of an exact commit.
clone_pinned() {
  local url="$1" commit="$2" dir="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone --filter=blob:none "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout --quiet "$commit"
}

# ext_venv <dir> <python version>: create <dir>/.venv with uv if missing; prints its python.
ext_venv() {
  local dir="$1" py="$2"
  if [[ ! -x "$dir/.venv/bin/python" ]]; then
    env -u VIRTUAL_ENV uv venv --quiet --python "$py" "$dir/.venv" >&2
  fi
  echo "$dir/.venv/bin/python"
}

# ext_pip <python> <args...>: install into an external venv (never ours).
ext_pip() {
  local py="$1"; shift
  env -u VIRTUAL_ENV uv pip install --quiet --python "$py" "$@"
}

require_model() {
  : "${MODEL:?set MODEL to the served model name (vllm --served-model-name, or the LoRA module name)}"
}
