#!/usr/bin/env bash
# Build the vendored pytest bundle the hermetic verifier uploads next to the tests (SPEC_V3 section 3.2).
#
#   scripts/build_verifier_bundle.sh [DEST]
#
# DEST defaults to $HONEST_VERIFIER_BUNDLE, else ${XDG_CACHE_HOME:-~/.cache}/honest_agents/verifier-bundle: a cache
# directory OUTSIDE the project venv, so building it never touches the shared environment. Needs `uv` and network access
# once. The result is pure Python (wheels only), resolved for Python 3.9 so it also carries the backports pytest needs on
# 3.9/3.10 (exceptiongroup, tomli, typing_extensions); one bundle serves every task image (ubuntu 22.04/24.04, python:3.x).
# The task images' own test.sh pin pytest==8.4.1, so the verifier runs the same version.
set -euo pipefail

PYTEST_VERSION="${PYTEST_VERSION:-8.4.1}"
PY_FLOOR="${PY_FLOOR:-3.9}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${HONEST_VERIFIER_BUNDLE:-${XDG_CACHE_HOME:-$HOME/.cache}/honest_agents/verifier-bundle}}"

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
case "$DEST" in
  "$ROOT"/.venv*|*/site-packages*) echo "refusing to build into a virtualenv: $DEST" >&2; exit 1;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/site"

# --python-platform linux + --only-binary: the bundle is for the sandbox, never for this machine; a compiled wheel
# would silently not run there.
env -u VIRTUAL_ENV uv pip install --no-cache --target "$TMP/site" --python-version "$PY_FLOOR" \
  --python-platform linux --only-binary :all: "pytest==${PYTEST_VERSION}"
rm -rf "$TMP/site/bin" "$TMP/site/__pycache__"
find "$TMP/site" -name __pycache__ -type d -prune -exec rm -rf {} +
if find "$TMP/site" \( -name '*.so' -o -name '*.pyd' -o -name '*.dylib' \) | grep -q .; then
  echo "bundle contains compiled extensions; it must be pure Python" >&2; exit 1
fi

# The digest is computed by the same function the runner and the grader use (_hv_run.py is standard library only, so
# plain python3 loads it by path: no project environment is touched).
python3 - "$TMP" "$PYTEST_VERSION" "$PY_FLOOR" "$ROOT/src/honest_agents/verifier/_hv_run.py" <<'PY'
import importlib.util, json, sys
from pathlib import Path
tmp, version, floor, runner = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location("_hv_run", runner)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
packages = sorted(p.name for p in (tmp / "site").glob("*.dist-info"))
(tmp / "BUNDLE.json").write_text(json.dumps(
    {"pytest": version, "python_floor": floor, "digest": mod.tree_digest(str(tmp / "site")), "dist_info": packages},
    indent=2) + "\n")
PY

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP" "$DEST"
trap - EXIT
echo "verifier bundle built at $DEST (pytest ${PYTEST_VERSION}, python >= ${PY_FLOOR})"
echo "export HONEST_VERIFIER_BUNDLE=$DEST   # only needed if DEST is not the default cache path"
