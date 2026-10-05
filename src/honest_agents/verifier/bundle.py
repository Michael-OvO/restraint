"""The vendored pytest bundle the hermetic verifier runs (SPEC_V3 section 3.2).

Layout: `<bundle>/site/` holds pure-Python packages (pytest 8.4.1 and its dependencies); `<bundle>/BUNDLE.json` records
what it was built from and the digest of `site/`. `scripts/build_verifier_bundle.sh` builds it into a cache directory
outside the shared venv.

Why pytest 8.4.1 and these dependencies: the Endless Terminals images are `ubuntu:22.04` (Python 3.10 from apt), SETA
images are `ubuntu:24.04` (3.12) or `python:3.11` / `python:3.13`, and every task's own `test.sh` pins `pytest==8.4.1`.
pytest 8.4.1 supports Python 3.9 to 3.14. The bundle is resolved for Python 3.9, so it also holds the backports pytest
needs on 3.9 and 3.10 (`exceptiongroup`, `tomli`, `typing_extensions`); on 3.11 and later they are simply unused.
Everything is a pure-Python wheel (no compiled extension), so one bundle serves every image and architecture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from honest_agents.verifier._hv_run import tree_digest

BUNDLE_ENV = "HONEST_VERIFIER_BUNDLE"
BUNDLE_JSON = "BUNDLE.json"
SITE_DIRNAME = "site"
PYTEST_VERSION = "8.4.1"
MIN_PYTHON = (3, 9)
BUILD_COMMAND = "scripts/build_verifier_bundle.sh"


class BundleError(RuntimeError):
    """The vendored pytest bundle is missing or does not match its recorded digest."""


def default_bundle_dir() -> Path:
    """`$HONEST_VERIFIER_BUNDLE`, else `$XDG_CACHE_HOME/honest_agents/verifier-bundle` (default `~/.cache/...`)."""
    env = os.environ.get(BUNDLE_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
    return Path(base) / "honest_agents" / "verifier-bundle"


def ensure_bundle(path: Path | None = None) -> Path:
    """Validate the bundle and return its root. Raises `BundleError`, naming the build command, if it is unusable."""
    root = Path(path) if path is not None else default_bundle_dir()
    hint = f"build it with `{BUILD_COMMAND}` (or set {BUNDLE_ENV} to a built bundle)"
    meta_path, site = root / BUNDLE_JSON, root / SITE_DIRNAME
    if not meta_path.is_file() or not (site / "pytest" / "__init__.py").is_file():
        raise BundleError(f"verifier bundle not found at {root}: {hint}")
    try:
        meta = json.loads(meta_path.read_text())
        expected = str(meta["digest"])
    except (OSError, ValueError, KeyError) as e:
        raise BundleError(f"verifier bundle at {root} has an unreadable {BUNDLE_JSON} ({e}): {hint}") from e
    # Hashed on every call (about 7 MB, tens of milliseconds): a cache keyed on modification times would miss an edit
    # in place, and the digest is what the in-sandbox runner is compared against.
    if tree_digest(str(site)) != expected:
        raise BundleError(f"verifier bundle at {root} does not match its recorded digest: rebuild it, {hint}")
    return root


def bundle_digest(root: Path) -> str:
    """The recorded digest of `<root>/site` (the runner compares what it finds in the sandbox against this)."""
    return str(json.loads((Path(root) / BUNDLE_JSON).read_text())["digest"])


def bundle_site(root: Path) -> Path:
    return Path(root) / SITE_DIRNAME
