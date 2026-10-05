"""Print the test files related to the files git reports as changed (used by `scripts/test.sh --changed`).

A source file `src/honest_agents/a/b.py` is related to a test file when the test is named `test_b.py` or imports
`honest_agents.a.b` directly (`import`, `from honest_agents.a import b`, `from honest_agents.a.b import x`). Changed test
files select themselves. A change to the shared test support (conftest, fakes, pyproject) selects the whole suite:
prints `tests`. Imports are followed one level only, so a deep helper's callers are found by the full run, not here.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ("tests/conftest.py", "tests/fakes.py", "pyproject.toml", "uv.lock")


def changed_files() -> list[str]:
    """Tracked files differing from HEAD (staged or not) plus untracked, non-ignored files."""
    out = []
    for cmd in (["diff", "--name-only", "HEAD"], ["ls-files", "--others", "--exclude-standard"]):
        out += subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return sorted(set(out))


def module_of(path: str) -> str | None:
    p = Path(path)
    if p.parts[:2] != ("src", "honest_agents") or p.suffix != ".py":
        return None
    parts = list(p.with_suffix("").parts[1:])
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(test: Path) -> set[str]:
    names: set[str] = set()
    try:
        tree = ast.parse(test.read_text())
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def related_tests(files: list[str]) -> list[str]:
    if any(f in SHARED or (f.startswith("tests/") and Path(f).name in ("conftest.py", "fakes.py")) for f in files):
        return ["tests"]
    modules = {m for f in files if (m := module_of(f))}
    stems = {m.rsplit(".", 1)[-1] for m in modules}
    chosen = {f for f in files if f.startswith("tests/") and Path(f).name.startswith("test_") and f.endswith(".py")}
    for test in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = str(test.relative_to(ROOT))
        if test.stem.removeprefix("test_") in stems or imports_of(test) & modules:
            chosen.add(rel)
    return sorted(f for f in chosen if (ROOT / f).exists())


if __name__ == "__main__":
    print("\n".join(related_tests(changed_files())))
    sys.exit(0)
