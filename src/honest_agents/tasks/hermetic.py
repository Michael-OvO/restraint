"""Convert a seed task to the hermetic pytest-verifier contract (docs/design.md sections 3, 6 and 7).

Why: Endless Terminals' `tests/test.sh` runs `apt-get update`, installs curl, downloads the uv installer and then
`uvx --python 3.12 --with pytest==8.4.1 pytest ...` inside the agent's own container on every grading (measured on 40
random tasks: 40 of 40 identical). A failed download writes reward 0, so a true completion claim is scored as
fabrication, and the agent's container state (PATH, DNS, apt) decides whether the verifier starts at all.
The same 40 tasks ship `ubuntu:22.04` (python3 3.10, pytest pip-installed), and their `test_final_state.py` imports only
the standard library and pytest, so nothing needs the network.

A converted task has:

* `[metadata.honest] verifier = "pytest"` (and `verifier_site = true` only when a test imports something outside the
  standard library, which then has to come from the image's site-packages);
* `[verifier] network_mode = "no-network"`, so the verifier phase is offline for every harness that honours Harbor's
  phase policies, ours included (validation asserts it);
* a `tests/test.sh` that runs the same files offline with the image's own python3 and still writes
  `/logs/verifier/reward.txt`, so the task stays a valid Harbor task for `harbor run` with other harnesses. Our grader
  does not run it: it uploads `tests/` and a vendored pytest bundle fresh and runs pytest on `/tests` itself;
* `[metadata.hermetic]`: what the conversion read (sha256 of the original `test.sh`, its working directory, the test
  files), kept for audit.

Tasks that cannot be converted safely are reported with a reason and left untouched.

    python -m honest_agents.tasks.hermetic --tasks-dir out/seeds/endless-terminals
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.spec import VERIFIER_PYTEST, HonestTaskSpec
from honest_agents.tasks.taskfs import (
    ENV_SCRUB_FILES,
    final_stage,
    forbidden_copy_sources,
    mask_heredoc_bodies,
    read_task_toml,
    scrub_environment,
    set_honest,
    write_task_toml,
)

log = logging.getLogger(__name__)

HERMETIC_MARKER = "# honest-agents hermetic verifier"
NO_NETWORK = "no-network"  # harbor.models.task.config.NetworkMode.NO_NETWORK

# Modules that are standard library in the Pythons images ship (3.9 to 3.11) but were removed from 3.12, the version
# `sys.stdlib_module_names` describes on the dev machine and the training node.
_REMOVED_STDLIB = frozenset(
    [
        "aifc",
        "asynchat",
        "asyncore",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "distutils",
        "imghdr",
        "imp",
        "lib2to3",
        "mailcap",
        "msilib",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "smtpd",
        "sndhdr",
        "spwd",
        "sunau",
        "telnetlib",
        "uu",
        "xdrlib",
    ]
)
# Modules newer than the oldest Python the verifier accepts (3.9); a test importing one needs a newer image.
_NEWER_STDLIB = {"tomllib": (3, 11)}
# What the grader's pytest bundle (and the image's own pytest) provides.
_PYTEST_MODULES = frozenset({"pytest", "_pytest", "py"})
# Pytest plugins whose presence in the original test.sh changes nothing a directory run would not also do.
_BENIGN_PLUGINS = frozenset({"pytest", "pytest-json-ctrf"})
# Options that change which tests run or how pytest configures itself; a directory run would not reproduce them.
_FILTERING_OPTIONS = (
    "-k",
    "-m",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "-c",
    "-o",
    "--co",
    "--collect-only",
)
_NETWORK_COMMAND = re.compile(
    r"(^|[\s;&|(`])(apt-get|apt|apk|yum|dnf|curl|wget|uv|uvx|pipx|npm|git\s+clone|pip3?\s+(install|download)|"
    r"python3?\s+-m\s+pip\s+install)(\s|$)"
)
_REWARD_WRITE = re.compile(r"/logs/verifier/reward\.(txt|json)")
_CD = re.compile(r"^\s*cd\s+(\S+)", re.MULTILINE)
_PY_VERSION = re.compile(r"python3\.(\d+)")
_FROM = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", re.IGNORECASE | re.MULTILINE)
_OLD_DEBIAN = {"jessie", "stretch", "buster", "wheezy"}


@dataclass
class HermeticReport:
    """What `to_pytest_verifier` did to one task, or why it did not."""

    task_dir: Path
    ok: bool = False  # the task is hermetic after the call (converted now, or already was)
    changed: bool = False
    already_hermetic: bool = False
    reasons: list[str] = field(default_factory=list)  # why it could not be converted (empty when ok)
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # converted, but something deserves a look
    verifier_site: bool = False
    site_modules: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    n_tests: int = 0
    cwd: str | None = None

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "task_dir": str(self.task_dir), "reason": self.reason}


# ── reading a task ───────────────────────────────────────────────────────────


def _stdlib() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names) | _REMOVED_STDLIB


def collectable_test_files(tests_dir: Path) -> list[Path]:
    """Files pytest collects when run on the directory (`test_*.py`, `*_test.py`), recursively and sorted."""
    if not tests_dir.is_dir():
        return []
    return sorted(
        p
        for p in tests_dir.rglob("*.py")
        if p.is_file() and (p.name.startswith("test_") or p.name.endswith("_test.py"))
    )


def _count_tests(tree: ast.AST) -> int:
    """Test functions pytest would collect from a module: `test*` functions and `test*` methods of `Test*` classes."""
    n = 0
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            n += 1
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            n += sum(
                isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef) and m.name.startswith("test") for m in node.body
            )
    return n


def _local_modules(tests_dir: Path) -> set[str]:
    names: set[str] = set()
    for p in tests_dir.rglob("*.py"):
        names.add(p.stem)
        names.update(part for part in p.relative_to(tests_dir).parts[:-1])
    return names


def _imported_modules(tree: ast.AST) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update((a.name.split(".")[0], node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add((node.module.split(".")[0], node.lineno))
    return out


@dataclass
class _TestFacts:
    files: list[Path]
    n_tests: int
    site_modules: list[str]
    newer_modules: list[str]
    uses_cwd: bool
    errors: list[str]


def _analyse_tests(tests_dir: Path) -> _TestFacts:
    files = collectable_test_files(tests_dir)
    std, local = _stdlib(), _local_modules(tests_dir)
    n_tests, site, newer, errors, uses_cwd = 0, set(), set(), [], False
    for f in sorted(tests_dir.rglob("*.py")):
        try:
            src = f.read_text()
            tree = ast.parse(src, feature_version=(3, 9))
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            errors.append(f"{f.relative_to(tests_dir)} does not parse: {e}")
            continue
        if f in files:
            n_tests += _count_tests(tree)
        uses_cwd = uses_cwd or bool(re.search(r"\bos\.getcwd\(|\bPath\.cwd\(|\bos\.chdir\(", src))
        for mod, _line in _imported_modules(tree):
            if mod in _NEWER_STDLIB:
                newer.add(mod)
            if mod in std or mod in _PYTEST_MODULES or mod in local or mod == "__future__":
                continue
            site.add(mod)
    return _TestFacts(files, n_tests, sorted(site), sorted(newer), uses_cwd, errors)


@dataclass
class _ShellFacts:
    pytest_targets: list[str]
    filters: list[str]
    plugins: list[str]
    cwd: str | None
    runs_pytest: bool


def _pytest_args(toks: list[str]) -> list[str] | None:
    """Arguments after the pytest invocation in one command line (`pytest ...`, `/x/pytest ...`, `-m pytest ...`)."""
    for i, tok in enumerate(toks):
        if tok == "pytest" or tok.endswith("/pytest"):
            return toks[i + 1 :]
        if tok == "-m" and toks[i + 1 : i + 2] == ["pytest"]:
            return toks[i + 2 :]
    return None


def _analyse_test_sh(text: str) -> _ShellFacts:
    """What the original `test.sh` asked pytest to do. Best effort: unparseable lines are skipped, not trusted."""
    targets: list[str] = []
    filters: list[str] = []
    plugins: list[str] = []
    runs = False
    for line in text.replace("\\\n", " ").splitlines():
        line = "" if line.lstrip().startswith("#") else line.split(" #", 1)[0]
        if not line.strip():
            continue
        try:
            toks = shlex.split(line)
        except ValueError:
            toks = line.split()
        for i, tok in enumerate(toks):
            if tok in ("--with", "-w") and i + 1 < len(toks):
                plugins.append(toks[i + 1])
            elif tok.startswith("--with="):
                plugins.append(tok.split("=", 1)[1])
        after = _pytest_args(toks)
        if after is None:
            continue
        runs = True
        for t in after:
            if t.endswith(".py") or t.startswith(("/tests", "tests")):
                targets.append(t)
            elif (
                t in _FILTERING_OPTIONS
                or any(t.startswith(o + "=") for o in _FILTERING_OPTIONS if o.startswith("--"))
                or re.match(r"-[kmco](?!-)\S", t)  # attached short form: `-k"expr"`, `-mslow`, `-ofoo=1`
            ):
                filters.append(t)
    cd = _CD.search(text)
    return _ShellFacts(targets, filters, plugins, cd.group(1).strip("\"'") if cd else None, runs)


def _plugin_name(spec: str) -> str:
    return re.split(r"[=<>!~\[ ]", spec, maxsplit=1)[0].lower().replace("_", "-")


def _python_floor_problem(dockerfile: str) -> str | None:
    """A reason when the image's base is known to ship a Python older than 3.9; None when fine or unknown."""
    stage = final_stage(dockerfile)
    froms = _FROM.findall(stage)
    if not froms:
        return None
    base = froms[0].lower()
    if any(int(m) >= 9 for m in _PY_VERSION.findall(stage)):
        return None  # the image installs a newer python3.N explicitly
    for pattern, floor in (
        (r"(?:[\w.\-/]*/)?ubuntu:(\d+)\.(\d+)", (22, 4)),
        (r"(?:[\w.\-/]*/)?python:(\d+)\.(\d+)", (3, 9)),
    ):
        m = re.match(pattern, base)
        if m and (int(m.group(1)), int(m.group(2))) < floor:
            return f"base image {base} ships python older than 3.9"
    m = re.match(r"(?:[\w.\-/]*/)?debian:([a-z]+)", base)
    if m and m.group(1) in _OLD_DEBIAN:
        return f"base image {base} ships python3 older than 3.9"
    return None


def _image_has_python(dockerfile: str) -> bool:
    stage = final_stage(dockerfile)
    froms = _FROM.findall(stage)
    base = froms[0].lower() if froms else ""
    return bool(re.search(r"python3?", stage, re.IGNORECASE)) or "python" in base or "conda" in base


# ── contract check ───────────────────────────────────────────────────────────


def image_leak_problems(task_dir: Path) -> list[str]:
    """Tests, expected files or task.json that reach the agent: COPY/ADD of them, or the expected-state files left in
    the docker build context (`environment/task.json`, `test_initial_state.py`, `container.def`)."""
    env = Path(task_dir) / "environment"
    problems = [f"environment/{f} is in the docker build context" for f in _present_scrub_files(env)]
    for df in sorted(env.rglob("Dockerfile*")) if env.is_dir() else []:
        bad = forbidden_copy_sources(df.read_text(errors="replace"))
        if bad:
            problems.append(f"{df.relative_to(task_dir)} COPY/ADDs {sorted(set(bad))} into the image")
    return problems


def _present_scrub_files(env: Path) -> list[str]:
    return sorted({f.name for n in ENV_SCRUB_FILES for f in (env.rglob(n) if env.is_dir() else [])})


def hermetic_problems(task_dir: Path) -> list[str]:
    """Everything that stops `task_dir` from meeting the section 5 contract. Empty means hermetic."""
    task_dir = Path(task_dir)
    try:
        cfg = read_task_toml(task_dir)
        spec = HonestTaskSpec.model_validate((cfg.get("metadata") or {}).get("honest") or {})
    except Exception as e:  # noqa: BLE001 — unreadable task.toml or labels
        return [f"task.toml unreadable: {e}"]
    problems: list[str] = image_leak_problems(task_dir)
    if spec.verifier != VERIFIER_PYTEST:
        problems.append('[metadata.honest] verifier is not "pytest"')
    if (cfg.get("verifier") or {}).get("network_mode") != NO_NETWORK:
        problems.append('[verifier] network_mode is not "no-network" (the verifier phase must be offline)')
    tests = task_dir / "tests"
    facts = _analyse_tests(tests)
    problems += facts.errors
    if facts.n_tests == 0:
        problems.append("tests/ has no collectable pytest test (test_*.py with test functions)")
    if facts.site_modules and not spec.verifier_site:
        problems.append(f"tests import {facts.site_modules} outside the standard library but verifier_site is false")
    sh = tests / "test.sh"
    if not sh.is_file():
        problems.append("tests/test.sh is missing (needed for harbor run with other harnesses)")
    else:
        text = sh.read_text()
        if _NETWORK_COMMAND.search(text):
            problems.append("tests/test.sh uses the network (apt/curl/uv/pip install)")
        if not _REWARD_WRITE.search(text):
            problems.append("tests/test.sh does not write /logs/verifier/reward.txt")
    return problems


def is_hermetic(task_dir: Path) -> bool:
    """True iff the task already meets the verifier contract."""
    return not hermetic_problems(Path(task_dir))


# ── conversion ───────────────────────────────────────────────────────────────


def _render_test_sh(cwd: str) -> str:
    return (
        "#!/bin/bash\n"
        f"{HERMETIC_MARKER}. Offline: runs the pytest files in /tests with the image's own python3 and writes the reward\n"
        "# file other Harbor harnesses read. honest_agents' grader does not run this script: it uploads /tests fresh and\n"
        "# runs the same files with a vendored pytest.\n"
        "mkdir -p /logs/verifier\n"
        f"cd {shlex.quote(cwd)} 2>/dev/null || cd /\n"
        "python3 -m pytest /tests -q -p no:cacheprovider --rootdir /tests --confcutdir /tests "
        "--junitxml=/logs/verifier/junit.xml\n"
        "if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"
    )


def _dockerfile_workdir(dockerfile: str) -> str | None:
    dirs = re.findall(r"^\s*WORKDIR\s+(\S+)", mask_heredoc_bodies(final_stage(dockerfile)), re.MULTILINE)
    return dirs[-1] if dirs and dirs[-1].startswith("/") else None


def to_pytest_verifier(task_dir: Path) -> HermeticReport:
    """Rewrite `task_dir` in place to the hermetic contract, or report why it cannot be (and leave it untouched)."""
    task_dir = Path(task_dir)
    rep = HermeticReport(task_dir)
    try:
        cfg = read_task_toml(task_dir)
        spec_raw = dict((cfg.get("metadata") or {}).get("honest") or {})
        spec = HonestTaskSpec.model_validate(spec_raw)
    except Exception as e:  # noqa: BLE001
        rep.reasons.append(f"task.toml unreadable: {e}")
        return rep
    tests = task_dir / "tests"
    facts = _analyse_tests(tests)
    rep.test_files = [str(f.relative_to(tests)) for f in facts.files]
    rep.n_tests, rep.site_modules = facts.n_tests, facts.site_modules
    rep.verifier_site = bool(facts.site_modules)
    if not hermetic_problems(task_dir):
        rep.ok = rep.already_hermetic = True
        rep.cwd = (cfg.get("metadata") or {}).get("hermetic", {}).get("cwd")
        return rep

    # ── refusal rules: leave the task alone rather than grade something other than what the source graded ──
    rep.reasons += [p for p in image_leak_problems(task_dir) if "COPY/ADDs" in p]  # scrubbing cannot fix these
    rep.reasons += facts.errors
    if facts.n_tests == 0:
        rep.reasons.append("no pytest test files with test functions under tests/")
    sh_path = tests / "test.sh"
    if not sh_path.is_file():
        rep.reasons.append("no tests/test.sh")
        return rep
    original = sh_path.read_text()
    marked = HERMETIC_MARKER in original
    shell = _analyse_test_sh(original) if not marked else _ShellFacts([], [], [], None, True)
    if not marked and not shell.runs_pytest:
        rep.reasons.append("tests/test.sh does not run pytest (its checks cannot be reproduced by a pytest run)")
    if shell.filters:
        rep.reasons.append(f"tests/test.sh filters the tests with {shell.filters}; a directory run would differ")
    extra_plugins = sorted({_plugin_name(p) for p in shell.plugins} - _BENIGN_PLUGINS)
    extra_plugins = [p for p in extra_plugins if p.startswith("pytest-")]
    if extra_plugins:
        rep.reasons.append(f"tests/test.sh needs pytest plugins {extra_plugins}")
    if shell.pytest_targets:
        named = {Path(t).name for t in shell.pytest_targets if t.endswith(".py")}
        collected = {f.name for f in facts.files}
        if named - collected:
            rep.reasons.append(f"test.sh names {sorted(named - collected)}, which a run on /tests would not collect")
        elif collected - named:
            rep.reasons.append(f"test.sh runs only {sorted(named)} but tests/ also holds {sorted(collected - named)}")
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    dockerfile = dockerfile_path.read_text() if dockerfile_path.is_file() else None
    if dockerfile is None:
        rep.reasons.append("no environment/Dockerfile (cannot check the image ships python3)")
    else:
        if not _image_has_python(dockerfile):
            rep.reasons.append("the image does not install python3, which the hermetic verifier needs")
        if old := _python_floor_problem(dockerfile):
            rep.reasons.append(old)
        if not re.search(r"pytest", dockerfile):
            rep.warnings.append("image does not install pytest: tests/test.sh (other harnesses) cannot run offline")
    if facts.newer_modules:
        rep.warnings.append(f"tests import {facts.newer_modules}, which need python 3.11 or newer in the image")
    if facts.uses_cwd:
        rep.warnings.append(
            "tests depend on the working directory; the grader must run them from metadata.hermetic.cwd"
        )
    if rep.reasons:
        return rep

    # ── rewrite ──
    cwd = shell.cwd or cfg.get("metadata", {}).get("hermetic", {}).get("cwd")
    cwd = cwd or (_dockerfile_workdir(dockerfile) if dockerfile else None) or "/"
    rep.cwd = cwd
    if shell.cwd is None and not marked:
        rep.warnings.append(f"original test.sh had no cd; used the image WORKDIR or / ({cwd})")
    if removed := scrub_environment(task_dir):
        rep.changes.append(f"removed {removed} (expected state must not sit in the build context)")
    sh_path.write_text(_render_test_sh(cwd))
    sh_path.chmod(0o755)
    rep.changes.append("tests/test.sh rewritten to run offline with the image's python3")

    spec_raw.update({"verifier": VERIFIER_PYTEST, "verifier_site": rep.verifier_site})
    new_spec = HonestTaskSpec.model_validate({**spec.model_dump(mode="json", exclude_none=True), **spec_raw})
    cfg = set_honest(cfg, new_spec, (cfg.get("metadata") or {}).get("honest_variant"))
    verifier = dict(cfg.get("verifier") or {})
    verifier["network_mode"] = NO_NETWORK
    cfg["verifier"] = verifier
    md = dict(cfg.get("metadata") or {})
    md["hermetic"] = {
        "cwd": cwd,
        "original_test_sh_sha256": md.get("hermetic", {}).get("original_test_sh_sha256")
        if marked
        else hashlib.sha256(original.encode()).hexdigest(),
        "test_files": rep.test_files,
        "n_tests": rep.n_tests,
    }
    cfg["metadata"] = md
    write_task_toml(task_dir, cfg)
    rep.changes.append(
        '[metadata.honest] verifier = "pytest"' + (", verifier_site = true" if rep.verifier_site else "")
    )
    rep.changes.append('[verifier] network_mode = "no-network"')
    rep.changed = True
    problems = hermetic_problems(task_dir)
    if problems:  # a conversion that does not satisfy its own contract is a bug here, never a silent pass
        rep.reasons += [f"internal: converted task still fails the contract: {p}" for p in problems]
        return rep
    rep.ok = True
    return rep


def convert_many(task_dirs: list[Path]) -> list[HermeticReport]:
    """Convert each task; failures are reports, not exceptions."""
    out = []
    for d in task_dirs:
        try:
            out.append(to_pytest_verifier(d))
        except Exception as e:  # noqa: BLE001 — one odd task must not abort a batch
            r = HermeticReport(Path(d))
            r.reasons.append(f"{type(e).__name__}: {e}")
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="dir whose subdirectories are task dirs",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dirs = sorted(p.parent for p in args.tasks_dir.glob("*/task.toml"))
    reports = convert_many(dirs)
    for r in reports:
        if not r.ok:
            log.warning("not converted %s: %s", r.task_dir.name, r.reason)
    n_ok = sum(r.ok for r in reports)
    print(f"{n_ok} of {len(reports)} tasks hermetic, {len(reports) - n_ok} not convertible")


if __name__ == "__main__":
    main()
