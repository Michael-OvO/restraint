"""What exists in the container when the agent starts, and whether the oracle touches a path."""

from __future__ import annotations

import functools
import posixpath
import re

from honest_agents.tasks.analysis.facts import TaskFacts
from honest_agents.tasks.analysis.image_files import _DOCKER_INSTR_RE, _command_of, _segments, _split_heredocs
from honest_agents.tasks.analysis.script import _resolve, _tokens

# ── L1 / L5 and the task-quality rules of the leak register ──────────────────────────────────────────────────────────


def paths_overlap(a: str, b: str) -> bool:
    """`a` and `b` are the same path, or one is a directory above the other."""
    a, b = a.rstrip("/") or "/", b.rstrip("/") or "/"
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


_PY_WRITE_RE = re.compile(
    r"""['"][wax]b?\+?['"]|write_text|write_bytes|\.write\(|os\.(?:remove|rename|replace|unlink|truncate|chmod)|"""
    r"""shutil\.(?:move|rmtree|copy)|unlink\(|\.touch\(|\.rename\(|\.replace\("""
)


def oracle_touches(facts: TaskFacts, path: str) -> bool:
    """The oracle may change `path`: an rm/mv/chmod/sed -i/redirect operand, or a Python line that writes near its name.

    Static and conservative. A path for which this is true cannot carry a "do not change it" constraint that the
    oracle can still satisfy.
    """
    if any(paths_overlap(path, m) for m in facts.script.mutated):
        return True
    base = posixpath.basename(path.rstrip("/"))
    for ln in (facts.solve or "").splitlines():
        if (path in ln or (base and base in ln)) and _PY_WRITE_RE.search(ln):
            return True
    return False


_MKDIR_BRACE_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")


def _brace_expand(word: str) -> list[str]:
    m = _MKDIR_BRACE_RE.search(word)
    if not m:
        return [word]
    return [x for alt in m.group(1).split(",") for x in _brace_expand(word[: m.start()] + alt + word[m.end() :])]


@functools.lru_cache(maxsize=512)
def boot_dirs(dockerfile: str) -> frozenset[str]:
    """Directories the build creates (`mkdir`, `WORKDIR`, `useradd -m`), with their ancestors (heuristic)."""
    stripped, _ = _split_heredocs(dockerfile)
    dirs: set[str] = set()
    workdir: str | None = None
    for raw in stripped.replace("\\\n", " ").splitlines():
        m = _DOCKER_INSTR_RE.match(raw)
        if not m:
            continue
        instr, rest = m.group(1).upper(), m.group(2)
        if instr == "WORKDIR":
            workdir = _resolve(rest.strip(), "/root", workdir) or workdir
            if workdir:
                dirs.add(workdir)
        elif instr == "RUN":
            for seg in _segments(_tokens(rest)):
                cmd, args, _r = _command_of(seg)
                if cmd == "mkdir":
                    for a in [a for a in args if not a.startswith("-")]:
                        for w in _brace_expand(a):
                            p = _resolve(w, "/root", workdir)
                            if p:
                                dirs.add(p)
                elif cmd in ("useradd", "adduser") and "-m" in args:
                    user = next((a for a in reversed(args) if not a.startswith("-")), None)
                    if user:
                        dirs.add(f"/home/{user}")
    out = set(dirs)
    for d in dirs:
        while d not in ("/", ""):
            d = posixpath.dirname(d)
            out.add(d)
    return frozenset(out)


@functools.lru_cache(maxsize=512)
def _removed_by_build(dockerfile: str) -> frozenset[str]:
    """Paths a build step deletes or moves away (`rm`, the source of `mv`), as written in the Dockerfile."""
    stripped, _ = _split_heredocs(dockerfile)
    out: set[str] = set()
    workdir: str | None = None
    for raw in stripped.replace("\\\n", " ").splitlines():
        m = _DOCKER_INSTR_RE.match(raw)
        if not m:
            continue
        if m.group(1).upper() == "WORKDIR":
            workdir = _resolve(m.group(2).strip(), "/root", workdir) or workdir
        elif m.group(1).upper() == "RUN":
            for seg in _segments(_tokens(m.group(2))):
                cmd, args, _r = _command_of(seg)
                plain = [a for a in args if not a.startswith("-")]
                srcs = plain[:-1] if cmd == "mv" else plain if cmd in ("rm", "unlink") else []
                out.update(p for p in (_resolve(a, "/root", workdir) for a in srcs) if p)
    return frozenset(out)


def exists_at_boot(facts: TaskFacts, path: str) -> bool:
    """Whether `path` is in the image when the agent starts, as far as the Dockerfile shows.

    A file the build lists, a directory it makes or a directory above one of its files; failing those, a path the
    build text names at all (the auditors' rule, which also catches files a Python step writes).
    """
    path = posixpath.normpath(path)
    files = facts.image_files
    if path in files or path in boot_dirs(facts.dockerfile or ""):
        return True
    if path in _removed_by_build(facts.dockerfile or ""):
        return False
    if any(f.startswith(path.rstrip("/") + "/") for f in files):
        return True
    return facts.created_in_env(path)
