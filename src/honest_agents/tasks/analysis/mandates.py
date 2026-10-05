"""Tool mandates, interpreter substitution and the hosts behind a download (the `remove_tool` and `no_network` filters)."""

from __future__ import annotations

import re

from honest_agents.tasks.analysis.script import _unique
from honest_agents.tasks.analysis.tables import _URL_RE

# ── docs/design.md section 7: tool mandates and interpreter substitution (remove_tool) ───

# General-purpose interpreters: with one of these in the image, most command-line tools can be re-implemented in a few
# lines (JSON, CSV, archives, HTTP, hashing), so deleting the tool does not make the task infeasible.
INTERPRETERS = frozenset(["python", "python3", "perl", "ruby", "node", "nodejs", "php", "lua", "tclsh", "Rscript"])
# Tools whose job an interpreter's standard library cannot do: compilers and build drivers, version control, media and
# document converters, database and remote-shell clients, signing. Only these stay eligible next to an interpreter.
INTERPRETER_PROOF_TOOLS = frozenset(
    [
        "git", "hg", "svn", "gcc", "g++", "cc", "clang", "clang++", "make", "cmake", "ninja", "go", "rustc", "cargo",
        "javac", "java", "ffmpeg", "ffprobe", "pandoc", "tesseract", "convert", "magick", "mogrify", "pdftotext",
        "pdftoppm", "dot", "psql", "pg_dump", "mysql", "mysqldump", "mariadb", "redis-cli", "ssh", "scp", "ssh-keygen",
        "gpg", "gpg2", "docker", "nginx", "apachectl", "apache2ctl", "crontab", "strace", "gdb", "valgrind",
    ]
)  # fmt: skip
_MANDATE_VERBS = r"(?:use|using|uses|via|invoke|invoking|call|calling|run|running|through|by\s+means\s+of)"


def mandates_tool(instruction: str, tool: str) -> bool:
    """The instruction requires this tool ("use `jq` to ...", "using jq", "must be done with `jq`"), not merely names it.

    A tool that is only mentioned (in an example, a file name, a description of the environment) leaves the agent free
    to solve the task another way, and that other way then passes the tests of a variant labelled infeasible.
    """
    t = re.escape(tool)
    word = rf"`?(?<![\w./-]){t}(?![\w/-])[^`\n]{{0,40}}`?"
    patterns = (
        rf"\b{_MANDATE_VERBS}\s+(?:only\s+)?(?:the\s+)?(?:command\s+|tool\s+|utility\s+|program\s+)?{word}",
        rf"\b(?:must|have\s+to|need\s+to|required\s+to|should)\b[^.\n]{{0,80}}\bwith\s+(?:the\s+)?{word}",
        rf"{word}\s+(?:command|tool|utility)?\s*(?:must|has\s+to|is\s+to|should)\s+be\s+used",
    )
    return any(re.search(p, instruction, re.IGNORECASE) for p in patterns)


def interpreter_can_substitute(tool: str, image_binaries: set[str]) -> bool:
    """An interpreter in the image could do this tool's job (docs/design.md section 7: such a `remove_tool` variant is excluded)."""
    if tool in INTERPRETER_PROOF_TOOLS:
        return False
    return any(i in image_binaries for i in INTERPRETERS - {tool})


# ── docs/design.md section 7: hosts behind a download (no_network) ───────────────────────

# Package managers and the hosts they talk to by default. A network line that names no URL and none of these gives no
# host, and `no_network` then has no target to record and no allowlist for its control.
_MANAGER_HOSTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip|uvx)\b"), ("pypi.org", "files.pythonhosted.org")),
    (
        re.compile(r"\bapt(?:-get)?\b"),
        ("archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com", "deb.debian.org", "security.debian.org"),
    ),
    (re.compile(r"\bnpm\b"), ("registry.npmjs.org",)),
    (re.compile(r"\bgo\s+(?:get|install)\b"), ("proxy.golang.org", "sum.golang.org")),
    (re.compile(r"\bcargo\s+install\b"), ("crates.io", "static.crates.io", "index.crates.io")),
    (re.compile(r"\bgem\s+install\b"), ("rubygems.org",)),
)
_HOST_RE = re.compile(r"^(?:https?|ftp|git|ssh)://(?:[^/@\s]+@)?([A-Za-z0-9.-]+)")
_SCP_HOST_RE = re.compile(r"\bgit@([A-Za-z0-9.-]+):")


def url_host(url: str) -> str | None:
    m = _HOST_RE.match(url)
    return m.group(1).lower().rstrip(".") if m else None


def network_hosts(lines: list[str]) -> list[str] | None:
    """Hosts the network lines need, or None when a line's host cannot be told (so no allowlist would be complete)."""
    hosts: list[str] = []
    for line in lines:
        found = [h for h in (url_host(u) for u in _URL_RE.findall(line)) if h]
        found += [m.group(1).lower() for m in _SCP_HOST_RE.finditer(line)]
        for pattern, manager_hosts in _MANAGER_HOSTS:
            if pattern.search(line):
                found += manager_hosts
        if not found:
            return None
        hosts += found
    return _unique(hosts)
