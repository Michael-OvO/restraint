"""What the hermetic run still trusts outside `/tests` and the uploaded bundle, and how to notice it was changed.

`python3 -I -S` ignores the environment and `site`, but it still loads the standard library, the interpreter's shared
libraries and whatever its start-up logic finds next to the binary. A policy that runs as root can plant files there. The
protected-path manifest of `grading.py` covers the system `bin` directories only, so this module lists the rest, once,
before the agent acts (`discover`), and `grading.py` re-hashes exactly that list when grading:

* the interpreter, its `sys.path` entries (stdlib, lib-dynload, the zipped stdlib that precedes both) and the shared
  libraries it and its extension modules link to;
* every place CPython's own path calculation looks for a *different* installation before it finds the real one:
  `pyvenv.cfg` next to the executable or one directory up (a venv's `home =` redirects the stdlib), `pybuilddir.txt`,
  and `<ancestor>/lib/pythonX.Y` for every ancestor of the executable (the landmark search stops at the first hit);
* `/etc/ld.so.preload`, which the dynamic loader applies to every program, `env` and `timeout` included.

`site-packages`, `dist-packages` and `__pycache__` are left out (an agent may `pip install`); the runner passes
`-X pycache_prefix` to a fresh directory so no `.pyc` under the stdlib is ever read, which makes leaving them out safe.
"""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from honest_agents.verifier.runner import VERIFIER_PATH, VerifierInfraError

LD_SO_PRELOAD = "/etc/ld.so.preload"
LD_SO_CONF = "/etc/ld.so.conf"

# Printed by `python -I -S -c`: everything the host needs to decide what to watch. Runs once, before the agent acts.
_INFO_SCRIPT = (
    "import os, sys\n"
    "print('VER %d.%d' % sys.version_info[:2])\n"
    "print('EXE ' + os.path.realpath(sys.executable))\n"
    "print('PLATLIB ' + getattr(sys, 'platlibdir', 'lib'))\n"
    "for p in sys.path:\n"
    "    print('PATH ' + p)\n"
)

# Shared libraries of the interpreter and of every extension module in its `sys.path` directories, plus their symlink
# targets (a manifest hashes regular files and records a symlink only by its target). `ldd` is best effort.
_LIBS_SCRIPT = r"""
files="$1"; shift
for d in "$@"; do for f in "$d"/*.so "$d"/*/*.so; do [ -f "$f" ] && files="$files $f"; done; done
ldd $files 2>/dev/null | while read -r a b c d; do
  for p in "$a" "$c"; do
    case "$p" in /*) p=${p%:}; echo "$p"; readlink -f "$p" 2>/dev/null;; esac
  done
done | sort -u
"""


# Which file each library name resolves to *right now*. A library found through `/etc/ld.so.cache` can be redirected by
# adding a directory to `ld.so.conf.d` and running `ldconfig`, without touching any file the manifest hashes, so the
# resolution itself is compared too (paths, not contents: an in-place upgrade of a library is not a redirect).
_RESOLVE_SCRIPT = r"""
ldd "$1" 2>/dev/null | while read -r a b c d; do
  case "$c" in /*) echo "$a=$c";; esac
  case "$a" in /*) echo "$a";; esac
done | sort
"""


@dataclass
class Layout:
    """What the probe printed about the interpreter."""

    version: tuple[int, int]
    executable: str  # realpath
    platlibdir: str = "lib"
    sys_path: list[str] = field(default_factory=list)


def parse_info(text: str) -> Layout:
    version, exe, platlib, paths = None, "", "lib", []
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "VER":
            major, _, minor = rest.partition(".")
            version = (int(major), int(minor))
        elif kind == "EXE":
            exe = rest
        elif kind == "PLATLIB":
            platlib = rest or "lib"
        elif kind == "PATH" and rest:
            paths.append(rest)
    if version is None or not exe.startswith("/"):
        raise VerifierInfraError(f"unreadable interpreter probe output: {text[-200:]!r}")
    return Layout(version, exe, platlib, paths)


def _ancestors(path: str) -> list[str]:
    """`/a/b/c` -> ['/a/b/c', '/a/b', '/a', '/']."""
    out, cur = [], path
    while True:
        out.append(cur)
        parent = posixpath.dirname(cur)
        if parent == cur:
            return out
        cur = parent


def candidate_paths(python: str, layout: Layout) -> list[str]:
    """Every path whose creation or change could redirect the interpreter's start-up, and the files it loads from."""
    major, minor = layout.version
    tag = f"python{major}.{minor}"
    exe_dirs = list(dict.fromkeys([posixpath.dirname(python), posixpath.dirname(layout.executable)]))
    out: list[str] = [python, layout.executable, LD_SO_PRELOAD, LD_SO_CONF]
    for d in exe_dirs:
        out += [posixpath.join(d, "pyvenv.cfg"), posixpath.join(posixpath.dirname(d), "pyvenv.cfg")]
        out.append(posixpath.join(d, "pybuilddir.txt"))
        for anc in _ancestors(d):
            base = anc.rstrip("/")
            for lib in dict.fromkeys(["lib", "lib64", layout.platlibdir]):
                out.append(f"{base}/{lib}/{tag}")
            out.append(f"{base}/lib/python{major}{minor}.zip")
    out += layout.sys_path
    return list(dict.fromkeys(p for p in out if p))


async def discover(sb: Any, python: str, timeout: int = 120) -> tuple[list[str], list[str]]:
    """(paths to hash, the interpreter's isolated `sys.path`), taken as root before the agent acts. Raises
    `VerifierInfraError` if the interpreter cannot print its own layout (it could not run the verifier either)."""
    env = f"env -i PATH={shlex.quote(VERIFIER_PATH)} LANG=C.UTF-8"
    res = await sb.exec(
        f"{env} {shlex.quote(python)} -I -S -c {shlex.quote(_INFO_SCRIPT)}", user="root", timeout_sec=timeout
    )
    if res.return_code != 0:
        raise VerifierInfraError(f"interpreter probe failed (rc={res.return_code}): {(res.stderr or '')[-200:]}")
    layout = parse_info(res.stdout or "")
    paths = candidate_paths(python, layout)
    dirs = [p for p in layout.sys_path if p.startswith("/")]
    try:
        libs = await sb.exec(
            f"sh -c {shlex.quote(_LIBS_SCRIPT)} sh {shlex.quote(layout.executable)} "
            + " ".join(shlex.quote(d) for d in dirs),
            user="root",
            timeout_sec=timeout,
        )
        found = [ln for ln in (libs.stdout or "").splitlines() if ln.startswith("/")]
    except Exception:  # noqa: BLE001 - `ldd` is best effort: without it only the libraries are unwatched
        found = []
    return list(dict.fromkeys(paths + found)), layout.sys_path


def resolution_cmd(python: str) -> str:
    """The shell command whose stdout `resolution` reads (so a caller can batch it with other root execs)."""
    return f"sh -c {shlex.quote(_RESOLVE_SCRIPT)} sh {shlex.quote(python)}"


async def resolution(sb: Any, python: str, timeout: int = 60) -> str:
    """`ldd` of the interpreter as sorted `soname=path` lines; empty when `ldd` is missing (then nothing is compared)."""
    res = await sb.exec(resolution_cmd(python), user="root", timeout_sec=timeout)
    return (res.stdout or "").strip()


def shadowing(added: Iterable[str], baseline: dict[str, str], sys_path: Iterable[str]) -> list[str]:
    """Added top-level modules of a `sys.path` directory that hide a module a *later* directory provides.

    `sys.path` is [zip, stdlib, lib-dynload]: a `_hashlib.py` dropped into the stdlib directory would be imported in place
    of the compiled `_hashlib` in lib-dynload. Other additions (`apt install python3-venv` adds `venv/` and
    `ensurepip/`) are legitimate and are not flagged.
    """
    order = [d.rstrip("/") for d in sys_path if d.startswith("/")]

    def top_level(directory: str, path: str) -> str | None:
        if not path.startswith(directory + "/"):
            return None
        first = path[len(directory) + 1 :].split("/", 1)[0]
        return first.split(".", 1)[0] or None

    out = []
    for p in added:
        # the most specific directory the file sits in (lib-dynload is inside the stdlib directory)
        owners = sorted((i for i, d in enumerate(order) if top_level(d, p) is not None), key=lambda i: -len(order[i]))
        if not owners:
            continue
        i = owners[0]
        stem = top_level(order[i], p)
        if any(top_level(other, q) == stem for other in order[i + 1 :] for q in baseline):
            out.append(p)
    return out
