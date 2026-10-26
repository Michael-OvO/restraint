"""Shell-script and Dockerfile reading: which files a script writes, mutates or fetches, and which binaries an image provides."""

from __future__ import annotations

import functools
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from honest_agents.tasks.analysis.tables import (
    _AMBIGUOUS_WORDS,
    _ASSIGN_RE,
    _HEREDOC_RE,
    _IGNORED_PREFIXES,
    _INSTALL_ARGS_RE,
    _NETWORK_PATTERNS,
    _PATH_HOME_RE,
    _PATH_RE,
    _URL_RE,
    _WRAPPER_ARG_OPTS,
    BASE_IMAGE_BINARIES,
    ESSENTIAL_BINARIES,
    PACKAGE_BINARIES,
    PIP_BINARIES,
    SHELL_BUILTINS,
    SHELL_KEYWORDS,
    WRAPPERS,
)


def strip_heredoc_bodies(script: str) -> str:
    """Drop heredoc bodies (data, not commands); keep the line that opens them."""
    out: list[str] = []
    lines = script.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC_RE.search(line)
        i += 1
        if m:
            term = m.group(2)
            while i < len(lines) and lines[i].strip() != term:
                i += 1
            i += 1  # skip terminator
    return "\n".join(out)


def _logical_lines(script: str) -> list[str]:
    text = strip_heredoc_bodies(script).replace("\\\n", " ")
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


_PUNCT = "();<>|&\n"


@functools.lru_cache(maxsize=4096)
def _tokens(text: str) -> tuple[str, ...]:
    """Shell-ish tokens over a whole script; newlines become separator tokens so quoted multi-line args survive.

    Cached (a Dockerfile RUN line is re-tokenised for every probed path) and therefore an immutable tuple.
    """
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=_PUNCT)
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        lex.commenters = "#"
        return tuple(lex)
    except ValueError:  # unbalanced quotes: fall back to a crude split
        return tuple(re.findall(r"[^\s;|&()<>]+|[;|&()<>\n]+", text))


_REDIRECTS = frozenset({">", ">>", "<", "<<", "<<<", ">&", "&>", "&>>", "2>", "2>>", ">|", "<<-", "<&"})
_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>"})
_FUNC_DEF_RE = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\s*\)", re.MULTILINE)
_ARITH_RE = re.compile(r"\$?\(\((?:[^()]|\([^()]*\))*\)\)")
_CMD_SUBST_RE = re.compile(r"\$\(\s*([A-Za-z0-9_.+-]+)|`\s*([A-Za-z0-9_.+-]+)")


def _is_separator(tok: str) -> bool:
    return tok not in _REDIRECTS and bool(tok) and all(c in _PUNCT for c in tok)


@dataclass
class ScriptFacts:
    commands: list[str] = field(default_factory=list)  # invoked command words, in order, unique
    written: list[str] = field(default_factory=list)  # absolute paths the script creates/writes
    mentioned: list[str] = field(default_factory=list)  # every absolute path literal
    mutated: list[str] = field(
        default_factory=list
    )  # paths an rm/mv/chmod/sed -i/redirect/... operand names (superset of written)
    # paths the script writes whole, replacing what was there: `>`, `cp`/`mv`/`install`/`rsync` onto it, `tee` without
    # `-a`, an output flag, `dd of=` without `conv=notrunc` (subset of written; not `>>`, `sed -i`, `touch`, `chmod`)
    replaced: list[str] = field(default_factory=list)


def analyse_script(script: str, *, home: str = "/root", cwd: str | None = None, _depth: int = 0) -> ScriptFacts:
    """Commands invoked and paths written by a shell script (heuristic; see module docstring)."""
    body = _ARITH_RE.sub("0", strip_heredoc_bodies(script).replace("\\\n", " "))
    functions = set(_FUNC_DEF_RE.findall(body))
    toks = _tokens(body)
    commands: list[str] = []
    written: list[str] = []
    mutated: list[str] = []
    replaced: list[str] = []
    expect_cmd, skip_until_sep, wrapper = True, False, None
    cmd: str | None = None
    args: list[str] = []

    def flush() -> None:
        nonlocal cmd, args, cwd
        if cmd is not None:
            written.extend(_writes_of(cmd, args, home, cwd))
            mutated.extend(_mutates_of(cmd, args, home, cwd))
            replaced.extend(_replaces_of(cmd, args, home, cwd))
            if cmd == "cd" and args:
                cwd = _resolve(args[0], home, cwd) or cwd
            if cmd in ("bash", "sh", "dash") and "-c" in args and _depth < 3:
                i = args.index("-c")
                if i + 1 < len(args):
                    sub = analyse_script(args[i + 1], home=home, cwd=cwd, _depth=_depth + 1)
                    commands.extend(sub.commands)
                    written.extend(sub.written)
                    mutated.extend(sub.mutated)
                    replaced.extend(sub.replaced)
        cmd, args = None, []

    i = 0
    while i < len(toks):
        t = toks[i]
        if _is_separator(t):
            flush()
            expect_cmd, skip_until_sep, wrapper = True, False, None
            i += 1
            continue
        if t in _REDIRECTS:
            if t in _WRITE_REDIRECTS and i + 1 < len(toks):
                p = _resolve(toks[i + 1], home, cwd)
                if p:
                    written.append(p)
                    mutated.append(p)
                    if t not in (">>", "&>>"):
                        replaced.append(p)
            i += 2
            continue
        if skip_until_sep:
            i += 1
            continue
        if expect_cmd:
            if t in ("for", "case", "select", "function"):
                skip_until_sep = True
                i += 1
                continue
            if _ASSIGN_RE.match(t) or t in SHELL_KEYWORDS or t in ("local", "declare", "export", "readonly"):
                i += 1
                continue
            if i + 1 < len(toks) and toks[i + 1].startswith(")"):  # case pattern `foo)`
                i += 1
                continue
            if wrapper is not None and t.startswith("-"):
                i += 2 if t in _WRAPPER_ARG_OPTS.get(wrapper, ()) else 1
                continue
            if wrapper == "timeout" and re.fullmatch(r"[0-9.]+[smhd]?", t):
                i += 1
                continue
            word = posixpath.basename(t) if t.startswith("/") else t
            if word in WRAPPERS:
                wrapper = word
                i += 1
                continue
            expect_cmd, wrapper = False, None
            if word == "cd":  # builtin, but it moves relative paths
                cmd, args = word, []
            elif _is_command_word(word) and word not in functions:
                commands.append(word)
                cmd, args = word, []
            i += 1
            continue
        if cmd is not None:
            args.append(t)
        i += 1
    flush()
    for m in _CMD_SUBST_RE.finditer(body):
        word = m.group(1) or m.group(2)
        if _is_command_word(word) and word not in functions:
            commands.append(word)
    return ScriptFacts(
        commands=_unique(commands),
        written=_unique([w for w in written if "{" not in w]),
        mentioned=extract_paths(script, home=home),
        mutated=_unique([m for m in [*written, *mutated] if "{" not in m]),
        replaced=_unique([r for r in replaced if "{" not in r]),
    )


def _is_command_word(word: str) -> bool:
    if not word or word in SHELL_BUILTINS or word in SHELL_KEYWORDS:
        return False
    if any(c in word for c in "$=\"'{}[]*?<>|&;()`"):
        return False
    if word.startswith(("./", "../", "~", "-")) or "/" in word:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", word))


_OUTPUT_FLAGS = frozenset({"-o", "-O", "--output", "--output-file", "--out"})


def _writes_of(cmd: str, args: list[str], home: str, cwd: str | None) -> list[str]:
    plain = [a for a in args if not a.startswith("-")]
    targets: list[str] = [args[i + 1] for i, a in enumerate(args[:-1]) if a in _OUTPUT_FLAGS]
    if cmd in ("touch", "mkdir", "tee"):
        targets += plain
    elif (
        cmd in ("cp", "mv", "install", "ln", "rsync")
        and len(plain) >= 2
        or cmd == "sed"
        and any(a.startswith("-i") for a in args)
        and plain
    ):
        targets.append(plain[-1])
    elif cmd in ("chmod", "chown") and len(plain) >= 2:
        targets += plain[1:]
    out = []
    for t in targets:
        p = _resolve(t, home, cwd)
        if p:
            out.append(p)
    return out


def _replaces_of(cmd: str, args: list[str], home: str, cwd: str | None) -> list[str]:
    """Paths a command writes whole (see `ScriptFacts.replaced`)."""
    plain = [a for a in args if not a.startswith("-")]
    targets: list[str] = [args[i + 1] for i, a in enumerate(args[:-1]) if a in _OUTPUT_FLAGS]
    if cmd == "tee" and not any(a in ("-a", "--append") for a in args):
        targets += plain
    elif cmd in ("cp", "mv", "install", "rsync") and len(plain) >= 2:
        targets.append(plain[-1])
    elif cmd == "dd" and not any(a.startswith("conv=") and "notrunc" in a for a in args):
        targets += [a[3:] for a in args if a.startswith("of=")]
    out = []
    for t in targets:
        p = _resolve(t, home, cwd)
        if p:
            out.append(p)
    return out


_DELETERS = frozenset({"rm", "rmdir", "unlink", "shred", "truncate", "touch", "tee", "mkdir", "chattr", "setfacl"})


def _mutates_of(cmd: str, args: list[str], home: str, cwd: str | None) -> list[str]:
    """Operands a command changes, beyond what `_writes_of` lists: sources of `mv`, targets of `rm`/`chmod`/`sed -i`.

    A path in this set cannot be promised untouched: forbidding changes to it (the control of `conflicting_constraint`)
    would contradict an oracle that moves, deletes or re-modes it.
    """
    plain = [a for a in args if not a.startswith("-")]
    targets: list[str] = []
    if cmd in _DELETERS or cmd == "mv":
        targets += plain
    elif cmd in ("chmod", "chown", "chgrp") and len(plain) >= 2:
        targets += plain[1:]
    elif cmd == "sed" and any(a.startswith("-i") or a == "--in-place" for a in args):
        targets += plain[1:] if not any(a in ("-e", "-f") for a in args) else plain
    elif cmd in ("cp", "install", "rsync", "ln") and plain:
        targets.append(plain[-1])
        targets += [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-t", "--target-directory")]
    elif cmd == "dd":
        targets += [a[3:] for a in args if a.startswith("of=")]
    elif cmd == "find" and any(a in ("-delete", "-exec", "-execdir") for a in args):
        targets += [a for a in args if a.startswith("/")][:1] + plain[:1]
    elif cmd in ("tar", "unzip"):
        targets += [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-C", "-d", "--directory")]
    targets += [a.split("=", 1)[1] for a in args if a.startswith("--output=")]
    out = []
    for t in targets:
        p = _resolve(t, home, cwd)
        if p:
            out.append(p)
    return out


def _resolve(token: str, home: str, cwd: str | None) -> str | None:
    if not token or "$" in token or "*" in token or token.startswith("-"):
        return None
    if token.startswith("~/") or token == "~":
        return posixpath.normpath(home + token[1:])
    if token.startswith("/"):
        return posixpath.normpath(token)
    if cwd and re.fullmatch(r"[A-Za-z0-9._+\-/]+", token):
        return posixpath.normpath(posixpath.join(cwd, token))
    return None


def extract_paths(text: str, *, home: str = "/root") -> list[str]:
    """Absolute (or `~`) path literals in free text or code, normalised, without trailing punctuation."""
    out = []
    for m in _PATH_RE.finditer(text):
        p = m.group(1).rstrip(".,:;")
        if p.startswith("~"):
            if not p.startswith("~/"):
                continue
            p = home + p[1:]
        p = posixpath.normpath(p)
        if p == "/" or p.startswith("//"):
            continue
        out.append(p)
    for m in _PATH_HOME_RE.finditer(text):
        out.append(posixpath.normpath(posixpath.join(home, m.group(1))))
    return _unique(out)


def is_ignored_path(p: str) -> bool:
    return any(p == pre or p.startswith(pre + "/") for pre in _IGNORED_PREFIXES)


def network_uses(script: str) -> list[str]:
    """Lines of the script that need network access (downloads / package installs)."""
    hits = []
    for line in _logical_lines(script):
        if any(p.search(line) for p in _NETWORK_PATTERNS):
            hits.append(line.strip())
    return hits


def network_targets(lines: list[str]) -> list[str]:
    """URLs and package names the network lines fetch."""
    targets: list[str] = []
    for line in lines:
        targets += _URL_RE.findall(line)
        for m in _INSTALL_ARGS_RE.finditer(line):
            for a in m.group(1).split():
                name = re.split(r"[=<>~!\[]", a)[0]
                if not a.startswith(("-", "$")) and re.fullmatch(r"[A-Za-z0-9._+\-]+", name) and not name.isdigit():
                    targets.append(name)
    return _unique([t for t in targets if t])


def _unique(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


@dataclass
class DockerfileFacts:
    base_image: str | None
    final_user: str | None
    workdir: str | None
    home: str | None
    apt_packages: list[str]
    pip_packages: list[str]
    has_nopasswd_sudo: bool


def analyse_dockerfile(text: str) -> DockerfileFacts:
    from honest_agents.tasks.taskfs import final_stage, mask_heredoc_bodies

    # Only the final image's stages count: a builder stage's USER / installs never reach the image the agent runs in.
    joined = final_stage(text)
    instructions = mask_heredoc_bodies(joined)  # heredoc bodies are data (SQL, scripts), not Docker instructions
    froms = re.findall(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", instructions, re.IGNORECASE | re.MULTILINE)
    users = re.findall(r"^\s*USER\s+(\S+)", instructions, re.IGNORECASE | re.MULTILINE)
    workdirs = re.findall(r"^\s*WORKDIR\s+(\S+)", instructions, re.IGNORECASE | re.MULTILINE)
    homes = re.findall(r"^\s*ENV\s+HOME[= ]\s*(\S+)", instructions, re.IGNORECASE | re.MULTILINE)
    apt: list[str] = []
    pip: list[str] = []
    for m in _INSTALL_ARGS_RE.finditer(joined):
        tool = m.group(0).split()[0]
        pkgs = [
            re.split(r"[=<>~!\[]", a)[0].lower()
            for a in re.split(r"\s+", m.group(1))
            if a and not a.startswith(("-", "$")) and re.fullmatch(r"[A-Za-z0-9._+\-=<>~!\[\],]+", a)
        ]
        (apt if tool.startswith("apt") else pip).extend(pkgs)
    return DockerfileFacts(
        base_image=froms[-1] if froms else None,
        final_user=users[-1] if users else None,
        workdir=workdirs[-1] if workdirs else None,
        home=homes[-1] if homes else None,
        apt_packages=_unique(apt),
        pip_packages=_unique(pip),
        has_nopasswd_sudo="NOPASSWD" in text,
    )


def likely_image_binaries(env_text: str, df: DockerfileFacts) -> set[str]:
    """Binaries the built image very likely contains: base image + apt/pip installs (by package-name mapping)."""
    bins = set(ESSENTIAL_BINARIES) | set(BASE_IMAGE_BINARIES)
    for pkg in df.apt_packages:
        bins.update(PACKAGE_BINARIES.get(pkg, (pkg,)))
    for pkg in df.pip_packages:
        bins.update(PIP_BINARIES.get(pkg, (pkg,)))
    # Install scripts copied into the build context (e.g. TMax's post_install.sh) use the same syntax.
    for m in _INSTALL_ARGS_RE.finditer(env_text):
        for a in m.group(1).split():
            name = re.split(r"[=<>~!\[]", a)[0].lower()
            if name and not name.startswith(("-", "$")):
                bins.update(PACKAGE_BINARIES.get(name, PIP_BINARIES.get(name, (name,))))
    return bins


def mentions_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w./-]){re.escape(word)}(?![\w/-])", text) is not None


def mentions_as_tool(text: str, word: str) -> bool:
    """The instruction names `word` as a tool: in code formatting, or as an unambiguous bare word."""
    if re.search(rf"`[^`\n]*(?<![\w.-]){re.escape(word)}(?![\w-])[^`\n]*`", text):
        return True
    return word not in _AMBIGUOUS_WORDS and mentions_word(text, word)


def _read_text(p: Path, limit: int = 1_000_000) -> str:
    try:
        if p.stat().st_size > limit:
            return ""
        return p.read_text(errors="strict")
    except (OSError, UnicodeDecodeError):
        return ""
