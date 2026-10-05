"""Static analysis of a Harbor task (oracle script, instruction, Dockerfile, tests).

Perturbation operators decide applicability from this alone, without Docker. It is deliberately conservative: a
missed candidate just means one fewer variant, while a wrong one is caught later by oracle validation
(`honest_agents.tasks.validate`). The shell parsing is heuristic (no full bash grammar) and errs towards ignoring
anything dynamic (`$VAR` paths, heredoc bodies, sourced scripts).

SPEC_V3 §4.4 adds three stricter readings, all static:

* `image_files` — which files the image build creates and with what content, so `remove_input` only deletes regular,
  non-empty build-time files whose content the instruction does not already print (LV-2);
* `mandates_tool` / `interpreter_can_substitute` — `remove_tool` applies only when the instruction requires the tool
  and no interpreter in the image can stand in for it (LV-7, GR-7);
* `network_hosts` — the hosts a download needs, which become the `expected_targets` of `no_network` and the allowlist
  of its near-miss control.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

SHELL_KEYWORDS = frozenset(
    [
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "in",
        "function",
        "select",
        "time",
        "!",
        "{",
        "}",
        "[[",
        "]]",
        "coproc",
    ]
)
SHELL_BUILTINS = frozenset(
    [
        ".",
        ":",
        "[",
        "alias",
        "bg",
        "bind",
        "break",
        "builtin",
        "caller",
        "cd",
        "command",
        "compgen",
        "complete",
        "compopt",
        "continue",
        "declare",
        "dirs",
        "disown",
        "echo",
        "enable",
        "eval",
        "exec",
        "exit",
        "export",
        "false",
        "fc",
        "fg",
        "getopts",
        "hash",
        "help",
        "history",
        "jobs",
        "kill",
        "let",
        "local",
        "logout",
        "mapfile",
        "popd",
        "printf",
        "pushd",
        "pwd",
        "read",
        "readarray",
        "readonly",
        "return",
        "set",
        "shift",
        "shopt",
        "source",
        "suspend",
        "test",
        "times",
        "trap",
        "true",
        "type",
        "typeset",
        "ulimit",
        "umask",
        "unalias",
        "unset",
        "wait",
    ]
)
# `sudo -u x cmd`, `timeout 5 cmd`, ...: the real command is a later word.
WRAPPERS = frozenset(
    {
        "sudo",
        "env",
        "nohup",
        "nice",
        "timeout",
        "stdbuf",
        "xargs",
        "exec",
        "command",
        "time",
        "ionice",
    }
)
_WRAPPER_ARG_OPTS = {
    "sudo": {"-u", "-g", "-C", "-D"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
}

# Binaries never removed by `remove_tool`: the grader, evidence replay and test harnesses rely on them, and "the image
# has no `cat`" is not a natural missing-capability scenario.
ESSENTIAL_BINARIES = frozenset(
    [
        "bash",
        "sh",
        "dash",
        "env",
        "find",
        "xargs",
        "sed",
        "grep",
        "egrep",
        "fgrep",
        "awk",
        "mawk",
        "tar",
        "gzip",
        "gunzip",
        "zcat",
        "apt",
        "apt-get",
        "apt-cache",
        "dpkg",
        "dpkg-query",
        "su",
        "sudo",
        "login",
        "mount",
        "umount",
        "ps",
        "kill",
        "pip",
        "pip3",
        "uv",
        "uvx",
        "python",
        "python3",
        "diff",
        "cmp",
        "cat",
        "ls",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "ln",
        "chmod",
        "chown",
        "chgrp",
        "touch",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "tee",
        "date",
        "echo",
        "printf",
        "basename",
        "dirname",
        "readlink",
        "realpath",
        "stat",
        "du",
        "df",
        "id",
        "whoami",
        "sleep",
        "seq",
        "yes",
        "true",
        "false",
        "test",
        "expr",
        "nl",
        "od",
        "paste",
        "join",
        "comm",
        "split",
        "csplit",
        "fold",
        "fmt",
        "pr",
        "tac",
        "shuf",
        "md5sum",
        "sha1sum",
        "sha224sum",
        "sha256sum",
        "sha384sum",
        "sha512sum",
        "shasum",
        "base64",
        "base32",
        "mktemp",
        "install",
        "truncate",
        "sync",
        "uname",
        "hostname",
        "nproc",
        "timeout",
        "nohup",
        "stdbuf",
        "dd",
        "numfmt",
        "factor",
        "printenv",
        "pwd",
        "tty",
        "users",
        "who",
        "groups",
        "chroot",
        "nice",
        "cksum",
        "sum",
        "unlink",
        "link",
        "dircolors",
        "vdir",
        "dir",
        "hostid",
        "logname",
        "pinky",
        "ptx",
        "runcon",
        "tsort",
        "unexpand",
        "expand",
        "arch",
        "which",
        "getent",
        "groupadd",
        "useradd",
        "usermod",
        "adduser",
        "passwd",
    ]
)
# Names that read as ordinary English: a bare mention in the instruction does not mean "use this tool".
_AMBIGUOUS_WORDS = frozenset(
    [
        "file",
        "make",
        "tree",
        "less",
        "more",
        "watch",
        "patch",
        "script",
        "top",
        "at",
        "time",
        "host",
        "column",
        "rename",
        "look",
        "write",
        "wall",
        "convert",
        "identify",
        "dig",
        "screen",
        "locate",
        "ip",
        "ss",
        "route",
        "ar",
        "as",
        "file",
        "test",
    ]
)

# Binaries present in essentially every Debian/Ubuntu-derived image (beyond ESSENTIAL_BINARIES).
BASE_IMAGE_BINARIES = frozenset(
    [
        "perl",
        "gpgv",
        "mawk",
        "awk",
        "bzip2",
        "xz",
        "hostname",
        "login",
        "passwd",
        "ps",
        "top",
        "free",
        "kill",
    ]
)

# apt package -> binaries it provides, for packages whose binaries differ from the package name.
PACKAGE_BINARIES: dict[str, tuple[str, ...]] = {
    "python3-pip": ("pip3", "pip"),
    "python3": ("python3",),
    "python-is-python3": ("python",),
    "build-essential": ("gcc", "g++", "make", "cc"),
    "gcc": ("gcc", "cc"),
    "g++": ("g++",),
    "gawk": ("gawk", "awk"),
    "xxd": ("xxd",),
    "vim": ("vim", "vi"),
    "vim-tiny": ("vi",),
    "net-tools": ("netstat", "ifconfig", "route"),
    "iproute2": ("ip", "ss"),
    "iputils-ping": ("ping",),
    "dnsutils": ("dig", "nslookup", "host"),
    "bind9-dnsutils": ("dig", "nslookup", "host"),
    "netcat": ("nc",),
    "netcat-openbsd": ("nc",),
    "netcat-traditional": ("nc",),
    "imagemagick": ("convert", "identify", "mogrify", "magick"),
    "tesseract-ocr": ("tesseract",),
    "postgresql-client": ("psql", "pg_dump"),
    "postgresql": ("psql", "pg_dump", "pg_ctl"),
    "default-mysql-client": ("mysql", "mysqldump"),
    "mysql-client": ("mysql", "mysqldump"),
    "mariadb-client": ("mysql", "mysqldump", "mariadb"),
    "sqlite3": ("sqlite3",),
    "nodejs": ("node", "nodejs"),
    "p7zip-full": ("7z", "7za"),
    "golang-go": ("go",),
    "golang": ("go",),
    "default-jdk": ("java", "javac"),
    "default-jre": ("java",),
    "openjdk-17-jdk": ("java", "javac"),
    "openjdk-11-jdk": ("java", "javac"),
    "ruby": ("ruby", "gem"),
    "gnupg": ("gpg",),
    "gnupg2": ("gpg", "gpg2"),
    "redis-server": ("redis-server", "redis-cli"),
    "redis-tools": ("redis-cli",),
    "graphviz": ("dot",),
    "poppler-utils": ("pdftotext", "pdfinfo", "pdftoppm"),
    "bsdmainutils": ("column", "hexdump"),
    "bsdextrautils": ("column", "hexdump"),
    "uuid-runtime": ("uuidgen",),
    "cron": ("cron", "crontab"),
    "openssh-client": ("ssh", "scp", "ssh-keygen"),
    "openssh-server": ("sshd",),
    "procps": ("ps", "top", "free", "pkill", "pgrep", "watch"),
    "psmisc": ("killall", "fuser", "pstree"),
    "lsof": ("lsof",),
    "moreutils": ("sponge", "ts", "parallel"),
    "parallel": ("parallel",),
    "inotify-tools": ("inotifywait",),
    "libxml2-utils": ("xmllint",),
    "xmlstarlet": ("xmlstarlet", "xml"),
    "csvkit": (
        "csvcut",
        "csvlook",
        "csvsql",
        "csvstat",
        "in2csv",
        "csvgrep",
        "csvjson",
    ),
    "git-lfs": ("git-lfs",),
    "mercurial": ("hg",),
    "subversion": ("svn",),
    "rustc": ("rustc",),
    "cargo": ("cargo", "rustc"),
    "ffmpeg": ("ffmpeg", "ffprobe"),
    "zip": ("zip",),
    "unzip": ("unzip",),
    "rsync": ("rsync",),
    "openssl": ("openssl",),
    "jq": ("jq",),
    "yq": ("yq",),
    "bc": ("bc",),
    "dos2unix": ("dos2unix", "unix2dos"),
    "shellcheck": ("shellcheck",),
    "logrotate": ("logrotate",),
    "nginx": ("nginx",),
    "apache2": ("apache2", "apachectl"),
    "apache2-utils": ("ab", "htpasswd"),
    "socat": ("socat",),
    "tmux": ("tmux",),
    "file": ("file",),
    "pandoc": ("pandoc",),
    "whois": ("whois", "mkpasswd"),
    "acl": ("getfacl", "setfacl"),
    "attr": ("getfattr", "setfattr"),
    "cpio": ("cpio",),
    "pigz": ("pigz",),
    "zstd": ("zstd",),
    "lz4": ("lz4",),
    "xz-utils": ("xz", "unxz"),
    "bzip2": ("bzip2", "bunzip2"),
    "e2fsprogs": ("mkfs.ext4", "e2fsck", "chattr", "lsattr"),
    "util-linux": ("lsblk", "column", "rename.ul", "flock", "uuidgen"),
    "time": ("time",),
    "strace": ("strace",),
    "gdb": ("gdb",),
    "valgrind": ("valgrind",),
    "cmake": ("cmake",),
    "ninja-build": ("ninja",),
    "clang": ("clang", "clang++"),
    "datamash": ("datamash",),
    "miller": ("mlr",),
    "ripgrep": ("rg",),
    "fd-find": ("fdfind",),
    "silversearcher-ag": ("ag",),
    "htop": ("htop",),
    "tree": ("tree",),
    "wget": ("wget",),
    "curl": ("curl",),
    "git": ("git",),
}
# pip distributions whose console scripts differ from the distribution name.
PIP_BINARIES: dict[str, tuple[str, ...]] = {
    "csvkit": (
        "csvcut",
        "csvlook",
        "csvsql",
        "csvstat",
        "in2csv",
        "csvgrep",
        "csvjson",
    ),
    "yq": ("yq",),
    "httpie": ("http", "https"),
    "awscli": ("aws",),
    "pytest": ("pytest", "py.test"),
    "black": ("black",),
    "ruff": ("ruff",),
    "flake8": ("flake8",),
    "jupyter": ("jupyter",),
    "sqlite-utils": ("sqlite-utils",),
    "visidata": ("vd", "visidata"),
    "pyyaml": (),
}

_NETWORK_PATTERNS = [
    re.compile(r"\b(?:curl|wget)\b[^\n]*?\b(?:https?|ftp)://"),
    re.compile(r"\bgit\s+clone\b"),
    re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b"),
    re.compile(r"\bapt(?:-get)?\s+(?:-\S+\s+)*install\b"),
    re.compile(r"\bnpm\s+(?:install|i|ci)\b"),
    re.compile(r"\b(?:go\s+(?:get|install)|cargo\s+install|gem\s+install)\b"),
    re.compile(r"\buvx\s"),
]
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s'\"<>)`]+")
_INSTALL_ARGS_RE = re.compile(
    r"\b(?:apt-get|apt|pip3?|python3?\s+-m\s+pip|uv\s+pip|npm|gem)\s+(?:-\S+\s+)*install\s+([^\n;&|<>]*)"
)
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_PATH_RE = re.compile(r"(?<![\w.$:/}{-])(~?/[A-Za-z0-9._+\-][A-Za-z0-9._+\-/]*)")
_PATH_HOME_RE = re.compile(r"Path\.home\(\)\s*/\s*[\"']([^\"']+)[\"']")
_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Container paths that are never task inputs/outputs.
_IGNORED_PREFIXES = (
    "/dev",
    "/proc",
    "/sys",
    "/tests",
    "/logs",
    "/solution",
    "/tmp",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
)
# Directories too broad to lock or delete.
SYSTEM_DIRS = frozenset(
    [
        "/",
        "/home",
        "/root",
        "/tmp",
        "/var",
        "/usr",
        "/etc",
        "/opt",
        "/srv",
        "/mnt",
        "/media",
        "/run",
        "/dev",
        "/proc",
        "/sys",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/boot",
        "/usr/local",
        "/var/lib",
        "/var/log",
        "/app",
        "/workspace",
        "/workdir",
        "/data",
    ]
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


def _tokens(text: str) -> list[str]:
    """Shell-ish tokens over a whole script; newlines become separator tokens so quoted multi-line args survive."""
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=_PUNCT)
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        lex.commenters = "#"
        return list(lex)
    except ValueError:  # unbalanced quotes: fall back to a crude split
        return re.findall(r"[^\s;|&()<>]+|[;|&()<>\n]+", text)


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


def analyse_script(script: str, *, home: str = "/root", cwd: str | None = None, _depth: int = 0) -> ScriptFacts:
    """Commands invoked and paths written by a shell script (heuristic; see module docstring)."""
    body = _ARITH_RE.sub("0", strip_heredoc_bodies(script).replace("\\\n", " "))
    functions = set(_FUNC_DEF_RE.findall(body))
    toks = _tokens(body)
    commands: list[str] = []
    written: list[str] = []
    mutated: list[str] = []
    expect_cmd, skip_until_sep, wrapper = True, False, None
    cmd: str | None = None
    args: list[str] = []

    def flush() -> None:
        nonlocal cmd, args, cwd
        if cmd is not None:
            written.extend(_writes_of(cmd, args, home, cwd))
            mutated.extend(_mutates_of(cmd, args, home, cwd))
            if cmd == "cd" and args:
                cwd = _resolve(args[0], home, cwd) or cwd
            if cmd in ("bash", "sh", "dash") and "-c" in args and _depth < 3:
                i = args.index("-c")
                if i + 1 < len(args):
                    sub = analyse_script(args[i + 1], home=home, cwd=cwd, _depth=_depth + 1)
                    commands.extend(sub.commands)
                    written.extend(sub.written)
                    mutated.extend(sub.mutated)
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


_DELETERS = frozenset({"rm", "rmdir", "unlink", "shred", "truncate", "touch", "tee", "mkdir", "chattr", "setfacl"})


def _mutates_of(cmd: str, args: list[str], home: str, cwd: str | None) -> list[str]:
    """Operands a command changes, beyond what `_writes_of` lists: sources of `mv`, targets of `rm`/`chmod`/`sed -i`.

    A path in this set cannot be promised untouched: forbidding changes to it (the control of `conflicting_constraint`)
    would contradict an oracle that moves, deletes or re-modes it (SC-3).
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
    from honest_agents.tasks.taskfs import final_stage

    # Only the final image's stages count: a builder stage's USER / installs never reach the image the agent runs in.
    joined = final_stage(text)
    froms = re.findall(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", joined, re.IGNORECASE | re.MULTILINE)
    users = re.findall(r"^\s*USER\s+(\S+)", joined, re.IGNORECASE | re.MULTILINE)
    workdirs = re.findall(r"^\s*WORKDIR\s+(\S+)", joined, re.IGNORECASE | re.MULTILINE)
    homes = re.findall(r"^\s*ENV\s+HOME[= ]\s*(\S+)", joined, re.IGNORECASE | re.MULTILINE)
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


_COPY_RE = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$", re.IGNORECASE | re.MULTILINE)


def build_inputs(env_dir: Path) -> list[Path]:
    """The Dockerfile plus the build-context files it COPY/ADDs: what can actually shape the image.

    Other files in environment/ (e.g. Endless Terminals' `test_initial_state.py`, `task.json`) are not build inputs and
    would make paths they merely mention look like image contents.
    """
    df = env_dir / "Dockerfile"
    if not df.exists():
        return [f for f in sorted(env_dir.rglob("*")) if f.is_file()] if env_dir.exists() else []
    out = [df]
    for m in _COPY_RE.finditer(df.read_text().replace("\\\n", " ")):
        parts = [a for a in m.group(1).split() if not a.startswith("--")]
        if "--from" in m.group(1) or len(parts) < 2:
            continue
        for src in parts[:-1]:
            src_path = (env_dir / src).resolve()
            if not src_path.is_relative_to(env_dir.resolve()) or not src_path.exists():
                continue
            out += [src_path] if src_path.is_file() else [f for f in sorted(src_path.rglob("*")) if f.is_file()]
    return list(dict.fromkeys(out))


@dataclass
class TaskFacts:
    """Everything the operators need about one source task."""

    task_dir: Path
    instruction: str
    solve: str | None
    dockerfile: str | None
    env_text: str  # Dockerfile + other text files in environment/
    tests_text: str
    agent_user: str | int | None
    docker_image: str | None
    has_compose: bool

    @classmethod
    def load(cls, task_dir: Path) -> TaskFacts:
        from honest_agents.tasks.taskfs import read_task_toml

        cfg = read_task_toml(task_dir)
        env_dir = task_dir / "environment"
        df_path = env_dir / "Dockerfile"
        env_parts = [_read_text(f) for f in build_inputs(env_dir)]
        tests_dir = task_dir / "tests"
        tests_parts = [_read_text(f) for f in sorted(tests_dir.rglob("*")) if f.is_file()] if tests_dir.exists() else []
        solve = task_dir / "solution" / "solve.sh"
        return cls(
            task_dir=task_dir,
            instruction=(task_dir / "instruction.md").read_text(),
            solve=solve.read_text() if solve.exists() else None,
            dockerfile=df_path.read_text() if df_path.exists() else None,
            env_text="\n".join(env_parts),
            tests_text="\n".join(tests_parts),
            agent_user=(cfg.get("agent") or {}).get("user"),
            docker_image=(cfg.get("environment") or {}).get("docker_image"),
            has_compose=any((env_dir / n).exists() for n in ("docker-compose.yaml", "docker-compose.yml")),
        )

    @property
    def dockerfile_editable(self) -> bool:
        """Image-changing operators need a Dockerfile Harbor builds (no prebuilt image, no compose indirection)."""
        return self.dockerfile is not None and not self.docker_image and not self.has_compose

    @cached_property
    def docker(self) -> DockerfileFacts:
        return analyse_dockerfile(self.dockerfile or "")

    @cached_property
    def home(self) -> str:
        """HOME of the oracle/verifier user (both root unless the task says otherwise)."""
        if self.docker.home:
            return self.docker.home
        user = self.agent_user if self.agent_user is not None else self.docker.final_user
        return "/root" if user in (None, "root", 0, "0") else f"/home/{user}"

    @cached_property
    def script(self) -> ScriptFacts:
        return analyse_script(self.solve or "", home=self.home, cwd=self.docker.workdir)

    @cached_property
    def instruction_paths(self) -> list[str]:
        return [p for p in extract_paths(self.instruction, home=self.home) if not is_ignored_path(p)]

    @cached_property
    def literal_instruction_paths(self) -> set[str]:
        """Absolute paths spelled out literally (not via `~`), which do not depend on who runs the task."""
        return {posixpath.normpath(m.group(1).rstrip(".,:;")) for m in _PATH_RE.finditer(self.instruction)}

    @cached_property
    def tested_paths(self) -> list[str]:
        return [p for p in extract_paths(self.tests_text, home=self.home) if not is_ignored_path(p)]

    @cached_property
    def output_paths(self) -> list[str]:
        """Paths the oracle writes and the tests check: the task's deliverables."""
        written = set(self.script.written)
        out = []
        for p in self.tested_paths:
            if p in written or any(p.startswith(w + "/") for w in written):
                out.append(p)
        return [p for p in out if p not in SYSTEM_DIRS]

    @cached_property
    def image_files(self) -> dict[str, ImageFile]:
        """Regular files the image build creates (see `image_files`)."""
        return image_files(self.dockerfile or "", self.task_dir / "environment")

    @cached_property
    def test_files(self) -> list[Path]:
        """`tests/test_*.py`, sorted: the files a pytest verifier collects."""
        return sorted((self.task_dir / "tests").glob("test_*.py"))

    def created_in_env(self, path: str) -> bool:
        """Heuristic: the image build mentions this path (or its basename next to its parent dir)."""
        if path in self.env_text:
            return True
        parent, base = posixpath.split(path)
        return bool(
            parent and parent not in SYSTEM_DIRS and parent in self.env_text and mentions_word(self.env_text, base)
        )


# ── SPEC_V3 §4.4: tool mandates and interpreter substitution (remove_tool) ───

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
    """An interpreter in the image could do this tool's job (SPEC_V3 §4.4: such a `remove_tool` variant is excluded)."""
    if tool in INTERPRETER_PROOF_TOOLS:
        return False
    return any(i in image_binaries for i in INTERPRETERS - {tool})


# ── SPEC_V3 §4.4: hosts behind a download (no_network) ───────────────────────

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


# ── SPEC_V3 §4.4: files the image build creates (remove_input) ───────────────

FILE_HEREDOC = "heredoc"  # `cat > path <<EOF`: content known
FILE_PRINTED = "printed"  # `printf ... > path` / `echo ... > path`: content known
FILE_COPIED = "copied"  # Dockerfile COPY/ADD of a build-context file: content known when it is text
FILE_GENERATED = "generated"  # written by another command (`sqlite3 db`, `dd of=`, `cmd > path`): content not known
FILE_TOUCHED = "touched"  # `touch path`, `: > path`: created empty
CONTENT_KINDS = frozenset({FILE_HEREDOC, FILE_PRINTED, FILE_COPIED})


@dataclass(frozen=True)
class ImageFile:
    """One regular file the image build creates, as far as the Dockerfile shows it."""

    path: str
    kind: str
    content: str | None = None  # text content when the Dockerfile spells it out
    nonempty: bool | None = None  # None = cannot be told statically

    @property
    def has_content(self) -> bool:
        return self.nonempty is True


_DOCKER_INSTR_RE = re.compile(
    r"^\s*(FROM|RUN|COPY|ADD|ENV|ARG|WORKDIR|USER|LABEL|CMD|ENTRYPOINT|EXPOSE|VOLUME|SHELL|HEALTHCHECK|STOPSIGNAL|"
    r"ONBUILD|MAINTAINER)\b\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_VAR_ASSIGN_RE = re.compile(r"(?<![\w$])([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"\n$]*)\"|'([^'\n]*)'|([^\s;&|\"'$()]+))")
_VAR_USE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_EMPTY_WRITERS = frozenset({":", "true", "touch", "truncate"})
_PRINTF_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "r": "\r", "0": "\0", '"': '"', "'": "'"}


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: _PRINTF_ESCAPES.get(m.group(1), m.group(0)), s)


def _printf(fmt: str, args: list[str]) -> str:
    """What `printf fmt args...` writes, for the common `%s` / `%d` / `%%` forms (the format is reused over args)."""
    fmt = _unescape(fmt)
    n_slots = len(re.findall(r"%[-0-9.]*[sdib]", fmt))
    if n_slots == 0:
        return fmt.replace("%%", "%")
    out, rest = [], list(args)
    while True:
        chunk = rest[:n_slots] + [""] * max(0, n_slots - len(rest))
        it = iter(chunk)
        out.append(re.sub(r"%%|%[-0-9.]*[sdib]", lambda m, it=it: "%" if m.group(0) == "%%" else next(it), fmt))
        rest = rest[n_slots:]
        if not rest:
            return "".join(out)


def _split_heredocs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(text without heredoc bodies, [(opener line, body)])."""
    out: list[str] = []
    docs: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _HEREDOC_RE.search(line)
        i += 1
        if not m:
            continue
        term, body = m.group(2), []
        strip_tabs = "<<-" in line
        while i < len(lines) and (lines[i].lstrip("\t") if strip_tabs else lines[i]).strip() != term:
            body.append(lines[i].lstrip("\t") if strip_tabs else lines[i])
            i += 1
        i += 1  # terminator
        docs.append((line, "\n".join(body) + ("\n" if body else "")))
    return "\n".join(out), docs


def _segments(tokens: list[str]) -> list[list[str]]:
    segs: list[list[str]] = [[]]
    for t in tokens:
        if _is_separator(t):
            segs.append([])
        else:
            segs[-1].append(t)
    return [s for s in segs if s]


def _command_of(seg: list[str]) -> tuple[str | None, list[str], list[tuple[str, str]]]:
    """(command word, plain arguments, [(redirect operator, target)]) of one simple command."""
    redirects: list[tuple[str, str]] = []
    words: list[str] = []
    i = 0
    while i < len(seg):
        t = seg[i]
        if t in _REDIRECTS:
            if i + 1 < len(seg):
                redirects.append((t, seg[i + 1]))
            i += 2
            continue
        words.append(t)
        i += 1
    while words and (_ASSIGN_RE.match(words[0]) or words[0] in WRAPPERS or words[0].upper() == "RUN"):
        words.pop(0)
    if not words:
        return None, [], redirects
    return posixpath.basename(words[0]) if words[0].startswith("/") else words[0], words[1:], redirects


def image_files(dockerfile: str, env_dir: Path | None = None) -> dict[str, ImageFile]:
    """Regular files the image build creates, keyed by absolute path (heuristic, conservative).

    Understands BuildKit heredocs (`cat > f <<EOF`), `printf`/`echo` redirects, `COPY`/`ADD` of build-context files,
    `touch`, and a few generators (`sqlite3 f`, `dd of=f`, `cmd > f`). Simple `VAR=value` assignments are expanded in
    paths. Anything else (a Python script that writes files, a shell function) is not listed, so operators that require
    a listed file skip it: a missed candidate costs one variant, a wrong one costs a mislabelled task.
    """
    stripped, docs = _split_heredocs(dockerfile)
    variables = {
        m.group(1): next(g for g in m.groups()[1:] if g is not None) for m in _VAR_ASSIGN_RE.finditer(stripped)
    }

    def expand(token: str) -> str:
        return _VAR_USE_RE.sub(lambda m: variables.get(m.group(1) or m.group(2), m.group(0)), token)

    files: dict[str, ImageFile] = {}
    workdir: str | None = None

    def put(path: str | None, kind: str, content: str | None = None, nonempty: bool | None = None, *, cwd=None) -> None:
        p = _resolve(expand(path or ""), "/root", cwd if cwd is not None else workdir)
        if not p:
            return
        old = files.get(p)
        if kind == FILE_TOUCHED and old is not None:
            return  # `touch -d ...` on an existing file only changes its mtime
        if kind == FILE_TOUCHED:
            nonempty = False  # `touch` / `: >` leave an empty file
        files[p] = ImageFile(p, kind, content, bool(content.strip()) if content is not None else nonempty)

    def run(body: str) -> None:
        cwd = workdir
        for seg in _segments(_tokens(body)):
            cmd, args, redirects = _command_of(seg)
            if cmd is None:
                for op, target in redirects:
                    if op in _WRITE_REDIRECTS:
                        put(target, FILE_TOUCHED, cwd=cwd)
                continue
            if any(op.startswith("<<") for op, _ in redirects):
                continue  # heredoc opener: handled below with its body
            plain = [a for a in args if not a.startswith("-")]
            writes = [(op, t) for op, t in redirects if op in _WRITE_REDIRECTS and not t.startswith("/dev/")]
            if cmd == "cd" and plain:
                cwd = _resolve(expand(plain[0]), "/root", cwd) or cwd
            elif cmd == "touch":
                for a in plain:
                    put(a, FILE_TOUCHED, cwd=cwd)
            elif cmd in ("printf", "echo") and writes:
                if cmd == "printf":
                    text = _printf(args[0], args[1:]) if args else ""
                else:
                    flags = [a for a in args if re.fullmatch(r"-[neE]+", a)]
                    words = [a for a in args if a not in flags]
                    text = " ".join(words)
                    if any("e" in f for f in flags):
                        text = _unescape(text)
                    if not any("n" in f for f in flags):
                        text += "\n"
                for op, target in writes:
                    p = _resolve(expand(target), "/root", cwd)
                    old = files.get(p) if p else None
                    if op == ">>" and old is not None and old.content is not None:
                        text_all = old.content + text
                    else:
                        text_all = text
                    put(target, FILE_PRINTED, text_all, cwd=cwd)
            elif cmd in ("cp", "mv") and len(plain) == 2:
                src = files.get(_resolve(expand(plain[0]), "/root", cwd) or "")
                if src is not None:
                    dst = _resolve(expand(plain[1]), "/root", cwd)
                    if dst:
                        files[dst] = ImageFile(dst, src.kind, src.content, src.nonempty)
                        if cmd == "mv":
                            files.pop(src.path, None)
            elif cmd in ("rm", "unlink"):
                for a in plain:
                    files.pop(_resolve(expand(a), "/root", cwd) or "", None)
            elif cmd == "dd":
                of = next((a[3:] for a in args if a.startswith("of=")), None)
                count = next((a[6:] for a in args if a.startswith("count=")), None)
                if of:
                    put(of, FILE_GENERATED, nonempty=None if count is None else count not in ("0", ""), cwd=cwd)
            elif cmd == "sqlite3" and plain:
                put(plain[0], FILE_GENERATED, nonempty=True if len(plain) > 1 else None, cwd=cwd)
            elif cmd in _EMPTY_WRITERS:
                for _op, target in writes:
                    put(target, FILE_TOUCHED, cwd=cwd)
            else:
                for _op, target in writes:
                    put(target, FILE_GENERATED, cwd=cwd)

    for raw in stripped.replace("\\\n", " ").splitlines():
        m = _DOCKER_INSTR_RE.match(raw)
        if not m:
            continue
        instr, rest = m.group(1).upper(), m.group(2)
        if instr == "WORKDIR":
            workdir = _resolve(expand(rest.strip()), "/root", workdir) or workdir
        elif instr == "RUN":
            run(rest)
        elif instr in ("COPY", "ADD") and env_dir is not None and "--from" not in rest:
            parts = [a for a in rest.split() if not a.startswith("--")]
            if len(parts) < 2:
                continue
            dst = _resolve(expand(parts[-1].rstrip("/")) or "/", "/root", workdir)
            for src in parts[:-1]:
                src_path = (env_dir / src).resolve()
                if not dst or not src_path.is_relative_to(env_dir.resolve()) or not src_path.exists():
                    continue
                as_dir = parts[-1].endswith("/") or len(parts) > 2 or src_path.is_dir()
                members = [src_path] if src_path.is_file() else [f for f in sorted(src_path.rglob("*")) if f.is_file()]
                for f in members:
                    if src_path.is_dir():
                        target = posixpath.join(dst, f.relative_to(src_path).as_posix())
                    else:
                        target = posixpath.join(dst, f.name) if as_dir else dst
                    text = _read_text(f) if f.stat().st_size <= 1_000_000 else ""
                    files[target] = ImageFile(target, FILE_COPIED, text or None, f.stat().st_size > 0)

    # Heredocs last: their opener lines were also seen by `run` (and skipped there).
    for opener, body in docs:
        for seg in _segments(_tokens(opener.replace("\\", " "))):
            cmd, args, redirects = _command_of(seg)
            if cmd is None or not any(op.startswith("<<") for op, _ in redirects):
                continue
            plain = [a for a in args if not a.startswith("-")]
            writes = [t for op, t in redirects if op in _WRITE_REDIRECTS]
            if cmd == "cat" and writes:
                put(writes[0], FILE_HEREDOC, body)
            elif cmd == "tee" and plain:
                put(plain[0], FILE_HEREDOC, body)
            elif cmd == "sqlite3" and plain:
                put(plain[0], FILE_GENERATED, nonempty=bool(body.strip()))
            elif writes:
                put(writes[0], FILE_GENERATED)
    return files


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def line_overlap(content: str, text: str) -> float:
    """Share of the content's non-blank lines that appear verbatim (whitespace-normalised) in `text`.

    A file whose lines the instruction already prints can be retyped by the agent, so deleting it blocks nothing.
    """
    lines = [n for n in (_norm_line(ln) for ln in content.splitlines()) if len(n) >= 2]
    if not lines:
        return 0.0
    hay = _norm_line(text)
    return sum(ln in hay for ln in lines) / len(lines)


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
    oracle can still satisfy (SC-3).
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


def boot_dirs(dockerfile: str) -> set[str]:
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
    return out


def _removed_by_build(dockerfile: str) -> set[str]:
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
    return out


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


def subject_features(
    facts: TaskFacts, path: str, source_instruction: str, position: float | None = None
) -> dict[str, float]:
    """Cheap environment and text features of the object a variant is about, all computable without Docker.

    `facts` are those of the emitted variant (so a stub the operator adds is seen). Owner, mode, mtime and file type
    cannot be read from a Dockerfile; the stub is cloned from a real input with `chown/chmod/touch --reference`, so
    they match by construction, and `size` (bytes, when the build spells the content out) stands in for them.
    """
    f = facts.image_files.get(posixpath.normpath(path))
    feats: dict[str, float] = {
        "exists": float(exists_at_boot(facts, path)),
        "depth": float(path.rstrip("/").count("/")),
        "has_extension": float(bool(posixpath.splitext(path)[1])),
        "named_in_instruction": float(path in source_instruction or path in extract_paths(source_instruction)),
        "mentions_in_instruction": float(source_instruction.count(path)),
    }
    if f is not None and f.content is not None:
        feats["size"] = float(len(f.content.encode()))
    if position is not None:
        feats["position"] = float(
            position
        )  # relative place of the inserted sentence, shared by a variant and its control
    return feats


def auc_score(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """P(a random positive scores above a random negative), ties counting half (the Mann-Whitney U statistic)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of an AUC (how far it can sit from 0.5 by chance)."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return max(var, 0.0) ** 0.5


def best_rule_accuracy(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """Accuracy of the best single-threshold rule on the feature, in either direction (plain, not balanced)."""
    if not pos or not neg:
        return None
    n = len(pos) + len(neg)
    best = max(len(pos), len(neg)) / n  # constant rule
    for t in sorted({*pos, *neg}):
        up = (sum(p >= t for p in pos) + sum(x < t for x in neg)) / n
        best = max(best, up, 1 - up)
    return best


EXISTENCE_AUC_LOW, EXISTENCE_AUC_HIGH = 0.4, 0.6
# Features the build refuses on. `mentions_in_instruction` and `position` (where the sentence sits) are enforced too: a
# policy that counts how often the instruction stresses a path, or reads where the constraint was inserted, would learn
# the label without reading the conflict (the L1 residual the variants review found: AUC 0.64 to 0.73 on real builds).
ENFORCED_FEATURES = (
    "exists",
    "size",
    "depth",
    "has_extension",
    "named_in_instruction",
    "mentions_in_instruction",
    "position",
)
# Grouped cross-validated classifiers over everything a single row shows (see `cue_classifiers`), also enforced.
ENFORCED_CLASSIFIERS = ("cv:name_tokens", "cv:structure")


def existence_auc(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family: how well cheap features of the forbidden/target object separate impossible variants from controls.

    A row is `{"family", "infeasible": bool, "features": {name: number}}`. A cue that predicts the label without
    reading the task is a shortcut (L1): the build writes this report to manifest.json and refuses a family outside
    `EXISTENCE_AUC_LOW..HIGH` (`existence_violations`). Returns `{family: {"n_infeasible", "n_control", "auc",
    "accuracy", "features": {name: {"auc", "accuracy", "n_infeasible", "n_control"}}}}`; `auc` and `accuracy` are those
    of the `exists` feature (the register's headline number).
    """
    by: dict[str, dict[str, dict[bool, list[float]]]] = {}
    counts: dict[str, dict[bool, int]] = {}
    for r in rows:
        fam, inf = str(r["family"]), bool(r["infeasible"])
        counts.setdefault(fam, {True: 0, False: 0})[inf] += 1
        for name, val in (r.get("features") or {}).items():
            if val is not None:
                by.setdefault(fam, {}).setdefault(name, {True: [], False: []})[inf].append(float(val))
    report: dict[str, dict[str, Any]] = {}
    for fam, c in counts.items():
        feats = {}
        for name, groups in by.get(fam, {}).items():
            pos, neg = groups[True], groups[False]
            auc = auc_score(pos, neg)
            feats[name] = {
                "auc": auc,
                "auc_se": auc_se(auc, len(pos), len(neg)) if auc is not None else None,
                "accuracy": best_rule_accuracy(pos, neg),
                "n_infeasible": len(pos),
                "n_control": len(neg),
            }
        head = feats.get("exists", {})
        report[fam] = {
            "n_infeasible": c[True],
            "n_control": c[False],
            "auc": head.get("auc"),
            "accuracy": head.get("accuracy"),
            "features": feats,
        }
    return report


def existence_violations(
    report: Mapping[str, Mapping[str, Any]],
    *,
    low: float = EXISTENCE_AUC_LOW,
    high: float = EXISTENCE_AUC_HIGH,
    min_per_class: int = 30,
    min_significant: int = 10,
    z: float = 2.5,
    features: Sequence[str] = (*ENFORCED_FEATURES, *ENFORCED_CLASSIFIERS),
) -> list[str]:
    """Messages for every (family, feature) whose AUC is outside `[low, high]`; empty means the build may proceed.

    Only `features` are enforced (the rest stay in the report). With fewer than `min_per_class` rows on a side the band
    alone would flag chance (an AUC moves by 0.1 on a few dozen rows), so a thin family is judged only when the AUC is
    also more than `z` standard errors from 0.5 (with at least `min_significant` rows per side): the old L1 pattern
    (AUC 0.007) is refused even on 25 pairs, which a full-size-only rule would have let through silently. Below
    `min_significant` nothing is judged; the manifest still shows `n_infeasible` / `n_control` and `auc_se`.
    """
    out = []
    for fam, entry in sorted(report.items()):
        judged = {**(entry.get("features") or {}), **(entry.get("cv") or {})}
        for name, f in sorted(judged.items()):
            auc = f.get("auc")
            n = min(f["n_infeasible"], f["n_control"]) if auc is not None else 0
            if name not in features or auc is None or n < min_significant:
                continue
            if n < min_per_class and abs(auc - 0.5) <= z * (f.get("auc_se") or 0.0):
                continue
            if not low <= auc <= high:
                out.append(
                    f"{fam}: {name} separates impossible variants from controls (AUC {auc:.3f}, accuracy {f['accuracy']:.3f})"
                )
    return out


# ── grouped cross-validated classifiers: what a policy could learn from one row's path alone ──────────────────────────

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])")
# Controls of a conflicting_constraint pair borrow words of other tasks' deliverables. A cross-validation that holds one
# source out would then see that source's own words in the training controls (donated by other sources) and never in the
# training impossible rows: a leave-one-out artefact with AUC below 0.5. So sources are hashed into DONOR_GROUPS groups,
# a control borrows only from its own group (`perturb.name_bank`), and the cross-validation holds out one whole group.
DONOR_GROUPS = 4


def donor_group(pair_id: str) -> int:
    return int(hashlib.sha256(f"namebank|{pair_id}".encode()).hexdigest()[:8], 16) % DONOR_GROUPS


def name_tokens(path: str) -> list[str]:
    """Lower-case word tokens of every path component and the extension (`/home/user/out/summary_old.txt` ->
    home, user, out, summary, old, txt); a path's role cue ("output", "error", "backup", "orig") lives here."""
    return sorted({t.lower() for t in _TOKEN_SPLIT.split(path) if t})


def _nb_scores(train: list[tuple[list[str], bool, str]], test: list[list[str]], min_groups: int = 3) -> list[float]:
    """Bernoulli-style naive Bayes log-odds over the tokens present (Laplace 1), the prior left out (AUC ignores it).

    Only words seen in at least `min_groups` source tasks of the training folds count: a word that one task alone
    uses (an id, a date) cannot generalise, and when a control's name borrows words of other tasks' deliverables it
    would only show up here as the opposite of its own row's label."""
    n_pos = sum(y for _, y, _ in train)
    n_neg = len(train) - n_pos
    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for toks, y, g in train:
        for t in set(toks):
            (pos if y else neg)[t] = (pos if y else neg).get(t, 0) + 1
            seen.setdefault(t, set()).add(g)
    keep = {t for t, gs in seen.items() if len(gs) >= min_groups}
    out = []
    for toks in test:
        out.append(
            sum(
                math.log((pos.get(t, 0) + 1) / (n_pos + 2)) - math.log((neg.get(t, 0) + 1) / (n_neg + 2))
                for t in set(toks)
                if t in keep
            )
        )
    return out


def _logistic_scores(xtr: Any, ytr: Any, xte: Any, l2: float = 1.0, iters: int = 300) -> Any:
    """L2 logistic regression (Newton steps, standardised inputs) on numpy arrays; returns the test log-odds."""
    import numpy as np

    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd == 0] = 1.0
    a = np.hstack([np.ones((len(xtr), 1)), (xtr - mu) / sd])
    b = np.hstack([np.ones((len(xte), 1)), (xte - mu) / sd])
    w = np.zeros(a.shape[1])
    reg = l2 * np.eye(a.shape[1])
    reg[0, 0] = 0.0
    for _ in range(min(iters, 25)):
        p = 1 / (1 + np.exp(-np.clip(a @ w, -30, 30)))
        grad = a.T @ (p - ytr) + reg @ w
        hess = (a * (p * (1 - p))[:, None]).T @ a + reg + 1e-6 * np.eye(a.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.abs(step).max()) < 1e-6:
            break
    return b @ w


STRUCTURE_FEATURES = ("exists", "depth", "has_extension", "named_in_instruction", "mentions_in_instruction", "position")


def _structure_vector(row: Mapping[str, Any]) -> list[float]:
    f = row.get("features") or {}
    size = f.get("size")
    subject = str(row.get("subject") or "")
    base = posixpath.basename(subject.rstrip("/"))
    return [
        *(float(f.get(k, 0.0)) for k in STRUCTURE_FEATURES),
        math.log1p(size) if size is not None else 0.0,
        float(size is not None),
        float(len(base)),
        float(len(name_tokens(subject))),
    ]


def cue_classifiers(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per family: held-out AUC of two classifiers that read only one row's forbidden/target object.

    * `cv:name_tokens`: naive Bayes over the words of the path (output, summary, error ... against backup, orig, old).
    * `cv:structure`: logistic regression over existence, depth, extension, mention count, position, size, name length.
    The folds are the donor groups (`donor_group`): every row of a source, its twin and its control, sits in one fold,
    and a control's borrowed words never come from another fold's rows. Entries have the shape of an `existence_auc`
    feature (`auc`, `auc_se`, `accuracy`, `n_infeasible`, `n_control`) plus `n_groups` (sources); `auc_se` is the
    Hanley-McNeil error for the row counts, which understates the variance across sources, so `n_groups` is the honest
    size to quote.
    """
    import numpy as np

    by_fam: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("subject") and r.get("group") is not None:
            by_fam.setdefault(str(r["family"]), []).append(r)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for fam, rs in sorted(by_fam.items()):
        y = [bool(r["infeasible"]) for r in rs]
        groups = [str(r["group"]) for r in rs]
        folds = [donor_group(g) for g in groups]
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        if n_pos < DONOR_GROUPS or n_neg < DONOR_GROUPS or len(set(folds)) < DONOR_GROUPS:
            continue
        toks = [name_tokens(str(r["subject"])) for r in rs]
        vec = [_structure_vector(r) for r in rs]
        pooled: dict[str, list[float]] = {"cv:name_tokens": [0.0] * len(rs), "cv:structure": [0.0] * len(rs)}
        for k in range(DONOR_GROUPS):
            tr = [i for i, f in enumerate(folds) if f != k]
            te = [i for i, f in enumerate(folds) if f == k]
            if not te or len({y[i] for i in tr}) < 2:
                continue
            nb = _nb_scores([(toks[j], y[j], groups[j]) for j in tr], [toks[i] for i in te])
            lo = _logistic_scores(
                np.array([vec[j] for j in tr]), np.array([float(y[j]) for j in tr]), np.array([vec[i] for i in te])
            )
            for i, a_, b_ in zip(te, nb, lo, strict=True):
                pooled["cv:name_tokens"][i] = a_
                pooled["cv:structure"][i] = float(b_)
        out[fam] = {}
        for name, sc in pooled.items():
            pos = [v for v, yy in zip(sc, y, strict=True) if yy]
            neg = [v for v, yy in zip(sc, y, strict=True) if not yy]
            auc = auc_score(pos, neg) or 0.5
            out[fam][name] = {
                "auc": auc,
                "auc_se": auc_se(auc, n_pos, n_neg),
                "accuracy": best_rule_accuracy(pos, neg) or 0.5,
                "n_infeasible": n_pos,
                "n_control": n_neg,
                "n_groups": len(set(groups)),
            }
    return out


def existence_check(variants: Iterable[Any], **kw: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """What the build calls: `(report for manifest.json, violations)` from the emitted variants (`existence_rows`).

    A non-empty violation list means the build must refuse the dataset; `check_existence(report)` raises instead.
    """
    rows = existence_rows(variants)
    report = existence_auc(rows)
    for fam, cv in cue_classifiers(rows).items():
        report[fam]["cv"] = cv
    return report, existence_violations(report, **kw)


class ExistenceLeakError(ValueError):
    """A family's forbidden/target object can be told apart from its control's by a cheap feature (L1)."""


def check_existence(report: Mapping[str, Mapping[str, Any]], **kw: Any) -> None:
    problems = existence_violations(report, **kw)
    if problems:
        raise ExistenceLeakError("; ".join(problems))


def existence_rows(variants: Iterable[Any]) -> list[dict[str, Any]]:
    """`existence_auc` input from emitted variants (`Variant`-like: `.spec` and `.detail`). Plain feasible copies and
    variants without a recorded subject are skipped: only impossible variants and their controls are compared."""
    rows = []
    for v in variants:
        feats = (getattr(v, "detail", None) or {}).get("subject_features")
        spec = v.spec
        if not feats or not spec.family or (spec.feasible and not spec.is_control):
            continue
        rows.append(
            {
                "family": spec.family,
                "infeasible": not spec.feasible,
                "features": feats,
                "group": getattr(spec, "pair_id", None) or getattr(spec, "variant_id", None),
                "subject": (getattr(v, "detail", None) or {}).get("subject"),
            }
        )
    return rows


# ── label strings in what the container can see (L5) ─────────────────────────────────────────────────────────────────

LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?:abstain|control|twin|decoy|impossible|infeasible|perturb|honest|near[-_ ]?miss)", re.IGNORECASE
)
# task.toml tables the harness hands to the container or to Docker (everything but `[metadata]`, which stays host-side).
_CONTAINER_TOML_TABLES = ("task", "agent", "environment", "verifier", "solution")


class LabelLeak(NamedTuple):
    path: str  # relative to the task dir
    line: int
    text: str


def _label_lines(rel: str, text: str, ignore: frozenset[str] = frozenset()) -> list[LabelLeak]:
    return [
        LabelLeak(rel, i, ln.strip()[:200])
        for i, ln in enumerate(text.splitlines(), 1)
        if LABEL_RE.search(ln) and ln.strip() not in ignore
    ]


def label_leaks(task_dir: Path, source_dir: Path | None = None) -> list[LabelLeak]:
    """Lines of an emitted task that carry a label word and can reach the container, comments and file names included.

    Scans every text file under `environment/` (the Docker build context: Dockerfile, comments, scripts, file names)
    and the container-facing tables of task.toml. With `source_dir` (the seed task), lines the seed already had are
    not counted, so a source whose own Dockerfile says "version control" does not block the build; what the operators
    add is what is checked. Labels live only in `[metadata.honest]`, which this scan skips by design.
    """
    import tomllib

    out: list[LabelLeak] = []
    env = task_dir / "environment"
    src_env = source_dir / "environment" if source_dir is not None else None
    for f in sorted(env.rglob("*")) if env.is_dir() else []:
        rel = f.relative_to(task_dir).as_posix()
        out += _label_lines(rel + " (name)", "\n".join(rel.split("/")[1:]))
        if not f.is_file():
            continue
        text = _read_text(f, limit=2_000_000)
        seen: frozenset[str] = frozenset()
        if src_env is not None and (src_env / f.relative_to(env)).is_file():
            seen = frozenset(
                ln.strip() for ln in _read_text(src_env / f.relative_to(env), limit=2_000_000).splitlines()
            )
        out += _label_lines(rel, text, seen)
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        try:
            cfg = tomllib.loads(toml_path.read_text())
        except tomllib.TOMLDecodeError:
            cfg = {}
        for table in _CONTAINER_TOML_TABLES:
            if table in cfg:
                out += _label_lines(f"task.toml [{table}]", json.dumps(cfg[table], indent=0, default=str))
    return out


# ── where a sentence can go in an instruction (SC-4) ─────────────────────────────────────────────────────────────────

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|\(?[a-zA-Z][.)])\s+\S")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s|^\s*(?:\*\*|__)[^*_\n]+(?:\*\*|__):?\s*$|^[=-]{3,}\s*$")
_SENTENCE_END_RE = re.compile(r"""[.!?]["')\]”’*_]*\s*$""")


_EXAMPLE_CUE_RE = re.compile(
    r"\be\.g\.|\bfor example\b|\bfor instance\b|\bsuch as\b|\bsample\b|\bplaceholder\b|/some/|example", re.IGNORECASE
)


def example_paths(instruction: str) -> set[str]:
    """Paths that only illustrate something: inside a code fence, on a line with an example cue, or `.../example.*`.

    A control must not forbid one of these: "do not touch /some/path/example.txt" names an object that is not part of
    the task.
    """
    out: set[str] = set()
    in_fence = False
    for ln in instruction.splitlines():
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or _EXAMPLE_CUE_RE.search(ln):
            out.update(extract_paths(ln))
    return out


def _paragraph_list(instruction: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", instruction.strip("\n")) if p.strip()]


def safe_boundaries(instruction: str) -> list[int]:
    """Paragraph boundaries `i` (the new paragraph becomes number `i`) where a sentence can go without breaking the text.

    The paragraph before must be a finished sentence: not a heading, not a list (or a wrapped list item), not ending in a colon, not
    inside a code fence. The one after must not continue it (list item, indented block, table, lowercase start).
    Boundary 0 never qualifies (nothing precedes it). The old rule only avoided code fences and split a lead-in from
    its list or code block in about half of the variants (SC-4).
    """
    paras = _paragraph_list(instruction)
    out = []
    for i in range(1, len(paras) + 1):
        if "\n\n".join(paras[:i]).count("```") % 2:
            continue
        lines = paras[i - 1].rstrip().splitlines()
        last = lines[-1]
        finished = last.strip().startswith("```") or _SENTENCE_END_RE.search(last)
        heading = _HEADING_RE.match(last) or (len(lines) == 1 and _HEADING_RE.match(lines[0]))
        if heading or last.rstrip().endswith(":") or last.lstrip().startswith("|"):
            continue
        if any(_LIST_ITEM_RE.match(ln) for ln in lines):
            continue  # the paragraph is (or ends in) a list: the last line may be a wrapped item
        if not finished:
            continue
        if i < len(paras):
            first = paras[i].lstrip("\n").splitlines()[0]
            if _LIST_ITEM_RE.match(first) or first.startswith(("    ", "\t", "|")) or first.lstrip()[:1].islower():
                continue
        out.append(i)
    return out


# ── static halves of the per-operator filters ────────────────────────────────────────────────────────────────────────

_SCOPE_PATTERNS = (
    r"\bno other (?:files?|director(?:y|ies)|folders?|paths?|changes?|modifications?|locations?|artefacts?|artifacts?)\b",
    (
        r"\b(?:do not|don't|must not|should not|shall not|never|without|may not|cannot|can't)\b[^.\n]{0,60}"
        r"\b(?:other|another|anything else|unrelated|additional|any more)\b[^.\n]{0,30}"
        r"\b(?:files?|director(?:y|ies)|folders?|paths?)\b"
    ),
    r"\b(?:nothing|anything) else\b[^.\n]{0,50}\b(?:modified|changed|created|deleted|removed|touched|altered|written)\b",
    (
        r"\b(?:do not|don't|must not|should not|never|without)\b[^.\n]{0,40}\b(?:touch|modify|change|alter|delete|remove)\b"
        r"[^.\n]{0,30}\b(?:anything|everything) else\b"
    ),
    (
        r"\bleave\b[^.\n]{0,40}\b(?:everything|all|the rest|other \w+)\b[^.\n]{0,30}"
        r"\b(?:untouched|alone|as is|unchanged|intact)\b"
    ),
    r"\bonly (?:modify|change|touch|create|write|edit|delete|remove)\b|\b(?:modify|touch|change) only\b",
    (
        r"\b(?:do not|don't|must not|never)\b[^.\n]{0,40}\b(?:modify|touch|change|alter|delete|remove|create)\b[^.\n]{0,40}"
        r"\b(?:outside|beyond|other than|apart from|except)\b"
    ),
)
_SCOPE_RES = tuple(re.compile(p, re.IGNORECASE) for p in _SCOPE_PATTERNS)


def has_scope_clause(instruction: str) -> bool:
    """The instruction limits which files may change ("no other files may be modified"): a request to delete another
    directory then contradicts the task, so `high_risk` is not built on it (the act twin could not be told to act)."""
    return any(r.search(instruction) for r in _SCOPE_RES)


def token_overlap(content: str, text: str) -> float:
    """Share of the content's distinct words (3+ characters) that also occur in `text`."""
    toks = set(re.findall(r"[A-Za-z0-9_]{3,}", content.lower()))
    if not toks:
        return 0.0
    other = set(re.findall(r"[A-Za-z0-9_]{3,}", text.lower()))
    return len(toks & other) / len(toks)


def content_shape(content: str) -> tuple[int, int]:
    """(characters, non-blank lines) of a file's text."""
    return len(content.strip()), sum(1 for ln in content.splitlines() if ln.strip())


def symlink_targets(dockerfile: str) -> set[str]:
    """Paths the build makes with `ln` (a link-only target: deleting it removes a pointer, not data)."""
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
                if cmd == "ln" and plain:
                    p = _resolve(plain[-1], "/root", workdir)
                    if p:
                        out.add(p)
    return out


def _clears_other(mode: str, need_x: bool) -> bool:
    """Whether a chmod mode leaves "other" without read (or search, for a directory)."""
    need = 0o005 if need_x else 0o004
    if re.fullmatch(r"[0-7]{3,4}", mode):
        return (int(mode, 8) & need) != need
    cleared = False
    for clause in mode.split(","):
        m = re.fullmatch(r"([ugoa]*)([-+=])([rwxXst]*)", clause)
        if not m:
            continue
        who, op, perms = m.groups()
        if who and not set(who) & {"o", "a"}:
            continue
        hit = "r" in perms or (need_x and ("x" in perms or "X" in perms))
        if op == "-" and hit or op == "=" and not ("r" in perms and (not need_x or "x" in perms or "X" in perms)):
            cleared = True
        elif op == "+" and hit:
            cleared = False
    return cleared


def unreadable_inputs(facts: TaskFacts) -> list[str]:
    """Paths the build makes owner-only (`chmod 600/700/go-rwx`, `umask 077`) that the task reads or names.

    Running the agent as an unprivileged user (`revoke_permission`) would make these unreadable, so the variant would
    fail for a reason other than the one it labels. Heuristic and conservative: any such `chmod` on a path that is a
    task path, an image file, or a directory above one counts.
    """
    text = facts.dockerfile or ""
    needed = {*facts.instruction_paths, *facts.image_files, *facts.script.mentioned}
    flagged: list[str] = []
    if re.search(r"\bumask\s+0?[0-7]?77\b", text):
        flagged.append("umask")
    stripped, _ = _split_heredocs(text)
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
                if cmd not in ("chmod", "install"):
                    continue
                if cmd == "install":
                    mode = next((args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-m", "--mode")), None)
                    paths = [a for a in args if not a.startswith("-")][-1:]
                else:
                    plain = [a for a in args if not a.startswith("-") or re.fullmatch(r"-[rwxXst]+", a)]
                    mode, paths = (plain[0], plain[1:]) if plain else (None, [])
                if mode is None:
                    continue
                for tok in paths:
                    p = _resolve(tok, "/root", workdir)
                    if p is None:
                        continue
                    for n in needed:
                        if not paths_overlap(p, n):
                            continue
                        # a directory above the needed path must stay searchable, the path itself readable
                        if _clears_other(mode, need_x=p != n and n.startswith(p.rstrip("/") + "/")):
                            flagged.append(p)
                            break
    return list(dict.fromkeys(flagged))


_PASS_THROUGH = frozenset(
    {"cp", "mv", "ln", "ls", "test", "[", "[[", "touch", "rm", "chmod", "chown", "chgrp", "stat", "file", "find", "cd"}
)
_DIR_TEST_RE = re.compile(r"is_dir|isdir|listdir|iterdir|\bglob|rglob|scandir|os\.walk|\bls\b")
_FILE_TEST_RE = re.compile(r"read_text|read_bytes|\bopen\(|is_file|isfile|getsize|\.stat\(")


def operand_commands(script: str, home: str = "/root", cwd: str | None = None) -> dict[str, set[str]]:
    """For every absolute path the script names as an argument or `<` input: the commands that take it.

    Lets an operator tell a file the oracle reads from one it only moves, copies or lists (a pass-through the repair
    probe would also survive).
    """
    out: dict[str, set[str]] = {}
    body = _ARITH_RE.sub("0", strip_heredoc_bodies(script).replace("\\\n", " "))
    for seg in _segments(_tokens(body)):
        cmd, args, redirects = _command_of(seg)
        if cmd is None:
            continue
        for a in args:
            p = _resolve(a, home, cwd)
            if p:
                out.setdefault(p, set()).add(cmd)
        for op, target in redirects:
            if op in ("<", "<<<"):
                p = _resolve(target, home, cwd)
                if p:
                    out.setdefault(p, set()).add("cat")
    return out


def is_pass_through(commands: set[str]) -> bool:
    """Every command that takes the path only moves, copies, lists or tests it: nothing depends on its content."""
    return bool(commands) and commands <= _PASS_THROUGH


def is_link_only(content: str) -> bool:
    """A file that holds only pointers (URLs, absolute paths, `a -> b`): recreating it needs no knowledge."""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return bool(lines) and all(re.fullmatch(r"(?:https?://\S+|/\S+|\S+\s*->\s*\S+)", ln) for ln in lines)


def _mkdir_operands(script: str, home: str, cwd: str | None) -> set[str]:
    out: set[str] = set()
    for seg in _segments(_tokens(strip_heredoc_bodies(script).replace("\\\n", " "))):
        cmd, args, _r = _command_of(seg)
        if cmd == "mkdir":
            for a in (a for a in args if not a.startswith("-")):
                out.update(p for p in (_resolve(w, home, cwd) for w in _brace_expand(a)) if p)
    return out


def is_dir_like(facts: TaskFacts, path: str) -> bool | None:
    """Whether `path` is a directory in this task: True, False, or None when nothing decides it.

    Evidence, strongest first: the build makes it with `mkdir`/`WORKDIR`; the oracle `mkdir`s it; a test lists or
    walks it; the instruction writes it with a trailing slash or calls it a directory; a test reads it as a file; it
    has an extension.
    """
    path = posixpath.normpath(path)
    if path in boot_dirs(facts.dockerfile or ""):
        return True
    if any(paths_overlap(path, w) and path != w and w.startswith(path + "/") for w in facts.script.written):
        return True
    if path in _mkdir_operands(facts.solve or "", facts.home, facts.docker.workdir):
        return True
    for ln in facts.tests_text.splitlines():
        if (path in ln or posixpath.basename(path) in ln) and _DIR_TEST_RE.search(ln):
            return True
    if path + "/" in facts.instruction or re.search(
        rf"(?:director(?:y|ies)|folder)\W{{0,4}}`?{re.escape(path)}|{re.escape(path)}`?\W{{0,3}}(?:director(?:y|ies)|folder)\b",
        facts.instruction,
        re.IGNORECASE,
    ):
        return True
    if any(_FILE_TEST_RE.search(ln) and path in ln for ln in facts.tests_text.splitlines()):
        return False
    return False if posixpath.splitext(path)[1] else None
