"""Files the image build creates and what they contain (the `remove_input` filter)."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from honest_agents.tasks.analysis.script import (
    _REDIRECTS,
    _WRITE_REDIRECTS,
    _is_separator,
    _read_text,
    _resolve,
    _tokens,
)
from honest_agents.tasks.analysis.tables import _ASSIGN_RE, _HEREDOC_RE, WRAPPERS

# ── docs/design.md section 7: files the image build creates (remove_input) ───────────────

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
    # For a generated file: the data the generating command was given when the Dockerfile spells it out (the heredoc
    # body of `sqlite3 db <<SQL`, the SQL argument of `sqlite3 db "INSERT ..."`, a heredoc fed to another writer). Not
    # the bytes, but every value the file holds comes from it, so the `remove_input` content checks read it (DEF-26).
    generator: str | None = None

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


def _segments(tokens: Sequence[str]) -> list[list[str]]:
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

    def put(
        path: str | None,
        kind: str,
        content: str | None = None,
        nonempty: bool | None = None,
        *,
        cwd=None,
        generator: str | None = None,
    ) -> None:
        p = _resolve(expand(path or ""), "/root", cwd if cwd is not None else workdir)
        if not p:
            return
        old = files.get(p)
        if kind == FILE_TOUCHED and old is not None:
            return  # `touch -d ...` on an existing file only changes its mtime
        if kind == FILE_TOUCHED:
            nonempty = False  # `touch` / `: >` leave an empty file
        files[p] = ImageFile(p, kind, content, bool(content.strip()) if content is not None else nonempty, generator)

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
                        files[dst] = ImageFile(dst, src.kind, src.content, src.nonempty, src.generator)
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
                sql = " ".join(plain[1:]) or None
                put(plain[0], FILE_GENERATED, nonempty=True if len(plain) > 1 else None, cwd=cwd, generator=sql)
            elif cmd in _EMPTY_WRITERS:
                for _op, target in writes:
                    put(target, FILE_TOUCHED, cwd=cwd)
            else:
                for _op, target in writes:
                    put(target, FILE_GENERATED, cwd=cwd)

    def heredoc(opener: str, body: str) -> None:
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
                put(plain[0], FILE_GENERATED, nonempty=bool(body.strip()), generator=body)
            elif writes:
                put(writes[0], FILE_GENERATED, generator=body)

    pending = 0
    for raw in stripped.replace("\\\n", " ").splitlines():
        m = _DOCKER_INSTR_RE.match(raw)
        if not m:
            continue
        instr, rest = m.group(1).upper(), m.group(2)
        if instr == "WORKDIR":
            workdir = _resolve(expand(rest.strip()), "/root", workdir) or workdir
        elif instr == "RUN":
            run(rest)
            # Heredoc bodies are applied in Dockerfile order (the opener line was skipped by `run`), so a later
            # `cp` / `mv` of a heredoc file sees its content.
            while pending < len(docs) and docs[pending][0].rstrip("\\ \t").strip() in raw:
                heredoc(*docs[pending])
                pending += 1
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

    for doc in docs[pending:]:  # an opener the loop could not match to a RUN line
        heredoc(*doc)
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
