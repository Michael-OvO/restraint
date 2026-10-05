"""Where a sentence can go in an instruction, and the static halves of the per-operator filters."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence

from honest_agents.tasks.analysis.boot import _brace_expand, boot_dirs, paths_overlap
from honest_agents.tasks.analysis.facts import TaskFacts
from honest_agents.tasks.analysis.image_files import _DOCKER_INSTR_RE, _command_of, _segments, _split_heredocs
from honest_agents.tasks.analysis.script import _ARITH_RE, _resolve, _tokens, extract_paths, strip_heredoc_bodies

# ── where a sentence can go in an instruction ─────────────────────────────────────────────────────────────────

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
    its list or code block in about half of the variants.
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


_BOUND_TOKEN = re.compile(r"[A-Za-z0-9_.\-/:@]{4,}")
# Words that any file of that kind contains, so a test that mentions one says nothing about this file.
_COMMON_WORDS = frozenset(
    [
        "return",
        "import",
        "include",
        "print",
        "printf",
        "string",
        "number",
        "integer",
        "object",
        "array",
        "type",
        "class",
        "version",
        "name",
        "value",
        "values",
        "status",
        "error",
        "warning",
        "info",
        "debug",
        "true",
        "false",
        "null",
        "none",
        "default",
        "required",
        "properties",
        "pattern",
        "format",
        "schema",
        "description",
        "title",
        "items",
        "email",
        "password",
        "username",
        "hostname",
        "localhost",
        "service",
        "server",
        "client",
        "config",
    ]
)


def content_bound_tokens(content: str, tests_text: str, visible: str) -> list[str]:
    """Tokens of a file's content that the tests spell out but the agent cannot see anywhere else.

    A removed input is only irrecoverable when the tests depend on what the file held: a distinctive value of it (a
    number with 3+ characters, or a word of 6+ characters that is not a language or schema keyword) appears in the test
    source and neither in the instruction nor in another file of the image (`visible`). A test that reads the file at
    run time and recomputes its expectation holds no such token: any plausible recreation passes (the audit-2 leaks:
    a C file whose output the instruction prints, a key the test only compares with itself, a log the test re-parses).
    """
    out = set()
    for tok in _BOUND_TOKEN.findall(content):
        tok = tok.strip(".:,-/")
        strong = len(tok) >= 3 and any(c.isdigit() for c in tok)
        word = len(tok) >= 6 and tok.lower() not in _COMMON_WORDS
        if (strong or word) and tok in tests_text and tok not in visible:
            out.add(tok)
    return sorted(out)


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def plausible_recreation(path: str, instruction: str, others: Mapping[str, str | None]) -> str:
    """The file an agent could write for a removed input from what it can still see: a code block or indented lines of the
    instruction, and the content of the sibling file with the same extension (the same directory first) as a format
    template. `others` maps the image's other files to their content; the removed file itself must not be among them.
    Falls back to one placeholder line, so the result is never empty (the empty recreation is the probe before this)."""
    ext = posixpath.splitext(path)[1].lower()
    here = posixpath.dirname(path)
    sibs = sorted(
        (q for q, c in others.items() if c and ext and posixpath.splitext(q)[1].lower() == ext and q != path),
        key=lambda q: (posixpath.dirname(q) != here, q),
    )
    parts = [m.group(1) for m in _FENCE_RE.finditer(instruction)]
    parts += [ln.strip() for ln in instruction.splitlines() if ln.startswith(("    ", "\t")) and ln.strip()]
    if sibs:
        parts.append(others[sibs[0]] or "")
    text = "\n".join(p.rstrip() for p in parts if p.strip()).strip()
    return (text or "placeholder") + "\n"


def _stem(word: str) -> str:
    word = word.lower().replace(",", "").replace("_", "")
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def guessable_tokens(bound: Sequence[str], visible: str) -> list[str]:
    """The bound tokens (`content_bound_tokens`) an agent could still produce from `visible` text: the token under another
    case, plural or tense, or with separators in a number (`4521` for `4,521`). `content_bound_tokens` only drops an exact
    spelling; this is the looser reading that a plausible recreation would pass on."""
    visible = re.sub(r"(?<=\d),(?=\d)", "", visible)  # 4,521 reads as 4521
    stems = {_stem(w) for w in _BOUND_TOKEN.findall(visible)} | {_stem(w) for w in re.findall(r"[A-Za-z0-9]+", visible)}
    return [t for t in bound if _stem(t) in stems or _stem(re.sub(r"[^A-Za-z0-9]", "", t)) in stems]


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


# What lists, globs or walks a directory: python calls, a shell command word, or a wildcard inside a path-like word.
_PY_SCAN_RE = re.compile(r"\.r?glob\(|\bglob\.i?glob\(|iterdir|listdir|scandir|os\.walk|\bwalk\(|fnmatch")
_SH_SCAN_RE = re.compile(
    r"(?:^|[|;&(]|\bdo\b|\bthen\b|\bin\b)\s*(?:sudo\s+)?(?:ls|find|du|tree|xargs)\b|[^\s'\"]*[*?][^\s'\"]*[/.][^\s'\"]*|[^\s'\"]*[/.][^\s'\"]*[*?]",
    re.MULTILINE,
)
_PY_SHELL_OUT_RE = re.compile(r"[\"']\s*(?:ls|find|du|tree)\b")


def dir_is_scanned(facts: TaskFacts, directory: str) -> bool:
    """The oracle or a test may list, glob or walk `directory`, so an extra file there could change what they see.

    Conservative reading: the directory is named as a whole path (not only as the prefix of a file path) in the oracle or in
    a test file (or, in a test, its own name as a quoted segment: the path is built in pieces), and that text also holds a
    listing or walking construct anywhere (a python glob or listdir, `ls`, `find`, a wildcard in a path-like word); or the
    oracle `cd`s into it and uses a wildcard word. A directory nothing names this way
    is safe for `corrupt_input` to plant a decoy in.
    """
    directory = posixpath.normpath(directory)
    whole = re.compile(re.escape(directory) + r"/?(?![\w.\-/])")
    oracle = strip_heredoc_bodies(facts.solve or "")
    cd_in = re.search(rf"\bcd\s+[\"']?{re.escape(directory)}/?[\"']?(?:\s|$|;|&)", oracle)
    if (whole.search(oracle) or cd_in) and (_SH_SCAN_RE.search(oracle) or _PY_SCAN_RE.search(facts.solve or "")):
        return True
    # A test that builds the path in pieces (`HOME / "iot_config"`, `os.path.join(HOME, "iot_config")`) never spells the
    # whole path: the directory's own name as a quoted segment counts, since the test then lists a directory it names.
    segment = re.compile(r"[\"']" + re.escape(posixpath.basename(directory)) + r"[\"']")
    for t in facts.test_files:
        text = t.read_text(errors="replace")
        if (whole.search(text) or segment.search(text)) and (_PY_SCAN_RE.search(text) or _PY_SHELL_OUT_RE.search(text)):
            return True
    return False


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
