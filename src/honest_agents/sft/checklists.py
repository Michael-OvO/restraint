"""Requirement checklists for the commitment-point data (iteration 16): typed loader, schema check and leak audit.

One JSON file per training-pool task under `data/commit_checks/checklists/<task>.json` (task = the variant / task
directory name under `data/decision/surfaces/tasks`), written by Claude from the task's `instruction.md` only:

    {"task": str, "source": str, "author": "claude", "inputs": ["instruction.md"],
     "requirements": [{"id": str, "quote": str, "check": {"cmd": str, "pass_if": str}}, ...]}

No other key is allowed at any level. `pass_if` is `rc==0`, `stdout_empty` or `stdout_contains:<s>`. Requirement ids
are unique within a file, the file name is the task, every command is read-only (`mutation.is_mutating` and
`readme_read.is_destructive_on` with no target are both false) and, when the task directory is at hand, every quote
is verbatim in its `instruction.md` (whitespace-normalised). `sft.commit_checks` reads the same format.

**Leak audit.** A checklist must not know more than the instruction says. A check leaks when its command (or
`pass_if`) holds a specific string that occurs in the task's hidden tests (`tests/`) but not in its `instruction.md`:
a path component, a file name, a magic value. The strings are the `[A-Za-z0-9_.-]` runs of the command after shell
escapes (`\\n`, `\\t`, ...) are blanked, outer `.`/`-` stripped; a run is specific when it is a number or has at least
three characters and is not generic vocabulary (`GENERIC`: Python keywords, builtins, the standard library's module
names and the attributes of the modules and types checks call, SQL keywords, shell command names, system path parts
and the local names checks give their own values; regex ranges such as `0-9a-f`; the numbers 0-9, which every check
uses as exit codes, fields and line numbers, so a one-digit magic value is not caught); a dotted run is generic when
its first part is a standard module or all of its parts are generic. Words match case-insensitively as substrings
(`company_` is in `company_YYYYMMDD.sql`), numbers as whole numbers; the instruction is read with its Markdown escapes
removed (`raw\\_data` is `raw_data`). Quotes are not audited: they are verbatim instruction text by the schema. The
rule is strict on purpose: a format string the author knew from general knowledge (a hash length, a `docker ps`
column) that the tests also check but the instruction never names is dropped too.

**Audit** (`audit`): each authored requirement is dropped when it is invalid (`parse_requirement` against the
instruction: not read-only by the conservative classifiers above, quote not verbatim, malformed) or leaks; the kept
ones are written to `data/commit_checks/checklists/` and every drop is recorded with its reason in
`data/commit_checks/leak_audit.json`, which also pins the sha256 of each instruction a kept file was audited on. A file
left with no requirement is dropped whole. `destructive` (an authoring mark on some destructive-request requirements,
not part of the format) is stripped and counted. Dropping per requirement, not per file, keeps the rest of a file that
`sft.commit_checks` would otherwise refuse whole.

    python -m honest_agents.sft.checklists audit --src DIR [--out data/commit_checks/checklists] [--tasks-root ROOT]
    python -m honest_agents.sft.checklists validate [--dir data/commit_checks/checklists] [--tasks-root ROOT]
"""

from __future__ import annotations

import argparse
import builtins
import configparser
import csv
import datetime
import glob
import gzip
import hashlib
import json
import keyword
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honest_agents.mutation import is_mutating
from honest_agents.readme_read import is_destructive_on

DEFAULT_DIR = Path("data/commit_checks/checklists")
DEFAULT_AUDIT = Path("data/commit_checks/leak_audit.json")
DEFAULT_TASKS_ROOT = Path("data/decision/surfaces/tasks")
AUTHOR = "claude"
INPUTS = ["instruction.md"]
PASS_IF_RC0 = "rc==0"
PASS_IF_EMPTY = "stdout_empty"
PASS_IF_CONTAINS = "stdout_contains:"
TOP_KEYS = frozenset({"task", "source", "author", "inputs", "requirements"})
REQ_KEYS = frozenset({"id", "quote", "check"})
CHECK_KEYS = frozenset({"cmd", "pass_if"})


class ChecklistError(ValueError):
    """A checklist file breaks the schema or the instruction-only rule."""


@dataclass(frozen=True)
class Requirement:
    id: str
    quote: str
    cmd: str
    pass_if: str

    def passes(self, rc: int | None, output: str) -> bool:
        if self.pass_if == PASS_IF_RC0:
            return rc == 0
        if self.pass_if == PASS_IF_EMPTY:
            return rc is not None and not output.strip()
        return rc is not None and self.pass_if[len(PASS_IF_CONTAINS) :] in output


@dataclass(frozen=True)
class Checklist:
    task: str
    source: str
    requirements: tuple[Requirement, ...]
    sha: str = ""  # sha256 of the file bytes, first 16 hex digits

    @property
    def commands(self) -> list[str]:
        return [r.cmd for r in self.requirements]


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


def _valid_pass_if(p: Any) -> bool:
    return isinstance(p, str) and (
        p in (PASS_IF_RC0, PASS_IF_EMPTY) or (p.startswith(PASS_IF_CONTAINS) and len(p) > len(PASS_IF_CONTAINS))
    )


def _str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _check_top(data: Any) -> None:
    if not isinstance(data, Mapping):
        raise ChecklistError("not a JSON object")
    if set(data) != TOP_KEYS:
        raise ChecklistError(f"keys {sorted(data)} != {sorted(TOP_KEYS)}")
    if not (_str(data["task"]) and _str(data["source"])):
        raise ChecklistError("task and source must be non-empty strings")
    if data["author"] != AUTHOR:
        raise ChecklistError(f"author {data['author']!r} is not {AUTHOR!r}")
    if data["inputs"] != INPUTS:
        raise ChecklistError(f"inputs {data['inputs']!r}: a checklist is written from instruction.md only")
    if not isinstance(data["requirements"], list) or not data["requirements"]:
        raise ChecklistError("no requirements")


def parse_requirement(r: Any, instruction: str | None = None) -> Requirement:
    """One requirement after the schema and read-only checks; with `instruction`, the quote must be verbatim in it."""
    if not isinstance(r, Mapping) or set(r) != REQ_KEYS or not isinstance(r["check"], Mapping):
        raise ChecklistError(f"malformed requirement {r!r}")
    if set(r["check"]) != CHECK_KEYS:
        raise ChecklistError(f"malformed check {r['check']!r}")
    rid, quote, cmd, pass_if = r["id"], r["quote"], r["check"]["cmd"], r["check"]["pass_if"]
    if not (_str(rid) and _str(quote)):
        raise ChecklistError(f"malformed requirement {r!r}")
    if not _str(cmd):
        raise ChecklistError(f"{rid}: empty command")
    if not _valid_pass_if(pass_if):
        raise ChecklistError(f"{rid}: pass_if {pass_if!r}")
    if is_mutating(cmd) or is_destructive_on(cmd, []):
        raise ChecklistError(f"{rid}: command is not read-only: {cmd!r}")
    if instruction is not None and _norm_ws(quote) not in _norm_ws(instruction):
        raise ChecklistError(f"{rid}: quote is not verbatim in instruction.md")
    return Requirement(rid, quote, cmd.strip(), pass_if)


def parse_checklist(data: Any, instruction: str | None = None, *, sha: str = "") -> Checklist:
    """A checklist after the schema check; with `instruction`, every quote must also be verbatim in it."""
    _check_top(data)
    out = tuple(parse_requirement(r, instruction) for r in data["requirements"])
    if len({r.id for r in out}) != len(out):
        raise ChecklistError("duplicate requirement ids")
    return Checklist(data["task"], data["source"], out, sha)


def file_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def read_instruction(task: str, tasks_root: Path | None) -> str | None:
    p = tasks_root / task / "instruction.md" if tasks_root is not None else None
    return p.read_text() if p is not None and p.is_file() else None


def load_checklist(path: str | Path, tasks_root: str | Path | None = None) -> Checklist:
    """One file: schema, file name = task, and the quotes against the instruction when `tasks_root` holds the task."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ChecklistError(f"not JSON: {e}") from e
    task = data.get("task") if isinstance(data, Mapping) else None
    instruction = read_instruction(task, Path(tasks_root)) if isinstance(task, str) and tasks_root else None
    cl = parse_checklist(data, instruction, sha=file_sha(raw))
    if path.name != f"{cl.task}.json":
        raise ChecklistError(f"file {path.name} names task {cl.task!r}")
    return cl


def load_checklists(
    directory: str | Path = DEFAULT_DIR, tasks_root: str | Path | None = None
) -> tuple[dict[str, Checklist], dict[str, str]]:
    """task -> checklist for every valid file of `directory`; file name -> error for the invalid ones."""
    good: dict[str, Checklist] = {}
    bad: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            cl = load_checklist(path, tasks_root)
            good[cl.task] = cl
        except ChecklistError as e:
            bad[path.name] = str(e)
    return good, bad


# ── leak audit ───────────────────────────────────────────────────────────────

_RUN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_SHELL_ESC = re.compile(r"\\[ntr0abfve]")
_MD_ESC = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~])")
_SQL = (
    "select from where insert into values update delete create table drop alter index view order group by having "
    "limit offset join left right inner outer on as and or not null is in like glob between distinct count sum avg "
    "min max case when then else end primary key integer text real blob default unique exists pragma begin commit "
    "transaction rollback union all asc desc cast coalesce ifnull round length lower upper substr replace trim "
    "sqlite_master sqlite_schema table_info integrity_check foreign_keys"
)
_CODE = (  # keyword arguments, URI options and the local names checks use for their own values and messages
    "mode uri ro rw encoding errors newline separators ensure_ascii indent sort_keys exp expected got actual want "
    "missing bad ok err out res row rows col cols line lines raw data perm perms val value blocks block"
)
_SHELL = (
    "awk bash basename cat cd chmod chown cmp column comm cp cut date dd df diff dirname du echo egrep env exec expr "
    "false fgrep file find fold getent grep gunzip gzip head hexdump id iconv jq join less ln locale ls md5sum mkdir "
    "mktemp mv nl nproc numfmt od paste perl pgrep printf ps pwd python python3 readlink realpath rev rm sed seq "
    "sha1sum sha256sum sh shasum sort sqlite3 stat sum tac tail tar tee test timeout touch tr true truncate type "
    "uname uniq unzip wc which whoami xargs xxd yes zcat zgrep gawk mawk tput ip ss netstat crontab systemctl "
    "openssl curl wget git ssh ssh-keygen docker sudo su useradd groups bc dc factor shuf split base64 "
    "local declare export read shift set unset return function done do then fi elif esac for while until if "
    "dev null stdin stdout stderr tmp proc etc usr bin sbin lib var opt home user root lc_all lang tz path utf-8 utf8"
)
_MODULES: tuple[Any, ...] = (
    os, os.path, re, json, csv, sqlite3, sqlite3.Connection, sqlite3.Cursor, hashlib, configparser,
    configparser.ConfigParser, subprocess, pathlib.Path, datetime, datetime.datetime, glob, sys, gzip, tarfile,
    hashlib.sha256(), str, bytes, list, dict, set, tuple, int, float,
)  # fmt: skip


def _generic_words() -> frozenset[str]:
    words: set[str] = set(keyword.kwlist) | set(dir(builtins)) | set(sys.stdlib_module_names)
    for m in _MODULES:
        words |= {a for a in dir(m) if not a.startswith("__")}
    words |= set(_SQL.split()) | set(_SHELL.split()) | set(_CODE.split())
    return frozenset(w.casefold() for w in words)


GENERIC = _generic_words()
_STDLIB = frozenset(m.casefold() for m in sys.stdlib_module_names)


_RANGES = re.compile(r"(?:[0-9A-Za-z]-[0-9A-Za-z])+")  # regex character-class ranges: `0-9`, `0-9a-f`


def is_generic(run: str) -> bool:
    k = run.casefold()
    if k in GENERIC or _RANGES.fullmatch(k) or (k.isdigit() and int(k) < 10):
        return True
    if "." in k and not k.replace(".", "").isdigit():
        parts = [p for p in k.split(".") if p]
        return bool(parts) and (parts[0] in _STDLIB or all(p in GENERIC for p in parts))
    return False


def specific_strings(cmd: str) -> list[str]:
    """The specific runs of a command (module docstring), sorted, unique."""
    out: set[str] = set()
    for m in _RUN.finditer(_SHELL_ESC.sub(" ", cmd)):
        run = m.group(0).strip(".-")
        if run and (run.isdigit() or len(run) >= 3) and not is_generic(run):
            out.add(run)
    return sorted(out)


def _occurs(run: str, text: str) -> bool:
    """`text` is already casefolded."""
    if run.isdigit():
        return re.search(rf"(?<![0-9]){run}(?![0-9])", text) is not None
    return run.casefold() in text


def instruction_text(instruction: str) -> str:
    return _MD_ESC.sub(r"\1", instruction).casefold()


def tests_text(task_dir: Path) -> str:
    d = task_dir / "tests"
    files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    return "\n".join(p.read_text(errors="replace") for p in files).casefold()


def leaked_strings(req: Requirement, instruction: str, tests: str) -> list[str]:
    """Specific strings of `req`'s command and pass_if that are in `tests` but not in `instruction` (both as returned
    by `instruction_text` / `tests_text`)."""
    runs = specific_strings(req.cmd + " " + (req.pass_if if req.pass_if.startswith(PASS_IF_CONTAINS) else ""))
    return [r for r in runs if _occurs(r, tests) and not _occurs(r, instruction)]


DROP_LEAK = "leak"
DROP_INVALID = "invalid"
STRIPPED_KEYS = ("destructive",)  # an authoring mark on some destructive-request requirements, not in the format


def audit_file(data: Mapping[str, Any], task_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    """(the file with its failing requirements removed, or None when none is left; the dropped requirements with
    their reason; the number of `STRIPPED_KEYS` removed). A requirement is dropped when it fails `parse_requirement`
    against the instruction (`invalid`: not read-only, quote not verbatim, malformed) or leaks (`leak`)."""
    _check_top(data)
    raw_ins = (task_dir / "instruction.md").read_text()
    ins = instruction_text(raw_ins)
    tests = tests_text(task_dir)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    stripped = 0
    for r in data["requirements"]:
        if isinstance(r, Mapping) and any(k in r for k in STRIPPED_KEYS):
            stripped += 1
            r = {k: v for k, v in r.items() if k not in STRIPPED_KEYS}
        try:
            req = parse_requirement(r, raw_ins)
        except ChecklistError as e:
            rid = r.get("id") if isinstance(r, Mapping) else None
            dropped.append({"task": data["task"], "id": rid, "reason": DROP_INVALID, "detail": str(e)})
            continue
        leaks = leaked_strings(req, ins, tests)
        if leaks:
            dropped.append({"task": data["task"], "id": req.id, "reason": DROP_LEAK, "leaked": leaks, "cmd": req.cmd})
        else:
            kept.append(dict(r))
    if len({r["id"] for r in kept}) != len(kept):
        raise ChecklistError(f"{data['task']}: duplicate requirement ids")
    return ({**data, "requirements": kept} if kept else None), dropped, stripped


def audit(src: Path, out: Path, tasks_root: Path) -> dict[str, Any]:
    """Audit every checklist of `src` against its task (`audit_file`), write the kept files to `out` and return the
    audit record. A file whose top level fails the schema, or whose task has no instruction, is not written; it is
    listed under `file_failures`."""
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.json"))
    file_failures: dict[str, str] = {}
    dropped: list[dict[str, Any]] = []
    dropped_files: list[str] = []
    n_req = n_kept = n_stripped = 0
    pins: dict[str, str] = {}
    for path in files:
        try:
            data = json.loads(path.read_bytes())
            _check_top(data)
            if path.name != f"{data['task']}.json":
                raise ChecklistError(f"file {path.name} names task {data['task']!r}")
            if read_instruction(data["task"], tasks_root) is None:
                raise ChecklistError(f"no instruction.md for {data['task']} under {tasks_root}")
            clean, gone, stripped = audit_file(data, tasks_root / data["task"])
        except (ChecklistError, json.JSONDecodeError) as e:
            file_failures[path.name] = str(e)
            continue
        n_req += len(data["requirements"])
        n_stripped += stripped
        dropped += gone
        if clean is None:
            dropped_files.append(data["task"])
            continue
        n_kept += len(clean["requirements"])
        (out / path.name).write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n")
        pins[data["task"]] = hashlib.sha256((tasks_root / data["task"] / "instruction.md").read_bytes()).hexdigest()
    by_reason = {k: sum(d["reason"] == k for d in dropped) for k in (DROP_LEAK, DROP_INVALID)}
    return {
        "note": "Audit of the instruction-only checklists (`python -m honest_agents.sft.checklists audit`, module "
        "docstring): a requirement is dropped when it is invalid (not read-only by `mutation.is_mutating` / "
        "`readme_read.is_destructive_on`, quote not verbatim, malformed) or leaks (its command holds a specific string "
        "that is in the task's hidden tests but not in its instruction.md). `instruction_sha256` pins the instruction "
        "each kept file was audited on.",
        "files": len(files),
        "file_failures": file_failures,
        "requirements": n_req,
        "dropped_by_reason": by_reason,
        "leak_rate": round(by_reason[DROP_LEAK] / n_req, 4) if n_req else None,
        "stripped_keys": {k: n_stripped for k in STRIPPED_KEYS},
        "kept_files": len(pins),
        "kept_requirements": n_kept,
        "mean_kept_per_task": round(n_kept / len(pins), 2) if pins else None,
        "dropped_files": dropped_files,
        "dropped": dropped,
        "instruction_sha256": dict(sorted(pins.items())),
    }


def counts(checklists: Mapping[str, Checklist]) -> dict[str, Any]:
    n = sum(len(c.requirements) for c in checklists.values())
    return {"tasks": len(checklists), "requirements": n, "mean_per_task": round(n / len(checklists), 2) if n else 0}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.checklists", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="validate, drop leaking checks, write the kept files and the audit record")
    a.add_argument("--src", required=True, type=Path, help="directory of authored checklist files")
    a.add_argument("--out", type=Path, default=DEFAULT_DIR)
    a.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="audit record to write")
    a.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    v = sub.add_parser("validate", help="schema-check every file (and the quotes when the task dirs are present)")
    v.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    v.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.command == "audit":
        rec = audit(args.src, args.out, args.tasks_root)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("dropped", "instruction_sha256", "note")}))
        return 1 if rec["file_failures"] else 0
    root = args.tasks_root if args.tasks_root.is_dir() else None
    good, bad = load_checklists(args.dir, root)
    print(json.dumps({**counts(good), "schema_failures": bad, "quotes_checked": root is not None}))
    return 1 if bad else 0


if __name__ == "__main__":
    from honest_agents.sft.checklists import main as _main  # `python -m` runs this file as __main__

    sys.exit(_main())
