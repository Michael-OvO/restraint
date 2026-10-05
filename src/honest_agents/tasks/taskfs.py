"""Filesystem helpers for Harbor task dirs: TOML round-trip, copying, Dockerfile edits, `[metadata.honest]` stamping."""

from __future__ import annotations

import json
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

from honest_agents.spec import HonestTaskSpec

# Files that are never part of a task: OS junk and caches. Model trajectories shipped next to some oracles are dropped
# by the source normaliser, not here (they are source-specific).
JUNK_PATTERNS = (".DS_Store", "__pycache__", "*.pyc", ".git")

HONEST_MARKER = "# honest-agents"

# Endless Terminals ships the expected state next to the Dockerfile (`environment/task.json`, `test_initial_state.py`,
# `container.def`). Harbor uses `environment/` as the docker build context, so anything left there is one `COPY` (or
# one `COPY .`) away from being inside the image the agent can read. They are deleted on normalisation.
ENV_SCRUB_FILES = ("task.json", "test_initial_state.py", "container.def")
# Dockerfile COPY/ADD sources that put tests, expected output, solutions or task metadata into the image.
_FORBIDDEN_SOURCE_RE = re.compile(
    r"(^|/)(tests?|solutions?)(/|$)|(^|/)(test_[^/]*|[^/]*_tests?\.py|solve\.sh|task\.(json|toml)|container\.def)$"
    r"|expected|_perturbation|golden|ground_?truth|oracle",
    re.IGNORECASE,
)
_COPY_ADD_RE = re.compile(r"^\s*(?:COPY|ADD)\s+(.*)$", re.IGNORECASE | re.MULTILINE)


_HEREDOC_OPERATOR = re.compile(r"<<(-?)[ \t]+(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


def normalize_heredocs(dockerfile: str) -> str:
    """Write every `<< 'EOF'` heredoc operator as `<<'EOF'`.

    Docker's BuildKit only recognises a heredoc when the delimiter follows `<<` directly; with a space it reads the body
    as Dockerfile instructions (`unknown instruction: print(...)`). Endless Terminals' Dockerfiles use the spaced form
    because the dataset ran them as Apptainer shell scripts, so about one source in twelve does not build at all on a
    real Docker daemon. Only an operator whose delimiter reappears alone on a later line is rewritten, so shell
    arithmetic such as `$((1 << 4))` and plain redirections are left alone.
    """
    lines = dockerfile.split("\n")
    for i, line in enumerate(lines):
        for m in reversed(list(_HEREDOC_OPERATOR.finditer(line))):
            word = m.group(3)
            if any(later.strip() == word for later in lines[i + 1 :]):
                line = line[: m.start()] + f"<<{m.group(1)}{m.group(2)}{word}{m.group(2)}" + line[m.end() :]
        lines[i] = line
    return "\n".join(lines)


def scrub_environment(task_dir: Path) -> list[str]:
    """Delete the expected-state files from `environment/`; returns the relative paths removed."""
    env = task_dir / "environment"
    removed = []
    for name in ENV_SCRUB_FILES:
        for f in sorted(env.rglob(name)) if env.is_dir() else []:
            if f.is_file() or f.is_symlink():
                f.unlink()
                removed.append(str(f.relative_to(task_dir)))
    return removed


def copy_sources(dockerfile: str) -> list[str]:
    """Source operands of every COPY/ADD (not `COPY --from=<stage>`), JSON-array form included."""
    out: list[str] = []
    for m in _COPY_ADD_RE.finditer(dockerfile.replace("\\\n", " ")):
        body = m.group(1).strip()
        if body.startswith("["):
            try:
                parts = [str(x) for x in json.loads(body)]
            except ValueError:
                parts = body.strip("[]").replace('"', "").replace(",", " ").split()
            flags: list[str] = []
        else:
            toks = body.split()
            flags = [t for t in toks if t.startswith("--")]
            parts = [t for t in toks if not t.startswith("--")]
        if any(f.startswith("--from") for f in flags):
            continue
        out += parts[:-1]  # the last operand is the destination
    return out


def forbidden_copy_sources(dockerfile: str) -> list[str]:
    """COPY/ADD sources that would ship tests, expected files, solutions or task.json into the image."""
    return [s for s in copy_sources(dockerfile) if _FORBIDDEN_SOURCE_RE.search(s)]


_BROAD_SOURCES = {".", "./", "*", "./*", "/", ".*"}


def forbidden_context_files(env_dir: Path, dockerfile: str) -> list[str]:
    """Files in the build context that a wildcard `COPY . /x` would ship although they look like tests or expected
    output. Empty unless the Dockerfile copies the whole context. The scrubbed expected-state files are not listed (they
    are deleted on normalisation, and `hermetic.image_leak_problems` flags them if they remain)."""
    if not any(src in _BROAD_SOURCES for src in copy_sources(dockerfile)) or not env_dir.is_dir():
        return []
    return sorted(
        str(f.relative_to(env_dir))
        for f in env_dir.rglob("*")
        if f.is_file()
        and not f.name.startswith("Dockerfile")
        and f.name not in ENV_SCRUB_FILES
        and _FORBIDDEN_SOURCE_RE.search(str(f.relative_to(env_dir)))
    )


def dumps_toml(data: dict[str, Any]) -> str:
    """Serialise TOML with whichever writer is installed.

    `tomli_w` / `tomlkit` are preferred; `toml` (0.10, a Harbor dependency) is the guaranteed fallback.
    """
    data = _drop_none(data)
    try:
        import tomli_w

        return tomli_w.dumps(data)
    except ImportError:
        pass
    try:
        import tomlkit

        return tomlkit.dumps(data)
    except ImportError:
        pass
    import toml

    return toml.dumps(data)


def _drop_none(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _drop_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_drop_none(x) for x in v if x is not None]
    return v


def read_task_toml(task_dir: Path) -> dict[str, Any]:
    return tomllib.loads((task_dir / "task.toml").read_text())


def write_task_toml(task_dir: Path, data: dict[str, Any]) -> None:
    text = dumps_toml(data)
    tomllib.loads(text)  # fail loudly here rather than at Harbor load time
    (task_dir / "task.toml").write_text(text)


def copy_task(src: Path, dst: Path) -> Path:
    """Copy a task dir to `dst` (replacing it), skipping junk files."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*JUNK_PATTERNS), symlinks=True)
    return dst


def set_honest(cfg: dict[str, Any], spec: HonestTaskSpec, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return `cfg` with `[metadata.honest]` = spec (and optional `[metadata.honest_variant]` operator details)."""
    md = dict(cfg.get("metadata") or {})
    md["honest"] = spec.model_dump(mode="json", exclude_none=True)
    if detail:
        md["honest_variant"] = detail
    else:
        md.pop("honest_variant", None)
    return {**cfg, "metadata": md}


def set_task_name(cfg: dict[str, Any], name: str, description: str | None = None) -> dict[str, Any]:
    """Give the task a unique `[task].name` (Harbor's `org/name`), keeping other package fields."""
    task = dict(cfg.get("task") or {})
    task["name"] = name
    if description is not None and not task.get("description"):
        task["description"] = description
    return {**cfg, "task": task}


_USER_RE = re.compile(r"^\s*USER\s+(\S+)", re.IGNORECASE | re.MULTILINE)
_FROM_RE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?", re.IGNORECASE | re.MULTILINE)


def final_stage(dockerfile: str) -> str:
    """Text of the stages that make up the final image: the last stage, prefixed by the stages it builds `FROM`.

    `USER`/`WORKDIR`/installs of an unrelated builder stage do not reach the final image, so reading them off the whole
    file would e.g. restore a builder's `USER` onto the final stage.
    """
    text = dockerfile.replace("\\\n", " ")
    froms = list(_FROM_RE.finditer(text))
    if not froms:
        return text
    stages = [
        (m.group(1).lower(), (m.group(2) or "").lower(), text[m.start() : nxt])
        for m, nxt in zip(froms, [f.start() for f in froms[1:]] + [len(text)], strict=True)
    ]
    chain = [stages[-1]]
    seen = {len(stages) - 1}
    while True:
        base = chain[0][0]
        idx = next(
            (i for i in range(len(stages)) if stages[i][1] == base and i not in seen),
            None,
        )
        if idx is None:
            break
        seen.add(idx)
        chain.insert(0, stages[idx])
    return "".join(s[2] for s in chain)


def final_user(dockerfile: str) -> str | None:
    users = _USER_RE.findall(final_stage(dockerfile))
    return users[-1] if users else None


def append_dockerfile(task_dir: Path, tag: str, run_lines: list[str]) -> None:
    """Append `RUN` steps to the final stage, as root, restoring the image's original final `USER` afterwards.

    Appending (instead of editing the original steps) keeps the perturbation auditable and lets Docker reuse the
    original layers.
    """
    path = task_dir / "environment" / "Dockerfile"
    text = path.read_text()
    user = final_user(text)
    # No label comment: a Dockerfile is agent-visible if a COPY ever ships it, so `tag` is only for the caller's logs.
    block = [""]
    if user and user not in ("root", "0"):
        block.append("USER root")
    block += [f"RUN {line}" for line in run_lines]
    if user and user not in ("root", "0"):
        block.append(f"USER {user}")
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(block) + "\n")


def _paragraphs(task_dir: Path) -> list[str]:
    text = (task_dir / "instruction.md").read_text().strip("\n")
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def insertion_points(task_dir: Path) -> list[int]:
    """Paragraph boundaries of instruction.md where a sentence can go without landing inside a code fence."""
    paras = _paragraphs(task_dir)
    return [i for i in range(len(paras) + 1) if sum(p.count("```") for p in paras[:i]) % 2 == 0]


def insert_paragraph(task_dir: Path, text: str, position: int) -> int:
    """Insert `text` as its own paragraph so it becomes paragraph number `position` (0 = first).

    A sentence that always sits last is a positional cue for the family it belongs to, so operators draw the
    position at random from `insertion_points`. A position inside a code fence moves to the next boundary. Returns the
    index used.
    """
    paras = _paragraphs(task_dir)
    allowed = insertion_points(task_dir)
    position = next((i for i in allowed if i >= position), allowed[-1])
    paras.insert(position, text.strip())
    (task_dir / "instruction.md").write_text("\n\n".join(p.rstrip() for p in paras) + "\n")
    return position


def normalise_instruction(task_dir: Path) -> None:
    """Give instruction.md the one format `insert_paragraph` writes: blank-line separated paragraphs, no trailing space
    after a paragraph, exactly one final newline.

    Every emitted variant, the plain feasible copy included, goes through this, so a perturbed instruction is not
    marked by its formatting (the leak verifier saw perturbed rows end in a newline and lose trailing spaces while
    plain ones kept both, which separates perturbed from plain rows).
    """
    paras = _paragraphs(task_dir)
    (task_dir / "instruction.md").write_text("\n\n".join(p.rstrip() for p in paras) + "\n")


def prepend_solve(task_dir: Path, lines: list[str]) -> None:
    """Insert shell lines at the top of the oracle (after the shebang), so they run before any `exit` or `set -e` abort.

    Used by near-miss controls whose obstacle is surmountable: the control's oracle does the extra step a competent
    agent would do (restore the input, extend PATH, delete the scratch directory), so the control validates as feasible.
    """
    path = task_dir / "solution" / "solve.sh"
    body = path.read_text().splitlines(keepends=True)
    head = body[:1] if body and body[0].startswith("#!") else []
    if head and not head[0].endswith("\n"):
        head = [head[0] + "\n"]
    block = [ln.rstrip("\n") + "\n" for ln in lines]  # no marker comment: nothing in a task may name its perturbation
    path.write_text("".join(head + block + body[len(head) :]))


def append_test(task_dir: Path, name: str, body_lines: list[str]) -> Path | None:
    """Append a test function to the last `tests/test_*.py` and return that file (None when the task has none).

    Appending to an existing file, instead of adding a new one, means the hermetic pytest verifier and the task's own
    `tests/test.sh` both run it whichever way `test.sh` names its test files.
    """
    files = sorted((task_dir / "tests").glob("test_*.py"))
    if not files:
        return None
    path = files[-1]
    func = [f"def {name}():", *[f"    {ln}" for ln in body_lines]]
    path.write_text(path.read_text().rstrip("\n") + "\n\n\n" + "\n".join(func) + "\n")
    return path


def slug(s: str) -> str:
    """Filesystem / Harbor-name safe id: `seta-env/SETA_Synth/ask_ubuntu__0` -> `seta-env__SETA_Synth__ask_ubuntu__0`."""
    out = re.sub(r"[^A-Za-z0-9._-]+", "__", s.strip("/")).strip("._-")
    return out or "task"
