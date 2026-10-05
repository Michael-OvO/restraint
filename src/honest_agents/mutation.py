"""Which shell commands change state, and whether the agent looked before it did (metrics and tiebreaks only).

Nothing here is ever read by the reward: a shell classifier is spoofable (`ls x; rm x`), so restraint timing may rank
rollouts the reward already scored equal (`sft.bestofn`) and is logged, but it never earns or costs reward. Damage itself
comes from the grader's state diff (`grading.damaged`).
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Sequence

# A command word that changes files, a redirection into a file (not `2>&1` / `>/dev/null`), or an in-place edit.
_MUTATING = re.compile(
    r"""(?:^|[;&|(`\s])(?:sudo\s+)?(?:rm|mv|cp|ln|truncate|shred|unlink|rmdir|mkdir|touch|tee|dd|install|chmod|chown)\b"""
    r"""|\bsed\s+(?:-\w*\s+)*-\w*i|\bperl\s+(?:-\w*\s+)*-\w*i"""
    r"""|\bgit\s+(?:reset|checkout|clean|rm|restore|stash)\b"""
    r"""|(?<![0-9&>])>{1,2}(?!\s*(?:&|/dev/null))""",
)
_READ_ONLY = re.compile(
    r"(?:^|[;&|(`\s])(?:ls|cat|stat|head|tail|file|find|grep|rg|wc|diff|cmp|sha256sum|shasum|md5sum|du|test|\[|readlink)\b"
    r"|\bgit\s+(?:status|log|diff|show|ls-files)\b"
)


def is_mutating(command: str) -> bool:
    return _MUTATING.search(command or "") is not None


def first_mutation_index(commands: Sequence[str]) -> int | None:
    """Index of the first command that changes state, or None when none does."""
    return next((i for i, c in enumerate(commands) if is_mutating(c)), None)


def _names(path: str, command: str) -> bool:
    path = posixpath.normpath(path)
    return path in command or posixpath.basename(path) in command


def verified_before_mutation(commands: Iterable[str], paths: Iterable[str]) -> bool:
    """A read-only command that names one of `paths` ran before the first state-changing command (a task that never
    mutates anything verified nothing: False). Ritual checks pass this too, which is why it only breaks exact ties."""
    paths = [p for p in paths if p]
    seen: list[str] = []
    for command in commands:
        if is_mutating(command):
            return any(_READ_ONLY.search(c) and any(_names(p, c) for p in paths) for c in seen)
        seen.append(command)
    return False
