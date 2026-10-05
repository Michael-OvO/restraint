"""Environment-text cues: can a policy tell an impossible variant from its control by the words and strings of the files
the operator planted, without reading what they mean?

The environment-only families (`high_risk`, `revoke_permission`, `remove_input`) share a byte-identical instruction, so
the path-level cue gate (`cues.py`) is pair-identical for them by construction. What differs is the environment: a
README sentence, the owner and mode of a lock, a file moved aside. That difference IS the evidence the agent has to find
by looking, so a classifier over it scores high by design. What must not separate the two sides is everything else the
operator writes: how long or how punctuated the README is, which other files and names come with it. A control whose
README is always a short fragment ("a transcoding cache.") and an impossible twin whose README is a long clause ("the
only copy of last night's production database backups.") hand the policy a length rule that holds in train and test.

Three channels are read off the emitted variant (`environment_text`), from the build steps the operator appended:

* `readme`: content of the planted README-like files (the carrier of `high_risk`);
* `files`: the planted files' names and the content of the other ones (should be identical across a pair);
* `lock`: the owner, group and mode words of the appended `chown` and `chmod` steps (the carrier of `revoke_permission`).

`env_text_classifiers` scores them with grouped cross-validation (folds are the donor groups of `cues.donor_group`, so a
source, its twin and its control sit in one fold). The gated entries (`cues.ENFORCED_ENV_TEXT`) are the README's style,
taken alone and as a classifier, and the words of every channel that is not the family's carrier (`CARRIER_CHANNELS`); the
carrier entries (`env:carrier_*`) are reported and never enforced, because what the README, the lock or the moved file says
is the label.
"""

from __future__ import annotations

import math
import posixpath
import re
from collections.abc import Iterable, Mapping
from typing import Any

from honest_agents.tasks.analysis.cues import (
    DONOR_GROUPS,
    _logistic_scores,
    _nb_scores,
    auc_score,
    auc_se,
    best_rule_accuracy,
    donor_group,
)
from honest_agents.tasks.analysis.facts import TaskFacts

_WORD = re.compile(r"[A-Za-z][A-Za-z']+|\d+")
_LOCK_WORDS = re.compile(r"\b[A-Za-z_][\w-]*:[A-Za-z_][\w-]*\b|\b[0-7]{3,4}\b|\b[ugoa]*[-+=][rwxXst]+\b")
_CHOWN_CHMOD = re.compile(r"\b(?:chown|chmod)\b[^;&|\n]*")

# The channel whose meaning IS each family's label: a classifier on its words scores high by design and is only reported.
# high_risk: what the README says the directory holds; revoke_permission: who owns the lock and its mode; remove_input: the
# input moved aside under a new name. A family not listed here has no carrier, so every channel is enforced.
CARRIER_CHANNELS: dict[str, tuple[str, ...]] = {
    "high_risk": ("readme",),
    "revoke_permission": ("lock",),
    "remove_input": ("files",),
}


def appended_text(variant_dockerfile: str | None, source_dockerfile: str | None) -> str:
    """The build steps an operator added: operators only append, so this is the variant's Dockerfile past the source's."""
    new, old = variant_dockerfile or "", source_dockerfile or ""
    return new[len(old) :] if old and new.startswith(old.rstrip()) else new


def environment_text(facts: TaskFacts, appended: str) -> dict[str, str]:
    """`{"readme", "files", "lock"}` of an emitted variant (empty strings for what the operator did not plant).

    `facts` are the variant's own; `appended` is `appended_text` of its Dockerfile. A file counts as planted when its path
    is spelled in the appended steps."""
    readme: list[str] = []
    files: list[str] = []
    for path, f in sorted(facts.image_files.items()):
        if f.content is None or path not in appended:
            continue
        if posixpath.basename(path).lower().startswith("readme"):
            readme.append(f.content)
        else:
            files += [posixpath.basename(path), f.content]
    lock = " ".join(w for step in _CHOWN_CHMOD.findall(appended) for w in _LOCK_WORDS.findall(step))
    return {"readme": "\n".join(readme), "files": "\n".join(files), "lock": lock}


def _words(text: str) -> list[str]:
    return sorted({w.lower() for w in _WORD.findall(text)})


def _style(readme: str) -> list[float]:
    """Length and punctuation of the README: shape features that carry no meaning."""
    words = _WORD.findall(readme)
    return [
        float(len(words)),
        float(len(readme.strip())),
        float(readme.count(",")),
        float(sum(c.isdigit() for c in readme)),
        float("'" in readme),
        float(sum(w[0].isupper() for w in words)),
    ]


def _entry(scores: list[float], y: list[bool], groups: list[str]) -> dict[str, Any]:
    pos = [v for v, yy in zip(scores, y, strict=True) if yy]
    neg = [v for v, yy in zip(scores, y, strict=True) if not yy]
    auc = auc_score(pos, neg)
    auc = 0.5 if auc is None else auc  # not `or 0.5`: an AUC of 0.0 is a perfect (inverted) separator, not chance
    return {
        "auc": auc,
        "auc_se": auc_se(auc, len(pos), len(neg)),
        "accuracy": best_rule_accuracy(pos, neg) or 0.5,
        "n_infeasible": len(pos),
        "n_control": len(neg),
        "n_groups": len(set(groups)),
    }


def env_text_classifiers(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per family: held-out AUC of what the planted environment text alone gives away (`{family: {name: entry}}`).

    Rows carry `env_text` (`environment_text`), `infeasible`, `group` and `family`. An entry has the shape of a
    `cues.cue_classifiers` entry (`auc`, `auc_se`, `accuracy`, `n_infeasible`, `n_control`, `n_groups`). Only channels
    some row of the family actually has appear, so an empty channel never reads as a clean one. The words of a family's
    carrier channel (`CARRIER_CHANNELS`) are reported as `env:carrier_<channel>` and never enforced; every other
    channel's words are `env:<channel>` and enforced, and the README's style (`env:style`, `env:readme_*`) always is.
    """
    import numpy as np

    by_fam: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if r.get("env_text") is not None and r.get("group") is not None:
            by_fam.setdefault(str(r["family"]), []).append(r)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for fam, rs in sorted(by_fam.items()):
        y = [bool(r["infeasible"]) for r in rs]
        groups = [str(r["group"]) for r in rs]
        folds = [donor_group(g) for g in groups]
        if sum(y) < DONOR_GROUPS or len(y) - sum(y) < DONOR_GROUPS or len(set(folds)) < DONOR_GROUPS:
            continue
        text = [{k: str(v) for k, v in r["env_text"].items()} for r in rs]
        out[fam] = {}
        if any(t.get("readme") for t in text):
            style = [_style(t["readme"]) for t in text]
            for j, name in ((0, "env:readme_words"), (1, "env:readme_chars"), (2, "env:readme_commas")):
                out[fam][name] = _entry([s[j] for s in style], y, groups)
            out[fam]["env:style"] = _entry(_cv(style, y, folds, "logistic", np, groups), y, groups)
        for channel in ("readme", "files", "lock"):
            if any(t.get(channel) for t in text):
                name = f"env:carrier_{channel}" if channel in CARRIER_CHANNELS.get(fam, ()) else f"env:{channel}"
                out[fam][name] = _entry(_cv([_words(t[channel]) for t in text], y, folds, "nb", np, groups), y, groups)
    return out


def _cv(feats: list[Any], y: list[bool], folds: list[int], kind: str, np: Any, groups: list[str]) -> list[float]:
    """Pooled out-of-fold scores of a naive Bayes over word sets (`nb`) or a logistic regression over numbers."""
    scores = [0.0] * len(y)
    for k in range(DONOR_GROUPS):
        tr = [i for i, f in enumerate(folds) if f != k]
        te = [i for i, f in enumerate(folds) if f == k]
        if not te or len({y[i] for i in tr}) < 2:
            continue
        if kind == "nb":
            fold = _nb_scores([(feats[j], y[j], groups[j]) for j in tr], [feats[i] for i in te])
        else:
            fold = [
                float(v)
                for v in _logistic_scores(
                    np.array([feats[j] for j in tr]),
                    np.array([float(y[j]) for j in tr]),
                    np.array([feats[i] for i in te]),
                )
            ]
        for i, v in zip(te, fold, strict=True):
            scores[i] = v if math.isfinite(v) else 0.0
    return scores
