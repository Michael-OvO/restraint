"""Existence-cue statistics: could a policy read the label off a row's path alone (AUC, cross-validated classifiers)?"""

from __future__ import annotations

import hashlib
import math
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from honest_agents.tasks.analysis.boot import exists_at_boot
from honest_agents.tasks.analysis.facts import TaskFacts
from honest_agents.tasks.analysis.script import extract_paths


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
    reading the task is a shortcut: the build writes this report to manifest.json and refuses a family outside
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
# a control borrows only from its own group (`perturb.conflicting_constraint.name_bank`), and the cross-validation holds out one whole group.
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
    """A family's forbidden/target object can be told apart from its control's by a cheap feature."""


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
