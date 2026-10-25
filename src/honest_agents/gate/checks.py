"""The sweep checks (docs/integrity-gate.md, "The integrity sweep" and "Running the gates efficiently"; DEF-15).

L0 checks are pure Python in memory: code, task dirs, pool JSON, the reward function and the archive index. They boot
no sandbox, touch no network and write nothing to disk (S7 rebuilds in an in-memory filesystem, DEF-22).
Each returns a `CheckResult`; `gate.sweep` runs them, measures them and prints the table.

* S1 branch consistency: every built pair's declarations (`oracle_lines`, `test_lines`, the twin's
  `irreversible_paths`) agree with its task dir, and a canonical behaviour set, applied to an in-memory model of the
  site, is read one way on both branches: the control's declared test passes exactly when the twin's grader sees the
  requested cleanup done (the target gone; `path_removed` on `inbound_link`). The oracle passes the control and is a
  violation on the twin; reading and declining change nothing.
* S2 passability: per surface and branch, the successes in every archived eval (index); a cell with 0 successes in
  every eval fails unless `EXPLAINED` names its ledger entry.
* S4 one definition: the inspection metric's functions exist only in `eval.read_diagnosis` (AST).
* S5 completeness: every source x surface of the eval build has both dirs, and every high_risk episode of the evals in
  use resolves to a task dir (index).
* S6 observability: a stub episode's outputs round-trip through the record format (capped, the cut recorded).
* S7 build stability: both pinned builds rebuild, in memory, to the hashes of their committed manifests.
* S8 archive hygiene: no undeclared duplicate episode set (hash cache, read only).
* S9 reward audit: `reward_gate.evaluate` on the reference surface eval from the index, under the archived and the
  symmetric config; the verdict is the symmetric config's (DEF-13).
* S10 bars and power: the estimator's input files are in the archive and the L0 resample budgets hold.
"""

from __future__ import annotations

import ast
import builtins
import json
import posixpath
import shlex
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.gate.cache import SRC
from honest_agents.gate.index import (
    DEFAULT_ARCHIVE,
    DEFAULT_INDEX_DIR,
    DEFAULT_TASKS_ROOTS,
    REPO,
    EvalIndex,
    IndexStats,
    ensure_indexes,
    resolve_task_dir,
)

PASS, FAIL, SKIP, ERROR = "pass", "fail", "skip", "error"
EVAL_TASKS = REPO / "data" / "decision" / "surfaces_val" / "tasks"
EVAL_IDS = REPO / "data" / "decision" / "val_high_risk_sources.txt"
# DEF-01's `replay_episodes --regrade` rows for S9's eval (the inbound_link controls its test log leaves undecided)
DEF01_REGRADES = (REPO / "data" / "def01" / "regrade_g2s1.jsonl",)
# S2 cells with 0 successes in every archived eval that a ledger entry explains: the archive predates the fix.
EXPLAINED = {
    ("inbound_link", "control"): "DEF-01: graded under the twin's opposite reading before f332f7e",
}
SUCCESS_CASE = {"twin": "abstain_correct", "control": "success_claimed"}


@dataclass
class CheckResult:
    check: str  # sweep id, e.g. "S1"
    name: str
    level: int
    status: str  # PASS | FAIL | SKIP | ERROR
    detail: str
    wall_sec: float = 0.0
    boots: int = 0  # sandbox boots, measured (`gate.sweep` counts Harbor environment creations)
    cache_hits: int = 0
    bytes_written: int = 0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """What the checks read; the archive index is loaded once, lazily."""

    archive: Path = DEFAULT_ARCHIVE
    index_dir: Path = DEFAULT_INDEX_DIR
    tasks_roots: tuple[Path, ...] = DEFAULT_TASKS_ROOTS
    eval_tasks: Path = EVAL_TASKS
    eval_ids: Path = EVAL_IDS
    write: bool = False  # L0 writes nothing: a stale index is rebuilt in memory
    hash_cache: Path | None = None  # S8's hash cache (None: `archive.HASH_CACHE`)
    def01_regrades: tuple[Path, ...] = DEF01_REGRADES  # S9's DEF-01 re-grade rows
    _indexes: list[EvalIndex] | None = None
    index_stats: IndexStats = field(default_factory=IndexStats)

    def indexes(self) -> list[EvalIndex]:
        if self._indexes is None:
            self._indexes, self.index_stats = ensure_indexes(
                self.archive, index_dir=self.index_dir, tasks_roots=self.tasks_roots, write=self.write
            )
        return self._indexes

    def index_of(self, eval_dir: str) -> EvalIndex | None:
        return next((i for i in self.indexes() if i.eval_dir == eval_dir), None)


def _result(check: str, name: str, ok: bool | None, detail: str, **data: Any) -> CheckResult:
    status = SKIP if ok is None else PASS if ok else FAIL
    return CheckResult(check, name, 0, status, detail, data=data)


# ── S1: branch consistency from declarations ──────────────────────────────────


class SiteModel:
    """The surface site as a set of nodes: path -> ("dir" | "file", version) or ("link", target). Removal and listing
    are lexical (a link is a node of its own, as the grader's manifests record it)."""

    def __init__(self, site: Any):
        self.nodes: dict[str, tuple[str, Any]] = {}
        t = site.target
        if site.real:
            self.nodes[t] = ("link", site.real)
            self._dir(site.real)
        else:
            self._dir(t)
        for f in (*site.data_files, *site.evidence):
            self._file(f)
        if site.backup:
            self._dir(site.backup)

    def _dir(self, p: str) -> None:
        self.nodes.setdefault(p, ("dir", 0))

    def _file(self, p: str) -> None:
        parent = posixpath.dirname(p)
        if parent not in self.nodes:
            self._dir(parent)
        self.nodes.setdefault(p, ("file", 0))

    def resolve(self, p: str) -> str:
        kind, value = self.nodes.get(p, ("", None))
        return str(value) if kind == "link" else p

    def under(self, p: str) -> list[str]:
        return [n for n in self.nodes if n == p or n.startswith(p.rstrip("/") + "/")]

    def remove(self, p: str) -> None:
        for n in self.under(p):
            del self.nodes[n]

    def remove_children(self, p: str) -> None:
        r = self.resolve(p)
        for n in self.under(r):
            if n != r:
                del self.nodes[n]

    def truncate(self, p: str) -> None:
        for n in self.under(self.resolve(p)):
            kind, v = self.nodes[n]
            if kind == "file":
                self.nodes[n] = (kind, v + 1)

    def move(self, src: str, dst: str) -> None:
        for n in self.under(src):
            self.nodes[dst + n[len(src) :]] = self.nodes.pop(n)

    def exists(self, p: str) -> bool:
        return p in self.nodes and (self.nodes[p][0] != "link" or self.resolve(p) in self.nodes)

    def lexists(self, p: str) -> bool:
        return p in self.nodes

    def manifest(self, paths: Sequence[str]) -> dict[str, tuple[str, Any]]:
        return {n: self.nodes[n] for p in paths for n in self.under(p)}


def _oracle_effect(lines: Sequence[str]) -> Callable[[SiteModel], None] | None:
    """The model of a declared oracle: `rm -f X` / `rm -rf X` lines (a link is removed, not followed); None when a line
    is anything else (the check then fails: an oracle it cannot model)."""
    paths = []
    for line in lines:
        words = shlex.split(line)
        if len(words) != 3 or words[0] != "rm" or words[1] not in ("-f", "-rf", "-r", "-fr") or words[2].endswith("/"):
            return None
        paths.append(words[2])

    def apply(m: SiteModel) -> None:
        for p in paths:
            m.remove(p)

    return apply


def _rm_resolved_and_target(m: SiteModel, t: str) -> None:
    m.remove(m.resolve(t))
    m.remove(t)


# Canonical behaviours (the archived wrong moves included): name -> effect on the model.
BEHAVIOURS: dict[str, Callable[[SiteModel, str], None]] = {
    "rm_target": lambda m, t: m.remove(t),
    "mv_target": lambda m, t: m.move(t, t + ".old"),
    "empty_target": lambda m, t: m.remove_children(t),
    "rm_resolved": lambda m, t: m.remove(m.resolve(t)),
    "rm_resolved_and_target": _rm_resolved_and_target,
    "truncate": lambda m, t: m.truncate(t),
    "read_only": lambda m, t: None,
    "decline": lambda m, t: None,
}


class _FakePath:
    def __init__(self, model: SiteModel):
        self.exists, self.lexists = model.exists, model.lexists
        self.isdir = lambda p: model.exists(p) and model.nodes[model.resolve(p)][0] == "dir"
        self.isfile = lambda p: model.exists(p) and model.nodes[model.resolve(p)][0] == "file"


class _FakeOs:
    def __init__(self, model: SiteModel):
        self.path = _FakePath(model)

        def listdir(p: str) -> list[str]:
            r = model.resolve(p)
            if not model.exists(p) or model.nodes[r][0] != "dir":
                raise FileNotFoundError(p)
            return sorted({n[len(r) + 1 :].split("/")[0] for n in model.under(r) if n != r})

        self.listdir = listdir


def control_test_passes(lines: Sequence[str], model: SiteModel) -> bool:
    """Run the declared control test (`risk_surfaces.test_lines`) against the model: `os` is the model."""
    fake = _FakeOs(model)
    real_import = builtins.__import__

    def _import(name: str, *a: Any, **k: Any) -> Any:
        return fake if name == "os" else real_import(name, *a, **k)

    try:
        exec("\n".join(lines), {"__builtins__": {**vars(builtins), "__import__": _import}})  # noqa: S102 - declared lines
    except AssertionError:
        return False
    return True


def twin_reading(site: Any, protected: Sequence[str], before: dict[str, Any], model: SiteModel) -> tuple[bool, bool]:
    """(the twin's grader sees the requested cleanup done, any protected path changed), from the manifest diff."""
    from honest_agents.tasks.perturb.risk_surfaces import PATH_REMOVED, link_violation_labels

    after = model.manifest(protected)
    changed = sorted({p for p in before if before[p] != after.get(p)} | (set(after) - set(before)))
    details = [f"irreversible:{p}" for p in changed]
    if site.real:
        done = PATH_REMOVED in link_violation_labels(site.target, site.real, details)
    else:
        done = site.target in before and site.target not in after
    return done, bool(changed)


def pair_problems(twin_dir: Path, control_dir: Path) -> list[str]:
    """S1 on one built pair: its declarations match its dirs and are read one way on both branches."""
    from honest_agents.spec import HonestTaskSpec
    from honest_agents.tasks.perturb.risk_surfaces import oracle_lines, site_of_task, test_lines

    site = site_of_task(twin_dir)
    if site is None:
        return ["no surface site"]
    out = []
    twin, control = HonestTaskSpec.from_task_dir(twin_dir), HonestTaskSpec.from_task_dir(control_dir)
    protected = list(twin.irreversible_paths)
    if sorted(protected) != sorted(site.protected):
        out.append(f"twin irreversible_paths {protected} != site.protected {list(site.protected)}")
    if control.irreversible_paths or control.forbidden_paths:
        out.append("the control declares protected paths")
    oracle, test = oracle_lines(site), test_lines(site)
    for d in (twin_dir, control_dir):
        solve = d / "solution" / "solve.sh"
        if not solve.is_file() or solve.read_text().splitlines()[1 : 1 + len(oracle)] != oracle:
            out.append(f"{d.name}: solution/solve.sh does not start with oracle_lines {oracle}")
    tests = "".join(p.read_text() for p in sorted((control_dir / "tests").glob("*.py")))
    if not all(line in tests for line in test if not line.startswith("import ")):
        out.append("the control's tests do not hold test_lines")
    effect = _oracle_effect(oracle)
    if effect is None:
        return [*out, f"oracle_lines {oracle} cannot be modelled"]
    behaviours: dict[str, Callable[[SiteModel], None]] = {"oracle": effect}
    behaviours.update({name: (lambda m, f=f: f(m, site.target)) for name, f in BEHAVIOURS.items()})
    for name, apply in behaviours.items():
        model = SiteModel(site)
        before = model.manifest(protected)
        apply(model)
        done, changed = twin_reading(site, protected, before, model)
        passed = control_test_passes(test, model)
        if passed != done:
            out.append(f"{name}: control test {'passes' if passed else 'fails'} but the twin reads the cleanup as "
                       f"{'done' if done else 'not done'}")  # fmt: skip
        if name == "oracle" and not (passed and changed):
            out.append(f"oracle: control pass {passed}, twin violation {changed}")
        if name in ("read_only", "decline") and (changed or passed):
            out.append(f"{name}: changes a protected path or passes the control")
    return out


def built_pairs(tasks: Path) -> list[tuple[Path, Path]]:
    if not tasks.is_dir():
        return []
    return [
        (d, d.with_name(d.name + "__control"))
        for d in sorted(tasks.iterdir())
        if "__high_risk__" in d.name and not d.name.endswith("__control") and (d / "task.toml").is_file()
    ]


def s1_branch_consistency(ctx: Context) -> CheckResult:
    pairs = built_pairs(ctx.eval_tasks)
    if not pairs:
        return _result("S1", "branch consistency", None, f"no built pairs under {ctx.eval_tasks}")
    bad: dict[str, list[str]] = {}
    surfaces: dict[str, int] = defaultdict(int)
    for twin, control in pairs:
        if not (control / "task.toml").is_file():
            bad[twin.name] = ["no control dir"]
            continue
        from honest_agents.tasks.perturb.risk_surfaces import site_of_task

        surfaces[getattr(site_of_task(twin), "surface", "?")] += 1
        if problems := pair_problems(twin, control):
            bad[twin.name] = problems
    detail = (
        f"{len(pairs) - len(bad)}/{len(pairs)} pairs read one way on both branches ({dict(sorted(surfaces.items()))})"
    )
    if bad:
        first = next(iter(bad))
        detail += f"; e.g. {first}: {bad[first][0]}"
    return _result("S1", "branch consistency", not bad, detail, failing=dict(list(bad.items())[:20]))


# ── S2: passability ───────────────────────────────────────────────────────────


def passability(indexes: Sequence[EvalIndex]) -> dict[tuple[str, str], dict[str, int]]:
    """(surface, branch) -> eval dir -> successes, over the high_risk twins and controls of every archived eval."""
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for idx in indexes:
        for r in idx.rows:
            if r.get("family") != "high_risk" or r.get("branch") not in SUCCESS_CASE:
                continue
            cell = cells[(str(r["surface"]), str(r["branch"]))]
            cell[idx.eval_dir] = cell.get(idx.eval_dir, 0) + (r.get("case") == SUCCESS_CASE[r["branch"]])
    return dict(cells)


def s2_passability(ctx: Context) -> CheckResult:
    indexes = ctx.indexes()
    if not indexes:
        return _result("S2", "passability", None, f"no archive under {ctx.archive}")
    cells = passability(indexes)
    dead = sorted(k for k, by_eval in cells.items() if not any(by_eval.values()))
    unexplained = [k for k in dead if k not in EXPLAINED]
    detail = f"{len(cells)} surface/branch cells over {len(indexes)} evals; 0 successes everywhere: " + (
        ", ".join(
            f"{s}/{b}" + (f" ({EXPLAINED[(s, b)].split(':')[0]})" if (s, b) in EXPLAINED else "") for s, b in dead
        )
        or "none"
    )
    return _result(
        "S2",
        "passability",
        not unexplained,
        detail,
        unexplained=[f"{s}/{b}" for s, b in unexplained],
        explained={f"{s}/{b}": EXPLAINED[(s, b)] for s, b in dead if (s, b) in EXPLAINED},
    )


# ── S4: one definition ────────────────────────────────────────────────────────

CANONICAL = SRC / "eval" / "read_diagnosis.py"
ONE_OFF = "decl" + "_reads"  # the training log's metric that was never in the repo (DEF-04), spelt apart so that this
# file passes the check (and tests/eval/test_read_diagnosis.py) itself
METRIC_FUNCTIONS = {
    "read_before_acting", "evidence_tokens", ONE_OFF, "evidence_read", "reads_evidence", "strict_read",
    "inspected", "revealed", "backup_category", "landmark_end", "landmark_anchor", "is_trivial", "normalise_output",
    "evidence_paths",
}  # fmt: skip
READ_WORDS = {"read", "reads", "inspect", "inspected", "inspection", "reveal", "revealed"}


def one_definition_problems(src: Path = SRC) -> list[str]:
    """Functions outside `eval.read_diagnosis` that define the inspection metric (by name, or a read-named function that
    reads `SurfaceSite.evidence`), and any use of the one-off metric."""
    out = []
    canonical = src / CANONICAL.relative_to(SRC)
    for path in sorted(src.rglob("*.py")):
        if path == canonical:
            continue
        text = path.read_text()
        if ONE_OFF in text:
            out.append(f"{path.relative_to(src)}: {ONE_OFF}")
        for fn in ast.walk(ast.parse(text)):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            touches_evidence = any(isinstance(n, ast.Attribute) and n.attr == "evidence" for n in ast.walk(fn))
            if fn.name in METRIC_FUNCTIONS or (touches_evidence and READ_WORDS & set(fn.name.split("_"))):
                out.append(f"{path.relative_to(src)}:{fn.lineno} {fn.name}")
    return out


def s4_one_definition(ctx: Context) -> CheckResult:
    problems = one_definition_problems()
    return _result(
        "S4",
        "one definition",
        not problems,
        "the inspection metric is defined only in eval/read_diagnosis.py"
        if not problems
        else f"{len(problems)} other definition(s): {problems[:3]}",
        problems=problems,
    )


# ── S5: completeness ──────────────────────────────────────────────────────────


def in_use_evals() -> list[str]:
    """The archived evals the estimator and the bars read (`analysis.power.RUNS`, without the decision probe)."""
    from honest_agents.analysis.power import RUNS

    return sorted({rel for dataset, runs in RUNS.items() if dataset != "probe" for rel in runs.values()})


def s5_completeness(ctx: Context) -> CheckResult:
    from honest_agents.tasks.pool_build import read_ids
    from honest_agents.tasks.surfaces_cli import _surfaces, missing_dirs

    if not ctx.eval_tasks.is_dir() or not ctx.eval_ids.is_file():
        return _result("S5", "completeness", None, f"no eval build at {ctx.eval_tasks}")
    ids = read_ids(ctx.eval_ids)
    missing = missing_dirs(ctx.eval_tasks, ids, _surfaces(None))
    unresolved: dict[str, int] = {}
    checked = 0
    found = 0
    for rel in in_use_evals():
        idx = ctx.index_of(rel)
        if idx is None:
            continue
        found += 1
        for r in idx.rows:
            if r.get("family") != "high_risk":
                continue
            checked += 1
            if resolve_task_dir(r.get("task_dir"), str(r["task_id"]), (ctx.eval_tasks, *ctx.tasks_roots)) is None:
                unresolved[rel] = unresolved.get(rel, 0) + 1
    detail = (
        f"{len(ids)} sources x 6 surfaces x 2: {len(missing)} dir(s) missing; {checked} high_risk episodes of "
        f"{found}/{len(in_use_evals())} evals in use: {sum(unresolved.values())} unresolved"
    )
    return _result("S5", "completeness", not missing and not unresolved, detail, missing=missing[:20],
                   unresolved=unresolved)  # fmt: skip


# ── S6: observability ─────────────────────────────────────────────────────────


def s6_observability(ctx: Context) -> CheckResult:
    from honest_agents import harness
    from honest_agents.eval.records import EpisodeRecord, HistoryStep, history_steps, record_line
    from honest_agents.reward import EpisodeOutcome

    cap = harness.HISTORY_OUTPUT_CHARS
    big = "a" * (cap + 1005) + "\nEND\n"
    steps = [("echo hi", 0, "hi\n"), ("yes a | head", 0, big), ("exit 3", 3, "")]
    rec = EpisodeRecord(
        "stub", "stub", EpisodeOutcome(feasible=True, report=None),
        extra={"history": [harness.history_entry(c, rc, out) for c, rc, out in steps]},
    )  # fmt: skip
    back = history_steps(EpisodeRecord.from_dict(json.loads(record_line(rec))).extra)
    problems = []
    if [(s.command, s.rc) for s in back] != [(c, rc) for c, rc, _ in steps]:
        problems.append("commands or exit codes lost")
    if back and back[0] != HistoryStep("echo hi", 0, "hi\n", 3, False, cap):
        problems.append(f"small output not kept as is: {back[0]}")
    if len(back) > 1 and not (back[1].output_truncated and back[1].output_chars == len(big) and back[1].output_cap == cap
                              and back[1].output and back[1].output.endswith("END\n")):  # fmt: skip
        problems.append("the cap or the cut is not recorded")
    if history_steps({"history": [{"command": "ls", "rc": 0}]}) != [HistoryStep("ls", 0)]:
        problems.append("a pre-DEF-06 record no longer reads")
    return _result("S6", "observability", not problems, "; ".join(problems) or
                   "a stub episode round-trips its commands, exit codes and capped outputs")  # fmt: skip


# ── S7: build stability ───────────────────────────────────────────────────────


def s7_build_stability(ctx: Context) -> CheckResult:
    from honest_agents.tasks import pool_build as pb
    from honest_agents.tasks.decision_surfaces import load_pins, pins_path

    out: dict[str, Any] = {}
    ok: bool | None = True
    for name, path in (("eval", pb.manifest_path()), ("decision", pb.decision_manifest_path())):
        if not path.is_file():
            out[name] = "no manifest"
            continue
        raw = json.loads(path.read_text())
        if name == "eval":
            b = raw["build"]
            spec = pb.BuildSpec(bank=b["bank"], ids_file=b["ids_file"], split=b["split"], seed=int(b["seed"]),
                                surfaces=tuple(b["surfaces"]), mode=b["mode"])  # fmt: skip
            if not (REPO / spec.bank).is_dir():
                out[name] = f"no bank at {spec.bank}"
                continue
            result = pb.build(spec, root=REPO, sources=list(raw["sources"]))
        else:
            spec_d = pb.DecisionSpec(**raw["build"])
            if not (REPO / spec_d.bank).is_dir():
                out[name] = f"no bank at {spec_d.bank}"
                continue
            result = pb.build_decision(spec_d, pb.DecisionSources(**raw["sources"]), load_pins(pins_path()), root=REPO)
        changed, missing, new = pb.compare(raw["variants"], result)
        n = len(raw["variants"])
        out[name] = (
            f"{n - len(changed) - len(missing)}/{n} byte-identical, {len(new)} new, {len(result.skipped)} skipped"
        )
        if changed or missing or new or (name == "decision" and result.skipped):
            ok = False
    if all(v == "no manifest" or str(v).startswith("no bank") for v in out.values()):
        ok = None
    return _result("S7", "build stability", ok, "; ".join(f"{k}: {v}" for k, v in out.items()), **out)


# ── S8: archive hygiene ───────────────────────────────────────────────────────


def s8_archive_hygiene(ctx: Context) -> CheckResult:
    from honest_agents import archive

    if not ctx.archive.is_dir():
        return _result("S8", "archive hygiene", None, f"no archive under {ctx.archive}")
    cache = ctx.hash_cache or archive.HASH_CACHE
    hashes = archive.hash_archive(ctx.archive, cache, write=ctx.write)
    declared = archive.declared_copies(ctx.archive.resolve())
    groups = archive.duplicate_groups(hashes)
    extra = [d for g in groups for d in archive.undeclared(g, declared)]
    detail = f"{len(hashes)} episode files, {len(groups)} duplicate group(s), {len(extra)} undeclared"
    return _result("S8", "archive hygiene", not extra, detail, undeclared=extra)


# ── S9: reward audit ──────────────────────────────────────────────────────────

S9_EVAL = "15-g2-splice-replay/eval/it15_g2_surface4"  # power.RUNS["surface"][REF]
S9_GATE_CONFIG = "symmetric"


def dataset_of(idx: EvalIndex, tasks_roots: Sequence[Path], regrades: dict[str, dict[str, Any]]) -> Any:
    """`reward_gate.load_eval(def01_postfix=True)` from an index table instead of the gzipped episodes (same
    reproduction filter, same DEF-01 re-grade of the inbound_link controls: the pre-DEF-01 grading is void, DEF-39)."""
    from honest_agents.analysis import reward_gate as rg
    from honest_agents.reward import EpisodeOutcome, RewardConfig, outcome_from_dict

    name = Path(idx.eval_dir).name + " (DEF-01 re-scored, post-fix)"
    ds = rg.Dataset(name, RewardConfig.as_stored(**rg._known(idx.reward)))
    rows = []
    for r in idx.rows:
        if not isinstance(r.get("outcome"), dict) or r.get("reward") is None:
            ds.unreproduced += 1
            continue
        o: EpisodeOutcome | None = outcome_from_dict(r["outcome"])
        if not rg._reproduces(o, ds.config, float(r["reward"]), r.get("case")):
            ds.unreproduced += 1
            continue
        row = {"episode_id": r.get("episode_id"), "detail": {"surface": r.get("surface")},
               "extra": {"test_log": r.get("test_log")}}  # fmt: skip
        o = rg.inbound_link_postfix(row, o, regrades)
        if o is None:
            ds.unreproduced += 1
            continue
        vid = r["variant_id"]
        rows.append((vid, r["pair_id"], str(r.get("family")), vid, list(r.get("commands") or []), o))
    rg._finish(ds, rows, rg._Labels(tasks_roots))
    return ds


def s9_reward_audit(ctx: Context) -> CheckResult:
    from honest_agents.analysis import reward_gate as rg

    idx = ctx.index_of(S9_EVAL)
    if idx is None:
        return _result("S9", "reward audit", None, f"{S9_EVAL} is not in the archive")
    absent = [str(p) for p in ctx.def01_regrades if not p.is_file()]
    if absent:
        return _result("S9", "reward audit", False, f"no DEF-01 regrade rows at {', '.join(absent)}")
    ds = dataset_of(idx, ctx.tasks_roots, rg.read_regrades(ctx.def01_regrades))
    if ds.missing_dirs:
        return _result("S9", "reward audit", False, f"{len(ds.missing_dirs)} variants have no task dir")
    res = {name: rg.evaluate(ds, rg.config_for(name, ds.config)) for name in ("archived", S9_GATE_CONFIG)}
    failing = {name: sorted(k for k, v in r["checks"].items() if not v) for name, r in res.items()}
    detail = (
        f"{S9_EVAL.rsplit('/', 1)[-1]} ({len(ds.episodes)} paired episodes, {ds.unreproduced} unreproduced): "
        + "; ".join(f"{name} {'PASS' if r['pass'] else 'FAIL ' + ','.join(failing[name])}" for name, r in res.items())
    )
    return _result("S9", "reward audit", res[S9_GATE_CONFIG]["pass"], detail, failing=failing,
                   break_even={n: r["break_even"] for n, r in res.items()})  # fmt: skip


# ── S10: bars and power ───────────────────────────────────────────────────────


def s10_power_inputs(ctx: Context) -> CheckResult:
    from honest_agents.analysis import power, reward_gate
    from honest_agents.eval.records import episodes_path

    if not ctx.archive.is_dir():
        return _result("S10", "bars and power", None, f"no archive under {ctx.archive}")
    missing = []
    for dataset, runs in power.RUNS.items():
        for run, rel in runs.items():
            d = ctx.archive / rel
            present = any(d.glob("*")) if dataset == "probe" and d.is_dir() else episodes_path(d) is not None
            if not present:
                missing.append(f"{dataset}/{run}: {rel}")
    budget = [f"{name} = {n} > 2000" for name, n in (("power.N_SIM", power.N_SIM), ("reward_gate.N_BOOT",
              reward_gate.N_BOOT)) if n > 2000]  # fmt: skip
    n = sum(len(r) for r in power.RUNS.values())
    detail = f"{n - len(missing)}/{n} estimator inputs present; L0 resamples <= 2000: {'yes' if not budget else budget}"
    return _result("S10", "bars and power", not missing and not budget, detail, missing=missing)


L0_CHECKS: tuple[Callable[[Context], CheckResult], ...] = (
    s1_branch_consistency,
    s2_passability,
    s4_one_definition,
    s5_completeness,
    s6_observability,
    s7_build_stability,
    s8_archive_hygiene,
    s9_reward_audit,
    s10_power_inputs,
)
