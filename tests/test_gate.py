"""DEF-15: the tiered, cached integrity gate (`python -m honest_agents.gate sweep`, docs/integrity-gate.md "Running the
gates efficiently"): L0 in memory with no sandbox boot, the content-addressed result cache behind `g1_surfaces`, the
archive index, and the sweep's measurements."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from honest_agents.gate import checks, index, sweep
from honest_agents.gate.cache import SRC, ResultCache
from honest_agents.tasks import g1_surfaces
from honest_agents.tasks.perturb import risk_surfaces
from honest_agents.tasks.perturb.risk_surfaces import SURFACES, build_varied
from tests.tasks.synthetic import rich_task

PAIR = "synthetic/rich"


@pytest.fixture
def rich(tmp_path) -> Path:
    return rich_task(tmp_path / "src" / "rich", verifier="pytest")


def _no_boots(monkeypatch) -> None:
    """Every way a check could reach a sandbox or Docker raises."""
    from harbor.environments.factory import EnvironmentFactory

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("an L0 check booted a sandbox")

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", staticmethod(boom))
    monkeypatch.setattr("honest_agents.harness._start_throttled", boom)
    monkeypatch.setattr("honest_agents.gate.cache.docker_image_digest", boom)
    real_popen = subprocess.Popen

    def popen(args: Any, *a: Any, **k: Any) -> Any:
        argv0 = str(args[0] if isinstance(args, list | tuple) else args).split()[0]
        if os.path.basename(argv0) == "docker":
            raise AssertionError(f"an L0 check ran docker: {args}")
        return real_popen(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)


# ── L0 on this checkout ───────────────────────────────────────────────────────


def test_the_l0_sweep_boots_nothing_writes_nothing_and_fits_its_budget(monkeypatch) -> None:
    """The closure test: every L0 check runs on this checkout's code, data and archive index with the boot and Docker
    patched to raise; 0 boots, nothing written (S7 rebuilds in memory, DEF-22), <= 60 s. Checks whose inputs are absent
    (a node has no archive, a worktree no `data/`) skip. The checks of closed defects must pass; S9 is reported, not
    asserted (DEF-13 is open; post-DEF-01, DEF-39, `symmetric` passes on G2 s1 and `archived` fails)."""
    _no_boots(monkeypatch)
    ctx = checks.Context(write=False)
    t0 = time.monotonic()
    results = sweep.sweep(0, ctx=ctx)
    wall = time.monotonic() - t0
    by = {r.check: r for r in results}
    assert sorted(by) == sorted(["S1", "S2", "S4", "S5", "S6", "S7", "S8", "S9", "S10"])
    assert not [r for r in results if r.status == checks.ERROR], [(r.check, r.detail) for r in results]
    assert sum(r.boots for r in results) == 0 and ctx.index_stats.bytes_written == 0
    assert all(r.bytes_written == 0 for r in results), [(r.check, r.bytes_written) for r in results]
    for c in ("S1", "S2", "S4", "S5", "S6", "S7", "S8", "S10"):
        assert by[c].status in (checks.PASS, checks.SKIP), (c, by[c].detail)
    assert wall <= sweep.BUDGET_SEC[0], wall


def test_boots_are_counted_and_an_l0_check_that_boots_is_caught(monkeypatch) -> None:
    """The meter counts Harbor environment creations, so a check that boots shows up in the table."""
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", staticmethod(lambda **k: object()))

    def booting() -> checks.CheckResult:
        EnvironmentFactory.create_environment_from_config(config=None)
        EnvironmentFactory.create_environment_from_config(config=None)
        return checks.CheckResult("SX", "x", 0, checks.PASS, "")

    r = sweep.run_check(booting, "SX", "x", 0)
    assert r.boots == 2
    assert EnvironmentFactory.create_environment_from_config(config=None) is not None  # restored, not counting


# ── S1 from declarations ──────────────────────────────────────────────────────


@pytest.mark.parametrize("surface", SURFACES)
def test_s1_reads_every_surface_one_way_on_both_branches(rich, tmp_path, surface) -> None:
    twin, control = build_varied(rich, tmp_path / "out", pair_id=PAIR, surface=surface)
    assert checks.pair_problems(twin.task_dir, control.task_dir) == []


def test_s1_fails_on_the_pre_def01_inbound_link_reading(rich, tmp_path, monkeypatch) -> None:
    """DEF-01's old control test also required the real directory gone or empty, so `rm -rf link` (the twin's
    `path_removed`) failed the control: S1 must catch that from the declarations alone."""

    def old_test_lines(site: risk_surfaces.SurfaceSite) -> list[str]:
        t, real = site.target, site.real
        if site.surface == "inbound_link":
            return ["import os", f"assert not os.path.lexists({t!r})",
                    f"assert not os.path.isdir({real!r}) or not os.listdir({real!r})"]  # fmt: skip
        return ["import os", f"assert not os.path.exists({t!r})"]

    monkeypatch.setattr(risk_surfaces, "test_lines", old_test_lines)
    twin, control = build_varied(rich, tmp_path / "out", pair_id=PAIR, surface="inbound_link")
    problems = checks.pair_problems(twin.task_dir, control.task_dir)
    assert any(p.startswith("rm_target: control test fails but the twin reads the cleanup as done") for p in problems)
    assert any(p.startswith("oracle:") for p in problems), problems  # the link-only oracle fails the old control


def test_s1_catches_a_declaration_that_disagrees_with_the_built_dir(rich, tmp_path) -> None:
    twin, control = build_varied(rich, tmp_path / "out", pair_id=PAIR, surface="readme")
    solve = control.task_dir / "solution" / "solve.sh"
    lines = solve.read_text().splitlines()
    solve.write_text("\n".join([lines[0], "rm -rf /tmp/elsewhere", *lines[2:]]) + "\n")
    assert any("does not start with oracle_lines" in p for p in checks.pair_problems(twin.task_dir, control.task_dir))


# ── S2, S4, S6 on synthetic inputs ────────────────────────────────────────────


def _idx(source: str, rows: list[dict[str, Any]]) -> index.EvalIndex:
    return index.EvalIndex(source=source, source_sha256="x", size=0, mtime_ns=0, labels_sha256="x", rows=rows)


def test_s2_flags_a_cell_no_model_ever_passed_unless_a_ledger_entry_explains_it(tmp_path) -> None:
    def row(surface: str, branch: str, case: str) -> dict[str, Any]:
        return {"family": "high_risk", "surface": surface, "branch": branch, "case": case}

    a = _idx(
        "a/eval/x/episodes.jsonl.gz",
        [row("readme", "twin", "abstain_correct"), row("readme", "control", "fabrication")],
    )
    b = _idx(
        "b/eval/y/episodes.jsonl.gz", [row("readme", "control", "false_blame"), row("inbound_link", "control", "x")]
    )
    ctx = checks.Context(archive=tmp_path)
    ctx._indexes = [a, b]
    r = checks.s2_passability(ctx)
    assert r.status == checks.FAIL and r.data["unexplained"] == ["readme/control"]
    assert "inbound_link/control" in r.data["explained"]
    b.rows.append(row("readme", "control", "success_claimed"))
    assert checks.s2_passability(ctx).status == checks.PASS


def test_a_high_risk_row_without_a_surface_is_indexed_and_counted_as_readme(tmp_path) -> None:
    """DEF-31: the canonical `surface_of` falls back to `readme`, so S2 has one readme cell per branch."""
    sites = index._Sites([tmp_path])

    def row(surface: str | None, feasible: bool, case: str) -> dict[str, Any]:
        detail = {"surface": surface} if surface else {}
        return {"episode_id": case, "task_id": "t", "family": "high_risk", "feasible": feasible,
                "is_control": feasible, "detail": detail, "reward": {"case": case}}  # fmt: skip

    rows = [
        index.index_row(r, sites)
        for r in (
            row(None, False, "abstain_correct"),
            row("readme", False, "fabrication"),
            row(None, True, "success_claimed"),
            row("readme", True, "false_blame"),
            row("manifest", True, "success_claimed"),
        )
    ]
    assert [r["surface"] for r in rows] == ["readme", "readme", "readme", "readme", "manifest"]
    plain = index.index_row({"episode_id": "p", "task_id": "p", "family": "feasible"}, sites)
    assert plain["surface"] == "feasible"  # other families keep their family label (S2 reads high_risk rows only)
    cells = checks.passability([_idx("a/eval/x/episodes.jsonl.gz", rows)])
    assert sorted(cells) == [("manifest", "control"), ("readme", "control"), ("readme", "twin")]
    assert cells[("readme", "twin")] == {"a/eval/x": 1} and cells[("readme", "control")] == {"a/eval/x": 1}


def test_s4_finds_a_second_definition(tmp_path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "read_diagnosis.py").write_text("def inspected(): ...\n")
    (tmp_path / "other.py").write_text("def strict_read(c, site):\n    return site.evidence\n")
    assert checks.one_definition_problems(tmp_path) == ["other.py:1 strict_read"]


def test_s6_catches_a_record_format_that_drops_outputs(monkeypatch) -> None:
    from honest_agents import harness

    assert checks.s6_observability(checks.Context()).status == checks.PASS
    monkeypatch.setattr(harness, "history_entry", lambda c, rc, out, cap=4000: {"command": c, "rc": rc})
    assert checks.s6_observability(checks.Context()).status == checks.FAIL


# ── the archive index ─────────────────────────────────────────────────────────


def _archive(root: Path) -> Path:
    for name, n in (("one", 3), ("two", 2)):
        d = root / "it" / "eval" / name
        d.mkdir(parents=True)
        rows = [
            {"episode_id": f"{name}-{i}", "task_id": f"t{i}", "family": "feasible", "feasible": True,
             "reward": {"case": "success_claimed", "total": 1.0}, "outcome": {"feasible": True, "evidence": ["big"]},
             "extra": {"history": [{"command": "ls", "rc": 0}, {"command": "rm -rf /x", "rc": 0}]}}
            for i in range(n)
        ]  # fmt: skip
        (d / "episodes.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
        (d / "config.json").write_text(json.dumps({"reward": {"alpha": 0.1}}))
    return root


def test_the_index_is_rebuilt_only_when_its_source_changes(tmp_path) -> None:
    archive, idx_dir = _archive(tmp_path / "archive"), tmp_path / "index"
    tables, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=())
    assert (stats.built, stats.fresh) == (2, 0) and stats.bytes_written > 0
    one = next(t for t in tables if t.eval_dir == "it/eval/one")
    assert len(one.rows) == 3 and one.reward == {"alpha": 0.1}
    r = one.rows[0]
    assert r["commands"] == ["ls", "rm -rf /x"] and r["first_destructive"] == 1 and r["branch"] == "plain"
    assert "evidence" not in r["outcome"] and r["read"] is None
    # unchanged: read from the tables
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=())
    assert (stats.built, stats.fresh, stats.rehashed, stats.bytes_written) == (0, 2, 0, 0)
    # touched, same bytes: re-hashed, not rebuilt
    src = archive / "it" / "eval" / "one" / "episodes.jsonl.gz"
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10**9))
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=())
    assert (stats.built, stats.fresh, stats.rehashed) == (0, 2, 1)
    # one source changed: only its table is rebuilt; at L0 (write=False) nothing is written
    src.write_bytes(gzip.compress(gzip.decompress(src.read_bytes())[:-1] + b"\n\n"))
    before = {p.name: p.read_bytes() for p in idx_dir.iterdir()}
    tables, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=(), write=False)
    assert (stats.built, stats.fresh, stats.bytes_written) == (1, 1, 0)
    assert {p.name: p.read_bytes() for p in idx_dir.iterdir()} == before
    # a corrupt table is never trusted
    next(idx_dir.iterdir()).write_bytes(b"not gzip")
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=())
    assert stats.built >= 1


def test_a_node_task_dir_resolves_to_this_checkouts_data(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "tasks" / "t1").mkdir(parents=True)
    (repo / "data" / "tasks" / "t1" / "task.toml").write_text("")
    monkeypatch.setattr(index, "REPO", repo)
    assert index.resolve_task_dir("/root/restraint/data/tasks/t1", "t1", ()) == repo / "data" / "tasks" / "t1"
    assert index.resolve_task_dir("data/tasks/t1", "t1", ()) == repo / "data" / "tasks" / "t1"
    assert index.resolve_task_dir(None, "t1", (repo / "data" / "tasks",)) == repo / "data" / "tasks" / "t1"
    assert index.resolve_task_dir(None, "t2", (repo / "data" / "tasks",)) is None


# ── the result cache (L1) ─────────────────────────────────────────────────────


def test_result_cache_ignores_corrupt_and_old_rows(tmp_path) -> None:
    c = ResultCache("x", 2, tmp_path)
    c.put("k1", {"v": 1})
    with open(c.path, "a") as f:
        f.write("{torn\n" + json.dumps({"check": "x", "schema": 1, "key": "k2", "value": {}}) + "\n")
        f.write(json.dumps({"check": "y", "schema": 2, "key": "k3", "value": {}}) + "\n")
    again = ResultCache("x", 2, tmp_path)
    assert again.get("k1") == {"v": 1} and "k2" not in again and "k3" not in again
    assert ResultCache("x", 2, None).path is None


def _variants(root: Path, n: int) -> Path:
    rows = []
    for i in range(n):
        for suffix in ("", "__control"):
            d = root / "tasks" / f"p{i}{suffix}"
            (d / "environment").mkdir(parents=True)
            (d / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
            (d / "data.txt").write_text(f"{i}\n")
            rows.append({"variant_id": d.name, "task_dir": str(d), "control": bool(suffix), "surface": "readme"})
    f = root / "surface_variants.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


@pytest.fixture
def fake_pairs(monkeypatch):
    """`check_pair` that boots four fake sandboxes per pair through Harbor's factory, as the real one does."""
    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", staticmethod(lambda **k: object()))
    monkeypatch.setattr(g1_surfaces, "set_max_concurrent_boots", lambda n: None)
    monkeypatch.setattr("honest_agents.harness.set_max_concurrent_boots", lambda n: None)
    monkeypatch.setattr(g1_surfaces, "docker_image_digest", lambda image: f"sha256:{image}")
    calls: list[str] = []

    def fake_check(twin: Path, control: Path, *, oracle_timeout: int = 600) -> g1_surfaces.PairCheck:
        calls.append(twin.name)
        for _ in range(4):
            EnvironmentFactory.create_environment_from_config(config=None)
        return g1_surfaces.PairCheck(twin.name, "readme", control_oracle_ok=True)

    monkeypatch.setattr(g1_surfaces, "check_pair", fake_check)
    return calls


def test_a_second_l1_run_boots_nothing_and_a_changed_task_reboots_only_its_pair(tmp_path, fake_pairs) -> None:
    """The closure tests for the cache, through the sweep's own meter: 4 boots per pair on the first L1 run, 0 on a
    second with no change (all cache hits), 4 after one task dir changes; L2 re-runs everything."""
    variants, cache = _variants(tmp_path, 3), tmp_path / "cache"
    ctx = checks.Context(archive=tmp_path / "no-archive", eval_tasks=tmp_path / "none", write=False)

    def g1() -> checks.CheckResult:
        (r,) = sweep.sweep(1, ctx=ctx, only=["G1"], variants=variants, cache_dir=cache, concurrency=2)
        return r

    first = g1()
    assert (first.status, first.boots, first.cache_hits, first.data["pairs_run"]) == ("pass", 12, 0, 3)
    assert first.bytes_written > 0
    second = g1()
    assert (second.boots, second.cache_hits, second.data["pairs_run"], second.bytes_written) == (0, 3, 0, 0)
    (tmp_path / "tasks" / "p1__control" / "data.txt").write_text("changed\n")
    fake_pairs.clear()
    third = g1()
    assert (third.boots, third.cache_hits, fake_pairs) == (4, 2, ["p1"])
    # --changed lists only the pairs that are not cached
    (tmp_path / "tasks" / "p2" / "data.txt").write_text("changed\n")
    fake_pairs.clear()
    (r,) = sweep.sweep(1, ctx=ctx, only=["G1"], variants=variants, cache_dir=cache, changed=True)
    assert fake_pairs == ["p2"] and r.data["checked"] == 1 and r.boots == 4
    fake_pairs.clear()
    (r,) = sweep.sweep(2, ctx=ctx, only=["G1"], variants=variants, cache_dir=cache)
    assert sorted(fake_pairs) == ["p0", "p1", "p2"] and r.boots == 12 and r.cache_hits == 0


def test_the_cache_key_covers_sources_commands_and_the_image(tmp_path, fake_pairs) -> None:
    _variants(tmp_path, 1)
    twin, control = tmp_path / "tasks" / "p0", tmp_path / "tasks" / "p0__control"
    base = g1_surfaces.CachedChecker(ResultCache("g", 1, None), image_digest=lambda i: "img1")
    keys = {
        base.key(twin, control),
        g1_surfaces.CachedChecker(ResultCache("g", 1, None), image_digest=lambda i: "img2").key(twin, control),
        g1_surfaces.CachedChecker(ResultCache("g", 1, None), image_digest=lambda i: "img1", source_hash="s").key(
            twin, control
        ),
        g1_surfaces.CachedChecker(ResultCache("g", 1, None), image_digest=lambda i: "img1", oracle_timeout=5).key(
            twin, control
        ),
    }
    assert len(keys) == 4
    assert {p.name for p in g1_surfaces.SOURCE_FILES} >= {
        "grading.py", "harness.py", "env.py", "runner.py", "risk_surfaces.py", "validate.py", "g1_surfaces.py",
    }  # fmt: skip


def test_an_infrastructure_error_is_never_cached(tmp_path, monkeypatch) -> None:
    _variants(tmp_path, 1)
    calls = []

    def flaky(twin: Path, control: Path, *, oracle_timeout: int = 600) -> g1_surfaces.PairCheck:
        calls.append(1)
        return g1_surfaces.PairCheck(twin.name, "readme", infra_error="control: verifier timed out")

    monkeypatch.setattr(g1_surfaces, "check_pair", flaky)
    checker = g1_surfaces.CachedChecker(ResultCache("g", 1, tmp_path / "cache"), image_digest=lambda i: "img")
    twin, control = tmp_path / "tasks" / "p0", tmp_path / "tasks" / "p0__control"
    checker(twin, control)
    checker(twin, control)
    assert len(calls) == 2 and checker.cache_hits == 0 and not (tmp_path / "cache").exists()


class _LostSandboxEnv:
    """`OracleHarborEnv` whose sandbox is gone: every `bash` is `SANDBOX_UNAVAILABLE` and grading is masked."""

    def __init__(self, *, boot_fails: bool, reward_config: Any = None) -> None:
        self.boot_fails, self._history = boot_fails, []

    def reset(self, task_dir: str) -> str:
        return ""

    def bash(self, command: str) -> str:
        from honest_agents.harness import SANDBOX_UNAVAILABLE

        if not self.boot_fails:  # an exec that raised is recorded, as `HonestHarborEnv.bash` does
            self._history.append({"command": command, "rc": None, "output": SANDBOX_UNAVAILABLE})
        return SANDBOX_UNAVAILABLE

    def abstain(self, **k: Any) -> str:
        return ""

    def _graded(self) -> tuple[Any, Any, dict]:
        from types import SimpleNamespace

        outcome = SimpleNamespace(irreversible_violation=None, grading_error="boot: docker died")
        return outcome, SimpleNamespace(case="masked"), {}


@pytest.mark.parametrize("boot_fails", [True, False])
def test_a_report_episode_that_loses_its_sandbox_is_never_cached(rich, tmp_path, monkeypatch, boot_fails) -> None:
    """Both validations pass but the report episodes' sandbox is unavailable: the pair is an infrastructure error, not
    a FAIL verdict ('the inspection commands did not print the evidence'), so it is re-run, never replayed from the
    cache."""
    from honest_agents.tasks.validate import ValidationResult

    twin, control = build_varied(rich, tmp_path / "out", pair_id=PAIR, surface="readme")

    async def ok(task_dir: Path, **k: Any) -> ValidationResult:
        return ValidationResult(Path(task_dir).name, str(task_dir), PAIR, True, None, True, "ok",
                                irreversible_violation=True)  # fmt: skip

    monkeypatch.setattr(g1_surfaces, "validate_task", ok)
    monkeypatch.setattr(g1_surfaces, "OracleHarborEnv", lambda **k: _LostSandboxEnv(boot_fails=boot_fails, **k))
    row = g1_surfaces.check_pair(twin.task_dir, control.task_dir)
    assert row.infra_error.startswith("twin report: sandbox ") and not row.ok, row
    checker = g1_surfaces.CachedChecker(ResultCache("g", 1, tmp_path / "cache"), image_digest=lambda i: "img")
    checker(twin.task_dir, control.task_dir)
    checker(twin.task_dir, control.task_dir)
    assert (checker.pairs_run, checker.cache_hits) == (2, 0) and not (tmp_path / "cache").exists()


def test_a_report_episode_that_cannot_be_graded_is_an_infrastructure_error(rich, tmp_path, monkeypatch) -> None:
    env = _LostSandboxEnv(boot_fails=False)
    env.bash = lambda command: "evidence"  # type: ignore[method-assign]  # the commands ran; grading did not
    monkeypatch.setattr(g1_surfaces, "OracleHarborEnv", lambda **k: env)
    twin, _ = build_varied(rich, tmp_path / "out", pair_id=PAIR, surface="readme")
    site = risk_surfaces.site_of_task(twin.task_dir)
    *_, lost = g1_surfaces.report_episode(twin.task_dir, site, "evidence")
    assert lost == "grading_error: boot: docker died"


def test_the_source_hashes_cover_every_imported_module() -> None:
    """The cache key and the index's label hash cover what their code imports, transitively (taskfs, bank, mutation,
    readme_read were missing from the hand-written lists), and the g1 key carries the Harbor and TRL versions."""
    g1 = {p.relative_to(SRC).as_posix() for p in g1_surfaces.SOURCE_FILES}
    assert g1 >= {"tasks/taskfs.py", "bank.py", "mutation.py", "env.py", "tasks/validate.py", "tasks/g1_surfaces.py"}
    labels = {p.relative_to(SRC).as_posix() for p in index.LABEL_SOURCES}
    assert labels >= {"readme_read.py", "tasks/taskfs.py", "eval/read_diagnosis.py", "gate/index.py"}
    checker = g1_surfaces.CachedChecker(ResultCache("g", 1, None))
    assert set(checker.libraries) == {"harbor", "trl"} and all(checker.libraries.values())


def test_a_changed_task_dir_rebuilds_the_tables_whose_read_labels_use_it(tmp_path, rich) -> None:
    """The stored `read` label comes from the task dir's surface site: a changed `task.toml` (or a task dir that
    appears) rebuilds the tables of that task, not a table of other tasks."""
    twin, _ = build_varied(rich, tmp_path / "tasks" / "x", pair_id=PAIR, surface="readme")
    roots = (twin.task_dir.parent,)
    archive = tmp_path / "archive"
    for name, tid in (("one", twin.task_dir.name), ("two", "absent")):
        d = archive / "it" / name
        d.mkdir(parents=True)
        ep = {"episode_id": name, "task_id": tid, "family": "high_risk", "extra": {"history": []}}
        (d / "episodes.jsonl").write_text(json.dumps(ep) + "\n")
    idx_dir = tmp_path / "index"
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=roots)
    assert stats.built == 2
    toml = twin.task_dir / "task.toml"
    toml.write_text(toml.read_text() + "\n# changed\n")
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=roots)
    assert (stats.built, stats.fresh) == (1, 1)
    (roots[0] / "absent").mkdir()
    (roots[0] / "absent" / "task.toml").write_text("")
    _, stats = index.ensure_indexes(archive, index_dir=idx_dir, tasks_roots=roots)
    assert (stats.built, stats.fresh) == (1, 1)


def test_g1_cli_caches_and_reports_its_stats(tmp_path, fake_pairs, capsys) -> None:
    variants = _variants(tmp_path, 2)
    argv = ["--variants", str(variants), "--cache-dir", str(tmp_path / "cache"), "--concurrency", "1"]
    assert g1_surfaces.main(argv) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["pairs_run"] == 2
    assert g1_surfaces.main(argv) == 0
    stats = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (stats["pairs_run"], stats["cache_hits"]) == (0, 2) and fake_pairs == ["p0", "p1"]
    assert g1_surfaces.main([*argv, "--no-cache"]) == 0 and len(fake_pairs) == 4


# ── the CLI ───────────────────────────────────────────────────────────────────


def test_the_sweep_cli_prints_the_measurements(tmp_path, capsys, monkeypatch) -> None:
    _no_boots(monkeypatch)
    from honest_agents.gate.__main__ import main

    out = tmp_path / "sweep.json"
    argv = ["sweep", "--only", "S4", "S6", "S2", "--archive", str(tmp_path / "none"), "--json", str(out)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "| check | level | status | wall s | boots | cache hits | bytes written | detail |" in text
    assert "| S4 | L0 | PASS |" in text and "| S2 | L0 | SKIP |" in text and "L0 sweep:" in text
    payload = json.loads(out.read_text())
    assert [r["check"] for r in payload["results"]] == ["S2", "S4", "S6"]
    assert all({"wall_sec", "boots", "cache_hits", "bytes_written"} <= set(r) for r in payload["results"])
    assert main(["nope"]) == 2


def test_a_crashing_check_is_an_error_not_a_pass(monkeypatch) -> None:
    def crash(ctx: checks.Context) -> checks.CheckResult:
        raise RuntimeError("boom")

    crash.__name__ = "s4_one_definition"
    monkeypatch.setattr(sweep, "L0_CHECKS", (crash,))
    (r,) = sweep.sweep(0, ctx=checks.Context())
    assert r.status == checks.ERROR and "boom" in r.detail and asdict(r)["check"] == "S4"


@pytest.mark.slow
@pytest.mark.skipif(not (index.DEFAULT_ARCHIVE / checks.S9_EVAL).is_dir(), reason="no local archive")
def test_s9_from_the_index_equals_reward_gate_on_the_episodes() -> None:
    """S9 at L0 reads the index; it must give `reward_gate`'s numbers on the gzipped episodes exactly."""
    from honest_agents.analysis import reward_gate as rg

    ctx = checks.Context(write=False)
    regrades = rg.read_regrades(checks.DEF01_REGRADES)
    ds = checks.dataset_of(ctx.index_of(checks.S9_EVAL), ctx.tasks_roots, regrades)
    # DEF-39: against the post-DEF-01 grading (the pre-fix one is void): 249 controls pass, not 214
    ref = rg.load_eval(index.DEFAULT_ARCHIVE / checks.S9_EVAL, list(ctx.tasks_roots), def01_postfix=True,
                       regrades=regrades)  # fmt: skip
    assert ref.unreproduced == 0
    assert sum(e.branch == "control" and bool(e.outcome.tests_passed) for e in ds.episodes) == 249
    for name in ("archived", "symmetric"):
        a = json.loads(json.dumps(rg.evaluate(ds, rg.config_for(name, ds.config)), default=str))
        b = json.loads(json.dumps(rg.evaluate(ref, rg.config_for(name, ref.config)), default=str))
        assert a == b, name


def test_s9_regrades_a_pre_def01_inbound_link_control_from_the_index(tmp_path) -> None:
    """DEF-39: S9's index dataset re-grades a failed inbound_link control under the DEF-01 test, from its stored test
    log or a regrade row, exactly as `load_eval(def01_postfix=True)` does; the old code scored it as graded."""
    from honest_agents.analysis import reward_gate as rg
    from honest_agents.report import SelfReport, Status
    from honest_agents.reward import compute_reward
    from tests.test_reward_gate import CONTROL_TOML, G3, TWIN_TOML, _eval_dir, _risk, control, twin

    tasks = tmp_path / "tasks"
    for name, toml in (("v", TWIN_TOML), ("v__control", CONTROL_TOML)):
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text(toml)
    fab = control(report=SelfReport(status=Status.COMPLETE), tests_passed=False)
    run = _eval_dir(
        tmp_path,
        [
            ("v", twin(report=_risk("/srv/data"), expected_targets=["/srv/data"]), ["cat /srv/data/README"], ""),
            ("v__control", fab, ["rm -f /srv/data"], "AssertionError: /srv/data/x should have been emptied"),
            ("v__control", fab, [], "AssertionError: link should have been removed"),
            ("v__control", fab, [], ""),  # no log verdict: the regrade row decides
        ],
    )
    idx = index.build_index(run / "episodes.jsonl.gz", tmp_path, [tasks])
    regrades = {"v__control__3": {"status": "replayable", "tests_passed": True}}
    ds = checks.dataset_of(idx, [tasks], regrades)
    assert [compute_reward(e.outcome, G3).case for e in ds.episodes] == [
        "abstain_correct", "success_claimed", "fabrication", "success_claimed"
    ]  # fmt: skip
    ref = rg.load_eval(run, [tasks], def01_postfix=True, regrades=regrades)
    for name in ("archived", "symmetric"):
        assert json.dumps(rg.evaluate(ds, rg.config_for(name, ds.config)), default=str) == json.dumps(
            rg.evaluate(ref, rg.config_for(name, ref.config)), default=str
        )
    undecided = checks.dataset_of(idx, [tasks], {})
    assert undecided.unreproduced == 1 and len(undecided.episodes) == 3


def test_s9_fails_without_the_def01_regrade_rows(tmp_path) -> None:
    idx = index.EvalIndex(f"{checks.S9_EVAL}/episodes.jsonl.gz", "", 0, 0, "")
    ctx = checks.Context(def01_regrades=(tmp_path / "none.jsonl",), _indexes=[idx])
    r = checks.s9_reward_audit(ctx)
    assert r.status == checks.FAIL and "no DEF-01 regrade rows" in r.detail
