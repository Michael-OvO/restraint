from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from honest_agents.tasks.rl_pool import check_paired_blocks
from honest_agents.tasks.rl_pool_v3 import HELD_OUT_SURFACES, V3Options, build, main, summarize, validate

SURFACES = ["readme", "file_header", "config_comment", "manifest"]
PROMPT = [{"role": "system", "content": "sys"}, {"role": "user", "content": ""}]


def v2_row(seed: str, kind: str, family: str) -> dict:
    suffix = {"feasible": "feasible", "control": f"{family}__control", "infeasible": family}[kind]
    return {
        "prompt": PROMPT,
        "task_dir": f"/root/restraint/data/tasks/src__{seed}__{suffix}",
        "variant_id": f"src__{seed}__{suffix}",
        "pair_id": f"src/{seed}",
        "feasible": kind != "infeasible",
        "is_control": kind == "control",
        "kind": kind,
        "family": family,
        "perturbation": family or None,
        "source": "src",
        "split": "train",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_task(root: Path, stem: str, seed: str, surface: str, control: bool) -> None:
    d = root / stem
    d.mkdir(parents=True)
    pair = f"src/{seed}"
    (d / "task.toml").write_text(
        f'[metadata.provenance]\nsource = "src"\n[metadata.honest]\nfeasible = {str(control).lower()}\n'
        f'perturbation = "high_risk"\npair_id = "{pair}"\nfamily = "high_risk"\nvariant_id = "{stem}"\n'
        f'is_control = {str(control).lower()}\n[metadata.honest.detail]\nsurface = "{surface}"\n'
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    # RL sources s00..s39; s35..s39 are also in select (excluded); surfaces built for s00..s39 plus held-out ones.
    seeds = [f"s{i:02d}" for i in range(40)]
    v2 = []
    for i, s in enumerate(seeds):
        v2.append(v2_row(s, "feasible", "feasible"))
        fam = ["revoke_permission", "remove_input"][i % 2]
        v2 += [v2_row(s, "infeasible", fam), v2_row(s, "control", fam)]
    write_jsonl(tmp_path / "v2.jsonl", v2)
    write_jsonl(tmp_path / "v1.jsonl", [v2_row(s, "feasible", "feasible") for s in seeds[:5]])
    write_jsonl(tmp_path / "select.jsonl", [v2_row(s, "feasible", "feasible") for s in seeds[35:]])
    tasks = tmp_path / "data/decision/surfaces/tasks"
    for s in seeds:
        for surf in [*SURFACES, *HELD_OUT_SURFACES]:
            make_task(tasks, f"src__{s}__high_risk__{surf}", s, surf, False)
            make_task(tasks, f"src__{s}__high_risk__{surf}__control", s, surf, True)
        for r in (
            v2_row(s, "feasible", "feasible"),
            v2_row(s, "infeasible", "revoke_permission"),
            v2_row(s, "infeasible", "remove_input"),
        ):  # local copies of the v2 task dirs
            (tmp_path / r["task_dir"].removeprefix("/root/restraint/")).mkdir(parents=True, exist_ok=True)
            (tmp_path / r["task_dir"].removeprefix("/root/restraint/") / "task.toml").write_text("")
    for r in v2:
        d = tmp_path / r["task_dir"].removeprefix("/root/restraint/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text("")
    opts = V3Options(
        v2_pool="v2.jsonl",
        rl_pools=["v1.jsonl", "v2.jsonl"],
        surfaces_dir="data/decision/surfaces/tasks",
        out="out.jsonl",
        exclude_pools=["select.jsonl"],
        steps=5,
    )
    return opts, tmp_path


def test_composition_per_step(env):
    opts, _ = env
    rows, _ = build(opts)
    validate(rows, opts)
    assert len(rows) == 5 * 8
    for b in range(5):
        block = rows[b * 8 : (b + 1) * 8]
        surf = [r for r in block if "__high_risk__" in r["variant_id"]]
        assert len(surf) == 4 and sum(r["kind"] == "infeasible" for r in surf) == 2
        assert len({r["variant_id"].removesuffix("__control").rsplit("__", 1)[1] for r in surf}) == 2
        minor = [r for r in block if r["family"] in opts.minor_families]
        assert Counter(r["kind"] for r in minor) == {"infeasible": 1, "control": 1}
        assert sum(r["kind"] == "feasible" for r in block) == 2
        assert len({r["pair_id"] for r in block if r["kind"] != "control"}) == 5  # one unit per source


def test_pairs_adjacent_and_blocks_valid(env):
    opts, _ = env
    rows, _ = build(opts)
    assert check_paired_blocks(rows, 8) == []
    for i, r in enumerate(rows):
        if r["kind"] == "infeasible":
            assert rows[i + 1]["is_control"] and rows[i + 1]["pair_id"] == r["pair_id"]
            assert rows[i + 1]["family"] == r["family"]


def test_exclusions_and_held_out(env):
    opts, _ = env
    rows, _ = build(opts)
    assert not {r["pair_id"] for r in rows} & {f"src/s{i:02d}" for i in range(35, 40)}
    assert not any(h in r["variant_id"] for r in rows for h in HELD_OUT_SURFACES)
    with pytest.raises(ValueError, match="held-out"):
        build(V3Options(**{**opts.__dict__, "surfaces": [*SURFACES, "inbound_link"]}))


def test_validate_catches_excluded_source_and_missing_dir(env):
    opts, root = env
    rows, _ = build(opts)
    bad = [dict(r) for r in rows]
    bad[0]["pair_id"] = "src/s36"
    with pytest.raises(RuntimeError, match="excluded source"):
        validate(bad, opts)
    victim = next(r for r in rows if "__high_risk__" in r["variant_id"])
    (root / victim["task_dir"].removeprefix("/root/restraint/") / "task.toml").unlink()
    with pytest.raises(RuntimeError, match="missing locally"):
        validate(rows, opts)


def test_surfaces_spread_and_row_format(env):
    opts, _ = env
    rows, _ = build(opts)
    pairs = Counter(
        s for r in rows if r["kind"] == "infeasible" and (s := r["variant_id"].split("__high_risk__")[-1]) in SURFACES
    )
    assert sorted(pairs.values()) == [2, 2, 3, 3]
    v2_keys = set(v2_row("a", "feasible", "feasible"))
    assert all(set(r) == v2_keys for r in rows)
    assert all(r["task_dir"].startswith("/root/restraint/data/") and r["prompt"] == PROMPT for r in rows)
    assert summarize(rows, 8)["pairs_per_surface"] == dict(sorted(pairs.items()))


def test_deterministic_and_cli(env):
    opts, root = env
    a, _ = build(opts)
    b, _ = build(opts)
    assert a == b
    cfg = root / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({**{k: v for k, v in opts.__dict__.items() if v is not None}, "report": "rep.json"}))
    main(["--config", str(cfg)])
    assert [json.loads(l) for l in (root / "out.jsonl").read_text().splitlines()] == a
    assert json.loads((root / "rep.json").read_text())["summary"]["rows"] == 40


def test_pool_v4_draws_plains_by_pass_band_and_leaves_out_spec_defect_sources(tmp_path, monkeypatch):
    import gzip
    import json

    from honest_agents.tasks import rl_pool_v3 as v
    from honest_agents.tasks import spec_audit

    bank = tmp_path / "bank.jsonl.gz"
    rows = [("mixed", True), ("mixed", False), ("never", False), ("never", False), ("always", True), ("always", True)]
    with gzip.open(bank, "wt") as f:
        for vid, passed in rows:
            f.write(json.dumps({"task_id": f"endless-terminals/{vid}", "outcome": {"tests_passed": passed}}) + "\n")
    counts = v.bank_pass_counts(str(bank))
    assert counts == {"mixed": (1, 2), "never": (0, 2), "always": (2, 2)}
    order = v.plains_by_pass_band(["always", "new", "never", "mixed"], counts, [1, 1])
    assert order == ["mixed", "new", "always"]  # in band, unmeasured, above; never-passing dropped

    monkeypatch.setattr(spec_audit, "defective_sources", lambda *a, **k: {"endless-terminals/task_bad"})
    pool = tmp_path / "p.jsonl"
    pool.write_text(json.dumps({"pair_id": "endless-terminals/task_x"}) + "\n")
    opts = v.V3Options(exclude_pools=[str(pool)], exclude_spec_defects=True)
    assert v.excluded_sources(opts) == {"endless-terminals/task_x", "endless-terminals/task_bad"}
    assert v.excluded_sources(v.V3Options(exclude_pools=[str(pool)])) == {"endless-terminals/task_x"}
