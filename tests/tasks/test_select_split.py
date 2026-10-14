"""The select split: a stable, exclusion-respecting carve-out of train sources, and the probe set built from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.tasks import probe_set, select_split

N_SOURCES = 40


def toml(pair: str, name: str, *, family: str | None, feasible: bool, control: bool = False) -> str:
    lines = [
        "[metadata.provenance]",
        'source = "endless-terminals"',
        f'pair_id = "{pair}"',
        "[metadata.honest]",
        f"feasible = {str(feasible).lower()}",
        f'pair_id = "{pair}"',
        f'variant_id = "{name}"',
        f"is_control = {str(control).lower()}",
    ]
    if family:
        lines += [f'family = "{family}"', f'perturbation = "{family}"']
    return "\n".join(lines) + "\n"


def make_data(tmp_path: Path) -> Path:
    """40 train sources t00..t39 (all with a high_risk pair and a plain task; t00..t05 also revoke_permission, t06 also
    a held-out corrupt_input pair), 5 val and 5 test sources; t00..t09 are distill sources, t10..t14 RL v1, t15..t17
    RL v2, t18..t19 teacher-only, t20 only in the SFT mixture; t39 has no high_risk twin."""
    d = tmp_path / "data"
    (d / "tasks").mkdir(parents=True)
    (d / "teacher").mkdir()
    seeds = {f"endless-terminals/t{i:02d}": "train" for i in range(N_SOURCES)}
    seeds |= {f"endless-terminals/v{i}": "val" for i in range(5)} | {
        f"endless-terminals/x{i}": "test" for i in range(5)
    }
    (d / "split_seeds.json").write_text(json.dumps(seeds))
    for pair in seeds:
        short = pair.split("/")[1]
        variants = [(f"{short}__feasible", None, True, False)]
        if short != f"t{N_SOURCES - 1}":
            variants += [
                (f"{short}__high_risk", "high_risk", False, False),
                (f"{short}__high_risk__control", "high_risk", True, True),
            ]
        if short in {f"t0{i}" for i in range(6)}:
            variants += [(f"{short}__revoke_permission", "revoke_permission", False, False)]
        if short == "t06":
            variants += [(f"{short}__corrupt_input", "corrupt_input", False, False)]
        for name, family, feasible, control in variants:
            (d / "tasks" / f"endless-terminals__{name}").mkdir()
            (d / "tasks" / f"endless-terminals__{name}" / "task.toml").write_text(
                toml(pair, f"endless-terminals__{name}", family=family, feasible=feasible, control=control)
            )

    def pool(name: str, ids: range) -> None:
        (d / name).write_text("".join(json.dumps({"pair_id": f"endless-terminals/t{i:02d}"}) + "\n" for i in ids))

    pool("distill_pool.jsonl", range(10))
    pool("distill_pool_x2.jsonl", range(4))
    pool("train_rl.jsonl", range(10, 15))
    pool("train_rl_v2.jsonl", range(15, 18))
    pool("train.jsonl", range(20, 21))
    (d / "teacher" / "sources.txt").write_text("endless-terminals/t18\nendless-terminals/t19\n")
    return d


def test_training_sources_are_the_union_of_every_pool_and_the_teacher_bank(tmp_path: Path):
    d = make_data(tmp_path)
    assert select_split.training_sources(d) == {f"endless-terminals/t{i:02d}" for i in range(21)}


def test_select_sources_are_train_split_unused_and_eligible(tmp_path: Path):
    d = make_data(tmp_path)
    chosen, rows = select_split.build(d, n=10)
    used = select_split.training_sources(d)
    train = select_split.split_sources(d, "train")
    assert len(chosen) == 10 and chosen == sorted(chosen)
    assert set(chosen) <= train and not set(chosen) & used
    assert "endless-terminals/t39" not in chosen  # no high_risk twin: not eligible
    assert {r["pair_id"] for r in rows} == set(chosen) and {r["split"] for r in rows} == {"select"}
    # all existing variants of a source, held-out families left out
    by_src = {s: [r for r in rows if r["pair_id"] == s] for s in chosen}
    assert all({(r["family"], r["kind"]) for r in v} >= {("high_risk", "infeasible"), ("high_risk", "control"), ("feasible", "feasible")}
               for v in by_src.values())  # fmt: skip
    assert not any(r["family"] == "corrupt_input" for r in rows)
    assert (
        rows[0]["task_dir"].startswith("/root/restraint/data/tasks/endless-terminals__")
        and rows[0]["prompt"][0]["role"] == "system"
    )
    assert [r["kind"] for r in by_src[chosen[0]]][:2] == ["feasible", "control"]  # plain, controls, then twins


def test_the_choice_is_a_function_of_the_source_ids_not_the_file_order(tmp_path: Path):
    d = make_data(tmp_path)
    a, _ = select_split.build(d, n=12)
    candidates = sorted(select_split.split_sources(d, "train") - select_split.training_sources(d), reverse=True)
    variants = select_split.variants_by_source(d / "tasks", candidates)
    assert sorted(select_split.choose_sources(candidates, variants, 12)) == a
    assert select_split.choose_sources(candidates, variants, 12, seed=1) != select_split.choose_sources(
        candidates, variants, 12, seed=0
    )
    with pytest.raises(ValueError, match="only 18 eligible"):
        select_split.choose_sources(candidates, variants, 19)  # 40 - 21 used - t39 = 18


def test_cli_writes_the_files_and_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    d = make_data(tmp_path)
    assert select_split.main(["--data", str(d), "--n-sources", "8"]) == 0
    report = json.loads(capsys.readouterr().out)
    sources = select_split.load_select_sources(d)
    assert (
        len(sources)
        == 8
        == report["sources"]
        == len({json.loads(x)["pair_id"] for x in (d / "select.jsonl").read_text().splitlines()})
    )
    assert report["by_family_kind"]["high_risk / infeasible"] == 8 and report["by_kind"]["feasible"] == 8
    with pytest.raises(SystemExit, match="frozen"):
        select_split.main(["--data", str(d)])
    assert select_split.main(["--data", str(d), "--n-sources", "8", "--force"]) == 0


def test_force_keeps_the_frozen_sources_and_only_refreshes_the_rows(tmp_path: Path):
    d = make_data(tmp_path)
    assert select_split.main(["--data", str(d), "--n-sources", "8"]) == 0
    frozen = select_split.load_select_sources(d)
    # Twins get built for 20 more train sources later (they would enter a fresh choice), and a frozen source gains a
    # new-surface variant.
    seeds = json.loads((d / "split_seeds.json").read_text())
    for i in range(20):
        pair = f"endless-terminals/n{i:02d}"
        seeds[pair] = "train"
        for name, family, feasible, control in [
            (f"n{i:02d}__feasible", None, True, False),
            (f"n{i:02d}__high_risk", "high_risk", False, False),
            (f"n{i:02d}__high_risk__control", "high_risk", True, True),
        ]:
            (d / "tasks" / f"endless-terminals__{name}").mkdir()
            (d / "tasks" / f"endless-terminals__{name}" / "task.toml").write_text(
                toml(pair, f"endless-terminals__{name}", family=family, feasible=feasible, control=control)
            )
    (d / "split_seeds.json").write_text(json.dumps(seeds))
    first = min(frozen)
    extra = d / "tasks" / f"endless-terminals__{first.split('/')[1]}__high_risk__manifest"
    extra.mkdir()
    extra.joinpath("task.toml").write_text(
        toml(first, extra.name, family="high_risk", feasible=False) + '[metadata.honest.detail]\nsurface = "manifest"\n'
    )
    fresh, _ = select_split.build(d, n=8)
    assert set(fresh) != frozen  # a fresh choice would move the split
    assert select_split.main(["--data", str(d), "--n-sources", "8", "--force"]) == 0
    assert select_split.load_select_sources(d) == frozen
    rows = select_split.read_jsonl(d / "select.jsonl")
    assert {r["pair_id"] for r in rows} == frozen
    assert any(r["variant_id"] == extra.name and r["detail"] == {"surface": "manifest"} for r in rows)
    with pytest.raises(SystemExit, match="delete both files"):
        select_split.main(["--data", str(d), "--n-sources", "9", "--force"])
    (d / "train_rl_v3.jsonl").write_text(json.dumps({"pair_id": first}) + "\n")
    with pytest.raises(SystemExit, match="now used by training"):
        select_split.main(["--data", str(d), "--n-sources", "8", "--force"])


def test_excluded_from_training_covers_select_val_and_test_and_needs_the_select_file(tmp_path: Path):
    d = make_data(tmp_path)
    with pytest.raises(FileNotFoundError, match="select_sources.txt"):
        select_split.excluded_from_training(d)
    select_split.main(["--data", str(d), "--n-sources", "8"])
    ex = select_split.excluded_from_training(d)
    assert select_split.load_select_sources(d) <= ex
    assert {f"endless-terminals/v{i}" for i in range(5)} | {f"endless-terminals/x{i}" for i in range(5)} <= ex
    assert not ex & select_split.training_sources(d)  # the sources training already uses are not in it


def test_the_select_probe_set_is_the_whole_split_and_refuses_other_splits(tmp_path: Path):
    d = make_data(tmp_path)
    _, rows = select_split.build(d, n=8)
    sources = {r["pair_id"] for r in rows}
    assert probe_set.select_split_probe(rows, sources=sources) == rows  # file order, nothing sampled
    dropped = probe_set.select_split_probe(rows, sources=sources, exclude={rows[0]["variant_id"]})
    assert len(dropped) == len(rows) - 1
    with pytest.raises(ValueError, match="select rows only"):
        probe_set.select_split_probe([{**rows[0], "split": "val"}])
    with pytest.raises(ValueError, match="not a select source"):
        probe_set.select_split_probe(rows, sources=sources - {rows[0]["pair_id"]})
    with pytest.raises(ValueError, match="have no row"):
        probe_set.select_split_probe([r for r in rows if r["pair_id"] != rows[0]["pair_id"]], sources=sources)
    # the validation probe still refuses select rows, and the same sampler accepts them when told to
    with pytest.raises(ValueError, match="validation rows only"):
        probe_set.select_probe_set(rows, n_infeasible=4, n_feasible=4)
    chosen = probe_set.select_probe_set(rows, n_infeasible=4, n_feasible=4, splits=probe_set.SELECT_SPLITS)
    assert sum(r["kind"] == "infeasible" for r in chosen) == 4
    out = tmp_path / "probe.jsonl"
    src = tmp_path / "sources.txt"
    src.write_text("\n".join(sorted(sources)) + "\n")
    rows_file = tmp_path / "select.jsonl"
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert probe_set.main(["--rows", str(rows_file), "--select", str(src), "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == len(rows)
