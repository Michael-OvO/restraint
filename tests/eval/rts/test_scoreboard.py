"""The RTS scoreboard: rows per model, one manifest and protocol per board, the delta command."""

from __future__ import annotations

import json

import pytest

from honest_agents.eval.records import write_records
from honest_agents.eval.rts.manifest import write_manifest
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.eval.rts.scoreboard import add_row, load_board, main, render_table, save_board
from honest_agents.eval.run_eval import reward_config_of_run
from tests.eval.rts.helpers import episodes, pairs

FAST = RtsProtocol(n_boot=200)


def freeze_fixture(items, manifest, proto):
    """What `freeze` writes: a task dir per item, hashed into the manifest, and the protocol next to it."""
    from dataclasses import replace

    from honest_agents.tasks.validate import task_hash

    hashed = []
    for i in items:
        i = replace(i, item_id=i.item_id.replace("/", "__").replace(":", "_"))  # path-safe, as the freeze's are
        d = manifest.parent / "tasks" / i.item_id
        d.mkdir(parents=True)
        (d / "instruction.md").write_text(f"{i.item_id}\n")
        hashed.append(replace(i, task_dir_sha256=task_hash(d)))
    write_manifest(hashed, manifest)
    (manifest.parent / "protocol.json").write_text(json.dumps(proto.to_json()))
    return hashed


@pytest.fixture
def setup(tmp_path):
    items = [*pairs("tb2", "near", "op_a", 4), *pairs("swe", "mid", "op_b", 4), *pairs("quix", "far", "op_c", 4)]
    manifest = tmp_path / "rts" / "v1" / "manifest.jsonl"
    items = freeze_fixture(items, manifest, FAST)
    runs = {}
    for name, hits in (("base", 2), ("trained", 10)):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        write_records(run / "episodes.jsonl", episodes(items, hits=hits, false_refusals=1))
        runs[name] = run
    return tmp_path, manifest, runs


def test_add_rows_sorted_and_replaced(setup):
    tmp, manifest, runs = setup
    board = load_board(tmp / "board.json")
    add_row(
        board, label="trained", order=1, run_dir=runs["trained"], manifest=manifest, units_dir=tmp / "u", proto=FAST
    )
    add_row(board, label="base", order=0, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST)
    add_row(board, label="base", order=0, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST)
    assert [r["label"] for r in board["rows"]] == ["base", "trained"]
    assert board["rows"][1]["scores"]["headline"]["rts"]["value"] == pytest.approx((3 / 4 + 4 / 4 + 2 / 4) / 3)
    assert board["rows"][0]["coverage"] == pytest.approx(1.0)
    save_board(board, tmp / "board.json")
    assert load_board(tmp / "board.json") == json.loads(json.dumps(board))
    table = render_table(board)
    assert "| 0 | base |" in table and "| 1 | trained |" in table


def test_board_refuses_another_manifest_or_protocol(setup):
    tmp, manifest, runs = setup
    board = load_board(tmp / "board.json")
    add_row(board, label="base", order=0, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST)
    other = tmp / "rts" / "v2" / "manifest.jsonl"
    write_manifest(pairs("tb2", "mid", "remove_tool", 13), other)
    with pytest.raises(ValueError, match="manifest"):
        add_row(board, label="x", order=2, run_dir=runs["base"], manifest=other, units_dir=tmp / "u", proto=FAST)
    with pytest.raises(ValueError, match="protocol"):
        add_row(
            board,
            label="x",
            order=2,
            run_dir=runs["base"],
            manifest=manifest,
            units_dir=tmp / "u",
            proto=RtsProtocol(n_boot=300),
        )


def test_run_without_episodes_is_refused(setup):
    tmp, manifest, _ = setup
    (tmp / "empty").mkdir()
    with pytest.raises(ValueError, match="no episodes"):
        add_row(
            load_board(tmp / "b.json"),
            label="x",
            order=0,
            run_dir=tmp / "empty",
            manifest=manifest,
            units_dir=tmp / "u",
            proto=FAST,
        )


def test_cli_add_show_delta(setup, capsys):
    tmp, manifest, runs = setup
    board = str(tmp / "board.json")
    common = ["--board", board, "--units-dir", str(tmp / "u"), "--n-boot", "200"]
    for order, name in enumerate(("base", "trained")):
        args = ["add", "--run", str(runs[name]), "--manifest", str(manifest), "--label", name, "--order", str(order)]
        assert main(args + common) == 0
    assert main(["show", "--board", board]) == 0
    assert "| 1 | trained |" in capsys.readouterr().out
    assert (
        main(["delta", "--board", board, "--base", "base", "--trained", "trained", "--out", str(tmp / "d.json")]) == 0
    )
    out = capsys.readouterr().out
    assert "verdict:" in out and "Generalisation ladder" in out
    assert json.loads((tmp / "d.json").read_text())["benchmark"] == "rts"


def test_run_config_json_reward_is_used(setup):
    tmp, manifest, runs = setup
    assert reward_config_of_run(runs["base"]).alpha != 0.4
    (runs["base"] / "config.json").write_text(json.dumps({"reward": {"alpha": 0.4, "not_a_field": 1}}))
    assert reward_config_of_run(runs["base"]).alpha == 0.4
    row = add_row(
        load_board(tmp / "b.json"),
        label="x",
        order=0,
        run_dir=runs["base"],
        manifest=manifest,
        units_dir=tmp / "u",
        proto=FAST,
    )
    assert row["coverage"] == pytest.approx(1.0)
    assert row["reward_config"]["alpha"] == 0.4


def test_failed_scoring_leaves_no_units_file(setup, monkeypatch):
    from honest_agents.eval.rts import scoreboard

    tmp, manifest, runs = setup

    def boom(*a, **k):
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(scoreboard, "score_model", boom)
    with pytest.raises(RuntimeError):
        add_row(
            load_board(tmp / "b.json"),
            label="x",
            order=0,
            run_dir=runs["base"],
            manifest=manifest,
            units_dir=tmp / "u",
            proto=FAST,
        )
    assert not (tmp / "u").exists() or not list((tmp / "u").iterdir())


def test_labels_sharing_a_units_file_are_refused(setup):
    tmp, manifest, runs = setup
    board = load_board(tmp / "board.json")
    add_row(board, label="a b", order=0, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST)
    with pytest.raises(ValueError, match="units file"):
        add_row(board, label="a-b", order=1, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST)


def test_delta_with_missing_units_or_empty_board_exits_cleanly(setup):
    tmp, manifest, runs = setup
    board = str(tmp / "board.json")
    common = ["--board", board, "--units-dir", str(tmp / "u"), "--n-boot", "200"]
    for order, name in enumerate(("base", "trained")):
        args = ["add", "--run", str(runs[name]), "--manifest", str(manifest), "--label", name, "--order", str(order)]
        assert main(args + common) == 0
    (tmp / "u" / "base.json").unlink()
    with pytest.raises(SystemExit, match="base.json"):
        main(["delta", "--board", board, "--base", "base", "--trained", "trained"])
    with pytest.raises(SystemExit, match="empty"):
        main(["delta", "--board", str(tmp / "none.json"), "--base", "a", "--trained", "b"])


def test_board_row_with_a_blank_headline_renders(setup):
    from honest_agents.eval.rts.plots import render_html

    tmp, manifest, runs = setup
    board = load_board(tmp / "board.json")
    row = add_row(
        board, label="base", order=0, run_dir=runs["base"], manifest=manifest, units_dir=tmp / "u", proto=FAST
    )
    for name in ("rts", "rts_strict"):
        row["scores"]["headline"][name].update(value=None, lo=None, hi=None)
    assert "| 0 | base | - |" in render_table(board)
    assert "base" in render_html(board)


def test_cli_add_scores_only_the_frozen_version_under_its_protocol(setup):
    tmp, manifest, runs = setup
    board = str(tmp / "board.json")
    base = ["add", "--run", str(runs["base"]), "--manifest", str(manifest), "--label", "base", "--order", "0"]
    common = ["--board", board, "--units-dir", str(tmp / "u")]
    with pytest.raises(SystemExit, match="differs from the frozen protocol"):
        main([*base, *common, "--n-boot", "300"])
    task_dir = manifest.parent / "tasks"
    item = min(task_dir.iterdir())
    (item / "instruction.md").write_text("edited after the freeze\n")
    with pytest.raises(SystemExit, match="task dir hashes to"):
        main([*base, *common])
    (item / "instruction.md").write_text(f"{item.name}\n")
    (task_dir / "stray").mkdir()  # an aborted freeze's leftover would be run by the eval too
    with pytest.raises(SystemExit, match="not in the manifest"):
        main([*base, *common])
    (task_dir / "stray").rmdir()
    (manifest.parent / "protocol.json").unlink()
    with pytest.raises(SystemExit, match="protocol.json"):
        main([*base, *common])
    assert not (tmp / "board.json").exists()
