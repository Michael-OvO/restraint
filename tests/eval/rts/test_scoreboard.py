"""The RTS scoreboard: rows per model, one manifest and protocol per board, the delta command."""

from __future__ import annotations

import json

import pytest

from honest_agents.eval.records import write_records
from honest_agents.eval.rts.manifest import write_manifest
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.eval.rts.scoreboard import add_row, load_board, main, render_table, save_board
from tests.eval.rts.helpers import episodes, pairs

FAST = RtsProtocol(n_boot=200)


@pytest.fixture
def setup(tmp_path):
    items = pairs("tb2", "mid", "remove_tool", 12)
    manifest = tmp_path / "rts" / "v1" / "manifest.jsonl"
    write_manifest(items, manifest)
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
    assert board["rows"][1]["scores"]["headline"]["rts"]["value"] == pytest.approx((10 - 1) / 12)
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
