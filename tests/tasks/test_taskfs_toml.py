"""task.toml bytes do not depend on which TOML writers are installed (DEF-45)."""

from __future__ import annotations

import sys
import types

import pytest

from honest_agents.tasks.taskfs import dumps_toml, read_task_toml, write_task_toml

TABLE = {
    "version": "1.0",
    "metadata": {"tags": ["echos", "x"], "honest": {"kind": "twin", "n": 3, "skip": None}},
    "agent": {"timeout_sec": 600.0},
}
# `toml` 0.10.2's bytes, the writer the committed builds (data/pools/*manifest.json) were made with
GOLDEN = (
    'version = "1.0"\n\n[metadata]\ntags = [ "echos", "x",]\n\n[agent]\ntimeout_sec = 600.0\n\n'
    '[metadata.honest]\nkind = "twin"\nn = 3\n'
)


def _fake_writer(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.dumps = lambda data: f"# written by {name}\n"  # type: ignore[attr-defined]
    return mod


def test_dumps_toml_gives_fixed_bytes():
    assert dumps_toml(TABLE) == GOLDEN


@pytest.mark.parametrize(
    "present",
    [(), ("tomli_w",), ("tomlkit",), ("tomli_w", "tomlkit")],
    ids=["toml-only", "tomli_w", "tomlkit", "both"],
)
def test_installed_writers_do_not_change_the_bytes(monkeypatch, present):
    # a stand-in for each other writer when present, and a failing import (None in sys.modules) when absent
    for name in ("tomli_w", "tomlkit"):
        monkeypatch.setitem(sys.modules, name, _fake_writer(name) if name in present else None)
    assert dumps_toml(TABLE) == GOLDEN


@pytest.mark.parametrize("name", ["tomli_w", "tomlkit"])
def test_real_other_writer_installed_gives_the_same_bytes(name):
    pytest.importorskip(name)  # e.g. a venv also synced with the openenv extra
    assert dumps_toml(TABLE) == GOLDEN


def test_write_task_toml_round_trips(tmp_path):
    write_task_toml(tmp_path, TABLE)
    assert (tmp_path / "task.toml").read_text() == GOLDEN
    assert read_task_toml(tmp_path)["metadata"]["honest"] == {"kind": "twin", "n": 3}
