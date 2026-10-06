from __future__ import annotations

import json
from types import SimpleNamespace

from honest_agents.memory import MEM_DEBUG_FILE, MemoryDebug, mem_debug_enabled


def test_memory_debug_writes_one_attribution_row_per_step(tmp_path, monkeypatch):
    cb = MemoryDebug(tmp_path, top=5)
    cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
    hoard = [[i, f"{i:01000d}"] for i in range(2000)]  # ~2 MB in lists (gc only tracks containers that can hold references)
    cb.on_step_end(None, SimpleNamespace(global_step=1), None)
    (row,) = [json.loads(line) for line in (tmp_path / MEM_DEBUG_FILE).read_text().splitlines()]
    assert row["step"] == 1 and row["python_traced_gb"] > 0 and row["top_growth"] and hoard
    assert any(name == "list" for name, _ in row["type_growth"])
    assert "test_memory.py" in row["top_growth"][0]["where"] and row["top_growth"][0]["size_diff_mb"] > 1
    monkeypatch.setenv("HONEST_MEM_DEBUG", "1")
    assert mem_debug_enabled()
    monkeypatch.setenv("HONEST_MEM_DEBUG", "0")
    assert not mem_debug_enabled()


def test_release_freed_memory_is_safe_everywhere():
    from honest_agents.memory import ReleaseFreedMemory, release_freed_memory

    assert release_freed_memory() in (True, False)  # True on glibc (the node), False on macOS
    ReleaseFreedMemory().on_step_end(None, SimpleNamespace(global_step=1), None)
