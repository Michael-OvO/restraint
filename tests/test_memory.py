from __future__ import annotations

import json
from types import SimpleNamespace

from honest_agents.memory import MEM_DEBUG_FILE, MemoryDebug, mem_debug_enabled


def test_memory_debug_writes_one_attribution_row_per_step(tmp_path, monkeypatch):
    cb = MemoryDebug(tmp_path, top=5)
    cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
    hoard = [bytearray(1000) for _ in range(2000)]  # ~2 MB that the diff must show
    cb.on_step_end(None, SimpleNamespace(global_step=1), None)
    (row,) = [json.loads(line) for line in (tmp_path / MEM_DEBUG_FILE).read_text().splitlines()]
    assert row["step"] == 1 and row["python_traced_gb"] > 0 and row["top_growth"] and hoard
    assert any(name == "bytearray" for name, _ in row["type_growth"])
    monkeypatch.setenv("HONEST_MEM_DEBUG", "1")
    assert mem_debug_enabled()
    monkeypatch.setenv("HONEST_MEM_DEBUG", "0")
    assert not mem_debug_enabled()
