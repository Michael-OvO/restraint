"""Host-memory accounting for training runs.

A real run was killed by the kernel's OOM killer after about 19 steps: the training process (trainer + colocated vLLM)
grew to 69 GB of the node's 73 GB at about 3.8 GB per step. `rss_gb` is logged with every training step
(`callbacks.JsonlLogger` adds it as `mem/rss_gb`) so the growth is visible, and `train.health` warns before RAM runs out.

With `HONEST_MEM_DEBUG=1`, `MemoryDebug` also writes `<output_dir>/mem_debug.jsonl` after each step: the resident size,
the Python heap traced by `tracemalloc` (top source lines by growth) and the object types whose counts grew the most,
which separates a Python-level leak (lists of episodes, tool outputs) from native growth (torch, vLLM, malloc arenas).
"""

from __future__ import annotations

import gc
import json
import os
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

MEM_DEBUG_ENV = "HONEST_MEM_DEBUG"
MEM_DEBUG_FILE = "mem_debug.jsonl"
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_gb(pid: int | str = "self") -> float | None:
    """Resident set size of a process in GB (Linux `/proc/<pid>/statm`); None where it is not available."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE / 1e9
    except (OSError, ValueError, IndexError):
        return None


def mem_available_gb() -> float | None:
    """`MemAvailable` of the host in GB, or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError):
        pass
    return None


def mem_debug_enabled() -> bool:
    return os.environ.get(MEM_DEBUG_ENV, "") not in ("", "0", "false")


class MemoryDebug(TrainerCallback):
    """Per-step memory attribution (see the module docstring). Costs tracemalloc's overhead: diagnostic runs only."""

    def __init__(self, output_dir: str | Path, top: int = 12):
        self.path = Path(output_dir) / MEM_DEBUG_FILE
        self.top = top
        self._snapshot: tracemalloc.Snapshot | None = None
        self._types: Counter[str] | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not tracemalloc.is_tracing():
            tracemalloc.start(1)
        self._snapshot = tracemalloc.take_snapshot()
        self._types = self._type_counts()

    @staticmethod
    def _type_counts() -> Counter[str]:
        return Counter(type(o).__name__ for o in gc.get_objects())

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        snap = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        types = self._type_counts()
        row: dict[str, Any] = {
            "step": state.global_step,
            "time": time.time(),
            "rss_gb": rss_gb(),
            "mem_available_gb": mem_available_gb(),
            "python_traced_gb": current / 1e9,
            "python_traced_peak_gb": peak / 1e9,
        }
        if self._snapshot is not None:
            diff = snap.compare_to(self._snapshot, "lineno")
            row["top_growth"] = [
                {"where": str(d.traceback[0]), "size_diff_mb": round(d.size_diff / 1e6, 2), "count_diff": d.count_diff}
                for d in diff[: self.top]
            ]
        if self._types is not None:
            grown = Counter({k: v - self._types.get(k, 0) for k, v in types.items()})
            row["type_growth"] = [(k, n) for k, n in grown.most_common(self.top) if n > 0]
        self._snapshot, self._types = snap, types
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
