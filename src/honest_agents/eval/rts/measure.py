"""Seed measurements (Docker): quantities an operator cannot read statically and needs before it can build its pair.

`disk_quota` sizes its tmpfs from the space the oracle's output takes (`output_bytes`), `mem_limit` its memory limit
from the oracle's peak (`oracle_peak_mb`). Their applicability names the quantity in the target
(`candidates.NEEDS`); `pairs` logs such a verdict as stage `measure`; this step boots the seed, runs its oracle, reads
the quantity as root, and appends one row per (source task, operator) to `measurements.jsonl`; `pairs` run again then
puts the value into the target and builds the pair. One sandbox per measurement, `concurrency` at once; resumable
(rows whose seed is unchanged are kept; rows with an infrastructure error are retried).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from honest_agents.tasks.validate import OracleHarborEnv, _abort, _on_env, task_hash

log = logging.getLogger(__name__)

MIB = 1024 * 1024
# cgroup v2 first (`memory.peak`, kernel 5.19+), then v1.
_PEAK_CMD = "cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes"


def measure_command(quantity: str, target: Mapping[str, Any]) -> str:
    """The root shell command whose stdout (one integer) is `quantity`, read after the oracle has run."""
    if quantity == "output_bytes":
        # `du -k` (busybox has no -B): allocated space, not apparent size, as the tmpfs quota counts it.
        return f"du -sk -- {shlex.quote(str(target['dir']))} | cut -f1"
    if quantity == "oracle_peak_mb":
        return _PEAK_CMD
    raise ValueError(f"no measurement for {quantity!r}")


def parse_value(quantity: str, stdout: str) -> float:
    n = int(stdout.strip().split()[0])
    if quantity == "output_bytes":
        return float(n * 1024)
    if quantity == "oracle_peak_mb":
        return round(n / MIB, 1)
    raise ValueError(f"no measurement for {quantity!r}")


async def measure_seed(
    seed_dir: Path,
    quantity: str,
    target: Mapping[str, Any],
    *,
    env_factory: Callable[[], OracleHarborEnv] = OracleHarborEnv,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Boot the seed, run its oracle, read `quantity`. `{quantity: value}` on success, else `{"error": ...}` (and
    `infra: True` when the sandbox, not the seed, failed)."""
    env = None
    try:
        env = env_factory()
        env.reset(task_dir=str(seed_dir))
        await _on_env(env, env._facts())
        oracle = await _on_env(env, env._run_oracle(timeout_sec=timeout_sec, destructive=[]))
        res = await _on_env(env, env._env.exec(measure_command(quantity, target), user="root", timeout_sec=120))
    except Exception as e:  # noqa: BLE001 — infrastructure: retried on resume
        return {"error": f"{type(e).__name__}: {e}", "infra": True}
    finally:
        await _abort(env, seed_dir.name)
    out: dict[str, Any] = {"oracle_rc": oracle.get("oracle_rc")}
    if oracle.get("oracle_rc") != 0:
        return {
            **out,
            "error": f"the seed's oracle exited {oracle.get('oracle_rc')}: {oracle.get('oracle_tail', '')[-300:]}",
        }
    try:
        return {**out, quantity: parse_value(quantity, res.stdout or "")}
    except (ValueError, IndexError):
        return {**out, "error": f"could not read {quantity} (rc {res.return_code}): {(res.stdout or '')[-200:]!r}"}


def read_measurements(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Rows by (source task, operator), the last row winning (append-only file)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["source_task"], r["operator"])] = r
    return out


def pending(applicability_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The `measure`-stage verdicts of a `pairs` run: what is still to measure."""
    return [dict(r) for r in applicability_rows if r.get("stage") == "measure"]


async def measure_many(
    todo: Iterable[Mapping[str, Any]],
    out_path: Path,
    *,
    measure: Callable[[Path, str, Mapping[str, Any]], Any],
    concurrency: int = 2,
    resume: bool = True,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Measure every pending row (`pending`), appending each result to `out_path` as it completes."""
    from honest_agents.eval.rts.candidates import NEEDS

    rows = list(todo)
    done = read_measurements(out_path) if resume else {}
    if not resume:
        out_path.unlink(missing_ok=True)

    def current(r: Mapping[str, Any]) -> bool:
        d = done.get((r["source_task"], r["operator"]))
        return bool(d) and not d.get("infra") and d.get("seed_hash") == task_hash(Path(r["seed_dir"]))

    run = [r for r in rows if not current(r)]
    log.info("measure: %d pending, %d to run", len(rows), len(run))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()

    async def one(r: Mapping[str, Any]) -> None:
        quantity = str(r["target"][NEEDS])
        seed_dir = Path(r["seed_dir"])
        async with sem:
            t0 = time.monotonic()
            got = await measure(seed_dir, quantity, r["target"])
        res = {
            "source_task": r["source_task"],
            "operator": r["operator"],
            "benchmark": r.get("benchmark"),
            "quantity": quantity,
            "seed_dir": str(seed_dir),
            "seed_hash": task_hash(seed_dir),
            "duration_sec": round(time.monotonic() - t0, 2),
            **got,
        }
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(res, default=str) + "\n")
            done[(r["source_task"], r["operator"])] = res
            log.info("%s %s: %s", r["operator"], r["source_task"], got)

    await asyncio.gather(*(one(r) for r in run))
    return done
