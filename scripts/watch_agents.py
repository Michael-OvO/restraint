#!/usr/bin/env python3
"""Live view of Claude Code workflow agents (the VS Code extension does not show them).

    python3 scripts/watch_agents.py            # refresh every 5 s, newest workflow
    python3 scripts/watch_agents.py --once     # print once and exit
    python3 scripts/watch_agents.py --lines 6  # show the last 6 actions per active agent

Reads the session's `subagents/workflows/<run>/` logs, so it works while the workflow runs in the background.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

ROOT = Path.home() / ".claude" / "projects" / "-Users-michael-Documents-Github-restraint"
ACTIVE_SECONDS = 180


def newest_workflow_dir() -> Path | None:
    dirs = [Path(p) for p in glob.glob(str(ROOT / "*" / "subagents" / "workflows" / "wf_*")) if (Path(p) / "journal.jsonl").exists()]
    return max(dirs, key=lambda d: (d / "journal.jsonl").stat().st_mtime, default=None)


def last_actions(path: Path, n: int) -> list[str]:
    out: list[str] = []
    try:
        lines = path.read_text().splitlines()[-400:]
    except OSError:
        return out
    for line in lines:
        try:
            msg = json.loads(line).get("message", {})
        except json.JSONDecodeError:
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "tool_use":
                i = b.get("input", {})
                what = i.get("description") or i.get("file_path") or i.get("command") or ""
                out.append(f"{b.get('name')}: {str(what).replace(chr(10), ' ')[:100]}")
    return out[-n:]


def render(d: Path, n: int) -> str:
    started: dict[str, str] = {}
    done: set[str] = set()
    for line in (d / "journal.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "started":
            started[e["agentId"]] = e.get("label", "?")
        elif e.get("type") == "result":
            done.add(e.get("agentId"))
    now = time.time()
    rows = []
    for aid, label in started.items():
        f = d / f"agent-{aid}.jsonl"
        if not f.exists():
            continue
        age = now - f.stat().st_mtime
        steps = sum(1 for _ in f.open())
        state = "done" if aid in done else ("ACTIVE" if age < ACTIVE_SECONDS else ("waiting?" if age < 900 else "STALLED?"))
        rows.append((state, label, steps, age, f))
    # A resumed workflow re-runs failed agents under the same label: keep only the newest entry per label.
    newest: dict[str, tuple] = {}
    for r in rows:
        if r[1] not in newest or r[3] < newest[r[1]][3]:
            newest[r[1]] = r
    rows = list(newest.values())
    rows.sort(key=lambda r: (r[0] != "ACTIVE", r[1]))
    out = [f"{d.name}   {time.strftime('%H:%M:%S')}   active={sum(r[0] == 'ACTIVE' for r in rows)} done={sum(r[0] == 'done' for r in rows)}", ""]
    for state, label, steps, age, f in rows:
        out.append(f"{state:11} {label:24} steps={steps:<5} last activity {int(age)}s ago")
        if state == "ACTIVE":
            out.extend(f"      > {a}" for a in last_actions(f, n))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--lines", type=int, default=3)
    p.add_argument("--interval", type=float, default=5.0)
    a = p.parse_args()
    while True:
        d = newest_workflow_dir()
        text = render(d, a.lines) if d else "no workflow found"
        if a.once:
            print(text)
            return
        os.system("clear")
        print(text)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
