"""A frozen RTS version is the one evaluated: the manifest, its task dirs, its protocol and (optionally) the prereg line.

  python -m honest_agents.eval.rts.verify --manifest data/rts/v1/manifest.jsonl [--prereg docs/iteration-16-prereg.md]

The manifest's sidecar digest covers the manifest file only. The episodes run on `<version>/tasks/<item id>/`, so each
item's dir must still hash (`tasks.validate.task_hash`) to the `task_dir_sha256` the freeze recorded, and no other dir
may sit under `tasks/` (an aborted freeze or a re-sync from another checkout would leave one, and the eval would run
it). `protocol.json` must be the pre-registered defaults (`protocol.PREREGISTERED`: the freeze refuses any other), and with
`--prereg` the prereg must hold the line `rts/<version> manifest sha256: <digest>`. Exit 0 when all hold, else 1 with
every problem listed. The chain steps `rts_freeze_check` (stages (c) and (d)) and `scoreboard add` run it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from honest_agents.eval.rts import protocol as protocol_mod
from honest_agents.eval.rts.manifest import load_manifest
from honest_agents.eval.rts.protocol import RtsProtocol

PROTOCOL_NAME = "protocol.json"
TASKS_NAME = "tasks"


def prereg_line(manifest: Path, digest: str) -> str:
    return f"rts/{manifest.parent.name} manifest sha256: {digest}"


def read_protocol(manifest: Path) -> RtsProtocol:
    """The protocol the freeze wrote next to the manifest."""
    path = Path(manifest).parent / PROTOCOL_NAME
    if not path.is_file():
        raise ValueError(f"{path}: missing; a frozen version carries its protocol")
    return RtsProtocol.from_json(json.loads(path.read_text()))


def frozen_problems(
    manifest: str | Path, *, prereg: str | Path | None = None, require_default_protocol: bool = True
) -> list[str]:
    """Why `manifest` is not the frozen version as written (empty when it is)."""
    from honest_agents.tasks.validate import task_hash

    manifest = Path(manifest)
    try:
        items = load_manifest(manifest)
    except (OSError, ValueError) as e:
        return [str(e)]
    digest = (manifest.with_name(manifest.name + ".sha256")).read_text().strip()
    problems: list[str] = []
    tasks = manifest.parent / TASKS_NAME
    ids = {i.item_id for i in items}
    for i in items:
        d = tasks / i.item_id
        if not i.task_dir_sha256:
            problems.append(f"{i.item_id}: no task_dir_sha256 in the manifest (not written by the freeze)")
        elif not d.is_dir():
            problems.append(f"{i.item_id}: no task dir {d}")
        elif (got := task_hash(d)) != i.task_dir_sha256:
            problems.append(f"{i.item_id}: task dir hashes to {got[:12]}, frozen as {i.task_dir_sha256[:12]}")
    if tasks.is_dir():
        extra = sorted(p.name for p in tasks.iterdir() if p.name not in ids)
        if extra:
            problems.append(f"{tasks}: {len(extra)} entries not in the manifest ({', '.join(extra[:5])})")
    try:
        proto = read_protocol(manifest)
        pre = protocol_mod.PREREGISTERED
        if require_default_protocol and proto != pre:
            problems.append(f"{PROTOCOL_NAME}: {proto.to_json()} is not the pre-registered {pre.to_json()}")
    except (OSError, ValueError, TypeError) as e:
        problems.append(f"{PROTOCOL_NAME}: {e}")
    if prereg is not None:
        want = prereg_line(manifest, digest)
        lines = Path(prereg).read_text().splitlines() if Path(prereg).is_file() else []
        if want not in lines:
            problems.append(f"{prereg}: no line {want!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--prereg", help="the pre-registration that must record the manifest's digest")
    args = p.parse_args(argv)
    problems = frozen_problems(args.manifest, prereg=args.prereg)
    for line in problems:
        print(f"rts verify: {line}", file=sys.stderr)
    if not problems:
        print(f"{args.manifest}: frozen as written (manifest, task dirs, protocol{', prereg' if args.prereg else ''})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
