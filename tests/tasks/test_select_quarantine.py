"""The select quarantine (`data/select_quarantine.txt`, read by every select and surface eval's `--exclude-quarantined`)
is a gate verdict on the frozen select split: each entry names a select variant and, since DEF-43, the umask its oracle
failed under. The 2026-10-14 list was graded under the local daemon's umask 0000 (DEF-16/19) and quarantined 24 variants
that pass under the nodes' 0022."""

from __future__ import annotations

import json
from pathlib import Path

from honest_agents.bank import read_quarantine

DATA = Path(__file__).resolve().parents[2] / "data"


def test_every_select_quarantine_entry_is_a_select_variant_gated_under_the_nodes_umask():
    entries = read_quarantine(DATA / "select_quarantine.txt")
    select = {json.loads(ln)["variant_id"] for ln in (DATA / "select.jsonl").read_text().splitlines() if ln.strip()}
    assert entries, "the select quarantine is tracked and not empty (8 variants fail the oracle gate under 0022)"
    assert {e.variant_id for e in entries} <= select
    stale = [e.variant_id for e in entries if e.source == "validation" and "(umask 0022)" not in (e.detail or "")]
    assert not stale, f"{len(stale)} entries were not gated under umask 0022, e.g. {stale[:2]}"
