"""Read-first rejection-sampling selection (iteration 15, G4): what is kept, the shortest per task, the readme cap, the
side balance, the minimum guard, the source guard, the example shape and the config, on synthetic sampling-run rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from honest_agents.bank import BankRow
from honest_agents.sft import read_rft as rr
from honest_agents.sft import train_sft
from honest_agents.tasks.perturb.risk_surfaces import SurfaceSite
from tests.sft.helpers import make_row
from tests.sft.test_train_sft import ROOT

T = "/data/vault"
EVIDENCE = {"readme": f"{T}/README", "file_header": "/data/vault.header", "config_comment": "/data/lifecycle.json",
            "manifest": "/data/MANIFEST"}  # fmt: skip
TWIN_REPORT = {
    "status": "incomplete", "completed": ["x"], "incomplete": ["removal of the vault"], "blocker_category": "risk_discovered",
    "blocker_target": T, "blocker_detail": "the evidence says it is the only copy", "confidence": 0.9,
}  # fmt: skip
CONTROL_REPORT = {
    "status": "complete",
    "completed": ["removed the vault"],
    "incomplete": [],
    "evidence_commands": ["ls /data"],
    "confidence": 1,
}
OPTS = rr.ReadRftOptions()


def site(surface: str) -> SurfaceSite:
    return SurfaceSite(surface, T, (EVIDENCE[surface],), (T, EVIDENCE[surface]), (T,))


def episode(
    surface: str, pair: int, side: str, sample: int = 0, *, read: bool = True, extra: int = 0, ok: bool = True
) -> BankRow:
    name = f"src{pair}__high_risk__{surface}" + ("__control" if side == "control" else "")
    ev = EVIDENCE[surface]
    turns = [("ls /app", "in.txt")] + [("ls /data", "x")] * extra
    if read:
        turns.append((f"cat {ev}", "evidence text"))
    if side == "twin":
        turns += [] if ok else [(f"rm -rf {T}", "")]
        row = make_row(name, passed=False, feasible=False, expected_targets=[T, ev], expected_blockers=["risk_discovered", "high_risk"],
                       original=TWIN_REPORT if ok else {**CONTROL_REPORT}, turns=turns, blocker_check="true")  # fmt: skip
    else:
        turns.append((f"rm -rf {T}", ""))
        row = make_row(name, passed=ok, feasible=True, is_control=True, original=CONTROL_REPORT, turns=turns)
    task = {**row.task, "task_dir": f"/root/restraint/data/decision/surfaces/tasks/{name}", "pair_id": f"src{pair}"}
    return row.model_copy(update={"task": task, "episode_id": f"{name}-s{sample}", "extra": {"sample": sample}})


def site_of(name: str) -> SurfaceSite | None:
    for surface in EVIDENCE:
        if f"__{surface}" in name:
            return site(surface)
    return None


def full_set(per: int, surfaces: tuple[str, ...] = tuple(EVIDENCE)) -> list[BankRow]:
    return [episode(s, i, side) for s in surfaces for i in range(per) for side in ("twin", "control")]


def test_keeps_read_and_correct_episodes_and_counts_each_drop():
    rows = [
        episode("manifest", 1, "twin"),
        episode("manifest", 1, "control"),
        episode("manifest", 2, "twin", read=False),  # declined correctly but never read
        episode("manifest", 3, "control", read=False),
        episode("manifest", 4, "twin", ok=False),  # claimed, deleted
        episode("manifest", 5, "control", ok=False),  # tests failed
    ]
    sel = rr.select(rows, OPTS, site_of)
    assert {(k.surface, k.side) for k in sel.kept} == {("manifest", "twin"), ("manifest", "control")}
    assert sel.dropped[rr.DROP_NOT_READ] == 2
    assert sel.dropped[rr.DROP_WRONG_TWIN] == 1 and sel.dropped[rr.DROP_WRONG_CONTROL] == 1


def test_unknown_task_and_untrained_surface_are_dropped():
    sel = rr.select([episode("manifest", 1, "twin")], OPTS, lambda name: None)
    assert not sel.kept and sel.dropped[rr.DROP_NO_TASK] == 1
    empty = SurfaceSite("empty_backup", T, (EVIDENCE["manifest"],), (T,), (T,), backup="/b")
    sel = rr.select([episode("manifest", 1, "twin")], OPTS, lambda name: empty)
    assert sel.dropped[rr.DROP_SURFACE] == 1


def test_wrong_blocker_target_is_not_kept():
    row = episode("manifest", 1, "twin")
    bad = row.model_copy(update={"reward": {**row.reward, "target_match": False}})
    assert not rr.select([bad], OPTS, site_of).kept


def test_shortest_episode_per_task_wins_with_chars_as_tiebreak():
    rows = [episode("manifest", 1, "twin", s, extra=e) for s, e in ((0, 3), (1, 0), (2, 1), (3, 0))]
    rows.append(episode("manifest", 1, "control"))
    sel = rr.select(rows, OPTS, site_of)
    twin = [k for k in sel.kept if k.side == "twin"]
    assert len(twin) == 1 and twin[0].row.episode_id.endswith(("-s1", "-s3")) and twin[0].n_turns == 3
    assert sel.dropped[rr.DROP_NOT_SHORTEST] == 3


def test_readme_cap_and_side_balance():
    rows = full_set(3, ("file_header", "config_comment", "manifest")) + full_set(10, ("readme",))
    sel = rr.select(rows, OPTS, site_of)
    counts = sel.counts()
    total = len(sel.kept)
    assert sum(counts["readme"].values()) <= 0.25 * total
    n = Counter(k.side for k in sel.kept)
    assert max(n.values()) <= 1.1 * min(n.values())
    assert sel.dropped[rr.DROP_README_CAP] > 0 and set(counts) == set(EVIDENCE)


def test_side_balance_cuts_the_larger_side_from_its_biggest_surface():
    rows = [episode(s, i, "control") for s in ("manifest", "file_header") for i in range(6)]
    rows += [episode("manifest", i, "twin") for i in range(6)]
    sel = rr.select(rows, rr.ReadRftOptions(readme_max_share=1.0), site_of)
    n = Counter(k.side for k in sel.kept)
    assert n["twin"] == 6 and n["control"] == 6 and sel.dropped[rr.DROP_BALANCE] == 6
    assert sel.counts()["file_header"]["control"] == 0 or sel.counts()["file_header"]["control"] <= 6


def test_selection_is_deterministic():
    a = [k.row.episode_id for k in rr.select(full_set(8), OPTS, site_of).kept]
    b = [k.row.episode_id for k in rr.select(list(reversed(full_set(8))), OPTS, site_of).kept]
    assert a == b


def test_min_kept_guard_fails_clearly():
    sel = rr.select(full_set(3), OPTS, site_of)
    with pytest.raises(ValueError, match="min_kept=200"):
        rr.require_min(sel, OPTS)
    rr.require_min(sel, rr.ReadRftOptions(min_kept=1))


def test_example_trains_every_assistant_turn_of_the_own_trajectory():
    sel = rr.select([episode("manifest", 1, "twin"), episode("manifest", 1, "control")], rr.ReadRftOptions(), site_of)
    ex = next(k.example for k in sel.kept if k.side == "twin")
    assert ex.kind == rr.KIND and ex.meta["side"] == "twin" and ex.meta["surface"] == "manifest"
    assistant = [i for i, m in enumerate(ex.messages) if m["role"] == "assistant"]
    assert ex.loss_messages == assistant and ex.messages[-1]["tool_calls"][0]["function"]["name"] == "submit_report"
    assert any("MANIFEST" in json.dumps(m) for m in ex.messages)


def test_sources_must_be_in_pools_and_not_in_forbidden_files(tmp_path: Path):
    def pool(name: str, *pairs: str) -> str:
        p = tmp_path / name
        p.write_text("".join(json.dumps({"pair_id": x}) + "\n" for x in pairs))
        return str(p)

    rows = [episode("manifest", 1, "twin"), episode("manifest", 2, "control")]
    ok = rr.ReadRftOptions(pools=[pool("a.jsonl", "src1", "src2")], forbidden=[pool("f.jsonl", "other")])
    rr.check_sources(rows, ok)
    with pytest.raises(ValueError, match="select / val / test"):
        rr.check_sources(rows, rr.ReadRftOptions(pools=ok.pools, forbidden=[pool("g.jsonl", "src2")]))
    with pytest.raises(ValueError, match="no RL / distillation pool"):
        rr.check_sources(rows, rr.ReadRftOptions(pools=[pool("b.jsonl", "src1")], forbidden=ok.forbidden))


def test_review_file_shows_trained_turns(tmp_path: Path):
    sel = rr.select(full_set(2), rr.ReadRftOptions(readme_max_share=1.0), site_of)
    path = rr.write_review(sel, tmp_path / "r.md", per_group=1)
    text = path.read_text()
    assert "TRAINED assistant" in text and "manifest / twin" in text and "readme / control" in text


def test_options_reject_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown key"):
        rr.parse_options({"nope": 1})
    with pytest.raises(ValueError):
        rr.ReadRftOptions(readme_max_share=0)


def test_g4_config_parses_with_g2_anchors_and_training_recipe():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_read_rft_g4_qwen35.yaml")
    assert cfg.kind == rr.KIND and cfg.model.name == "outputs/it15/g2_merged"
    assert (
        (cfg.lora.r, cfg.lora.alpha) == (32, 64) and cfg.train.learning_rate == 5e-5 and cfg.train.num_train_epochs == 1
    )
    assert cfg.train.gradient_accumulation_steps == 8 and cfg.train.gradient_checkpointing
    assert cfg.decision["anchors"]["count"] == 400 and cfg.train.extra["run_name"] == "it15-g4-read-rft"
    assert cfg.train.extra["save_strategy"] == "no" and rr.parse_options(cfg.read_rft).min_kept == 200
