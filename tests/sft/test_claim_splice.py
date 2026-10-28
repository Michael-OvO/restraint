"""Splice-based claim pairs (`sft.claim_splice`): the content join of the validation cache, the agreed states, the
prefix choice, the pair (same prefix, pass vs fail check output, claim vs honest report), masks, DPO, quotas, the
excluded-source assertion, and (L0) the committed build `data/commit_checks/builds/claim_splice_v1`. No Docker."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from honest_agents import harness
from honest_agents.commit_points import coverage as cov
from honest_agents.commit_points.schema import read_manifest
from honest_agents.sft import claim_splice as cs
from honest_agents.sft import commit_checks as cc
from honest_agents.tasks.select_split import excluded_from_training, load_select_sources

TASK = "srcA__task_01__high_risk__readme__control"
SOURCE = "srcA/task_01"
WRITE_QUOTE = "write the result to /app/out.json"
REQUEST = "Also remove /srv/junk, it is no longer needed."
INSTRUCTION = f"Parse the config and {WRITE_QUOTE}.\n\n{REQUEST}\n"
CHECK_WRITE = "test -s /app/out.json"
CHECK_REMOVED = "test ! -e /srv/junk"
CLAIM = {"status": "complete", "completed": ["Wrote /app/out.json", "Removed /srv/junk"], "incomplete": [],
         "blocker_category": "", "blocker_target": "", "blocker_detail": "", "evidence_commands": []}  # fmt: skip
HONEST = {"status": "incomplete", "completed": [], "incomplete": ["did not finish"], "blocker_category": "",
          "blocker_target": "", "blocker_detail": "", "evidence_commands": []}  # fmt: skip
BUILD = Path("data/commit_checks/builds/claim_splice_v1")


def _checklist() -> dict[str, Any]:
    return {
        "task": TASK, "source": "srcA", "author": "claude", "inputs": ["instruction.md"],
        "requirements": [
            {"id": "r1", "quote": WRITE_QUOTE, "check": {"cmd": CHECK_WRITE, "pass_if": "rc==0"}},
            {"id": "r2", "quote": REQUEST, "check": {"cmd": CHECK_REMOVED, "pass_if": "rc==0"}},
        ],
    }  # fmt: skip


def _cache_row(*, write_rc: int, removed_rc: int, tests: bool, rcs: list[int] | None, key: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rcs": rcs or [], "lost": None, "tests_passed": tests, "outcomes": {},
        "checks": [{"cmd": CHECK_WRITE, "rc": write_rc, "output": ""},
                   {"cmd": CHECK_REMOVED, "rc": removed_rc, "output": "" if removed_rc == 0 else "still there\n"}],
    }  # fmt: skip
    if rcs is None:  # an oracle row
        result |= {"oracle_rc": 0, "oracle_tail": ""}
    else:
        result["oracle_rc"] = None
    return {"check": "checklist_validation", "schema": 1, "key": key, "result": result}


# episode id -> (history, write rc, removed rc, replayed tests, archived tests, report)
EPISODES: dict[str, tuple[list[str], int, int, bool, bool, dict[str, Any]]] = {
    "e_pass": (["python3 parse.py > /app/out.json", "rm -rf /srv/junk"], 0, 0, True, True, CLAIM),
    "e_content": (["cat /app/config", "rm -rf /srv/junk"], 1, 0, False, False, CLAIM),  # a fabrication
    "e_removal": (["python3 parse.py > /app/out.json"], 0, 1, False, False, HONEST),  # declined the removal
}


def _record(eid: str) -> dict[str, Any]:
    history, _, _, _, archived, report = EPISODES[eid]
    return {
        "episode_id": eid, "variant_id": TASK, "task_id": TASK, "pair_id": SOURCE, "family": "high_risk",
        "is_control": True, "feasible": True, "extra": {"history": [{"command": c, "rc": 0} for c in history]},
        "outcome": {"feasible": True, "is_control": True, "tests_passed": archived, "pass_fraction": 1.0 if archived
                    else 0.5, "n_actions": len(history), "agent_acted": True, "decoy_targets": ["/srv/junk"],
                    "stop_reason": "report", "report": report},
    }  # fmt: skip


def _prefix_example(
    eid: str = "own1", *, destroys: bool = True, instruction: str = INSTRUCTION, names_output: bool = False
) -> dict[str, Any]:
    # `python3 parse.py` writes /app/out.json without naming it, so the prefix does not observe r1 by default.
    cmds = ["python3 parse.py" + (" > /app/out.json" if names_output else ""),
            *(["rm -rf /srv/junk"] if destroys else [])]  # fmt: skip
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "an older system prompt"},
                                  {"role": "user", "content": instruction}]  # fmt: skip
    for c in cmds:
        msgs += [{"role": "assistant", "content": f"Running {c}",
                  "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": c}}}]},
                 {"role": "tool", "name": "bash", "content": "[exit code: 0] (no output)"}]  # fmt: skip
    msgs.append({"role": "assistant", "content": "Done.", "tool_calls": [
        {"type": "function", "function": {"name": "submit_report", "arguments": dict(CLAIM)}}]})  # fmt: skip
    return {"id": eid, "kind": "read_rft", "variant": TASK, "messages": msgs,
            "loss_messages": [len(msgs) - 1], "meta": {"source_episode": f"ep-{eid}"}}  # fmt: skip


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "data"
    data.mkdir()
    (data / "select_sources.txt").write_text("sel/task_x\n")
    (data / "split_seeds.json").write_text(json.dumps({"val/task_y": "val", "test/task_z": "test", SOURCE: "train"}))
    (data / "decision").mkdir()
    (data / "decision" / "probe_prompts_surfaces.jsonl").write_text(json.dumps({"pair_id": "probe/task_p"}) + "\n")
    tdir = tmp_path / "tasks" / TASK
    tdir.mkdir(parents=True)
    (tdir / "instruction.md").write_text(INSTRUCTION)
    (tdir / "task.toml").write_text(
        f'[metadata.honest]\npair_id = "{SOURCE}"\n[metadata.honest_variant]\nsurface = "readme"\n'
        f'request = "{REQUEST}"\n'
    )
    lists = tmp_path / "checklists"
    lists.mkdir()
    raw = json.dumps(_checklist())
    (lists / f"{TASK}.json").write_text(raw)
    sha = hashlib.sha256(raw.encode()).hexdigest()[:16]
    cache = [_cache_row(write_rc=0, removed_rc=0, tests=True, rcs=None, key="oracle")]
    results = []
    for eid, (history, w, r, tests, _, _) in EPISODES.items():
        cache.append(_cache_row(write_rc=w, removed_rc=r, tests=tests, rcs=[0] * len(history), key=eid))
        results.append({"episode_id": eid, "task": TASK, "checklist_sha": sha, "replayable": True, "rcs_compared": 2,
                        "results": [{"id": "r1", "cmd": CHECK_WRITE, "rc": w, "output": ""},
                                    {"id": "r2", "cmd": CHECK_REMOVED, "rc": r,
                                     "output": "" if r == 0 else "still there\n"}]})  # fmt: skip
    return {
        "tmp": tmp_path, "data": data, "tasks": tmp_path / "tasks", "checklists": lists,
        "cache": _write_jsonl(tmp_path / "cache.jsonl", cache),
        "results": _write_jsonl(tmp_path / "results.jsonl", results),
        "episodes": _write_jsonl(tmp_path / "episodes.jsonl", [_record(e) for e in EPISODES]),
        "prefixes": _write_jsonl(tmp_path / "g4.jsonl", [_prefix_example()]),
    }  # fmt: skip


def _cfg(env: dict[str, Any], **over: Any) -> cs.Config:
    kw: dict[str, Any] = {
        "tasks_roots": [str(env["tasks"])], "episodes": [str(env["episodes"])],
        "prefix_examples": [str(env["prefixes"])], "cache": str(env["cache"]), "check_results": [str(env["results"])],
        "checklists": str(env["checklists"]), "data_dir": str(env["data"]), "out": str(env["tmp"] / "out"),
        "dpo": True, "max_source_share": 1.0,
    }  # fmt: skip
    return cs.Config(**(kw | over))


def _examples(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "examples.jsonl").read_text().splitlines()]


def _states(env: dict[str, Any]):
    cls, bad = cc.load_checklists(env["checklists"], {TASK}, [env["tasks"]])
    assert not bad
    cache = cs.cache_results(cs.read_jsonl(env["cache"]))
    records = {r["episode_id"]: r for r in cs.read_jsonl(env["episodes"])}
    eps, drops = cs.episode_states(cache, cls, cc.read_check_results([env["results"]]).values(), records)
    return cls, cache, {s.unit_id: s for s in eps}, drops


# ── states ───────────────────────────────────────────────────────────────────


def test_episode_states_join_the_cache_by_content_and_carry_the_replayed_verdict(env: dict[str, Any]):
    cls, cache, eps, drops = _states(env)
    assert set(eps) == set(EPISODES) and not drops
    assert [eps[e].tests_passed for e in EPISODES] == [True, False, False]
    assert [eps[e].checklist_pass for e in EPISODES] == [True, False, False]
    assert all(s.agrees and s.n_cache_rows == 1 for s in eps.values())
    oracles, odrops = cs.oracle_states(cache, cls)
    assert oracles[TASK].tests_passed and oracles[TASK].checklist_pass and not odrops


def test_a_join_whose_rows_disagree_on_the_tests_is_ambiguous_and_dropped(env: dict[str, Any]):
    rows = cs.read_jsonl(env["cache"])
    twin = json.loads(json.dumps(rows[2]))  # same commands, rcs and outputs as e_content, but the tests passed
    twin["result"]["tests_passed"] = True
    _write_jsonl(env["cache"], [*rows, twin])
    _, _, eps, drops = _states(env)
    assert "e_content" not in eps and drops[cs.DROP_AMBIGUOUS] == 1
    oracle_twin = json.loads(json.dumps(rows[0]))
    oracle_twin["result"]["checks"][1]["rc"] = 1  # an oracle row with other outputs for the same commands
    cls = cc.load_checklists(env["checklists"], {TASK}, [env["tasks"]])[0]
    oracles, odrops = cs.oracle_states(cs.cache_results([*rows, oracle_twin]), cls)
    assert TASK not in oracles and odrops[cs.DROP_AMBIGUOUS] == 1


def test_recorded_exit_codes_must_match_the_replayed_ones(env: dict[str, Any]):
    recs = [_record(e) for e in EPISODES]
    recs[0]["extra"]["history"][0]["rc"] = 2  # e_pass's archived step 0 exited 2; the cached replay exited 0
    _write_jsonl(env["episodes"], recs)
    _, _, eps, drops = _states(env)
    assert "e_pass" not in eps and drops[cs.DROP_NO_CACHE_ROW] == 1


def test_a_state_that_disagrees_or_does_not_reproduce_is_no_label(env: dict[str, Any]):
    recs = [_record(e) for e in EPISODES]
    recs[1]["outcome"]["tests_passed"] = True  # archived pass, replayed fail: the verdict does not reproduce
    _write_jsonl(env["episodes"], recs)
    summary = cs.build(_cfg(env))
    assert summary["states"]["episode/fail:archived_replayed_disagree"] == 1
    # Without e_content the only fail state fails the removal, which the own prefix contradicts: no pair (the
    # template is opt-in).
    assert summary["pairs"] == 0


# ── pairs ────────────────────────────────────────────────────────────────────


def test_matched_pair_same_prefix_and_checks_different_output_and_decision(env: dict[str, Any]):
    summary = cs.build(_cfg(env))
    out = env["tmp"] / "out"
    ex = {e["meta"]["outcome"]: e for e in _examples(out)}
    assert summary["examples"] == {"claim_done/fail": 1, "claim_done/pass": 1}
    p, f = ex["pass"], ex["fail"]
    n = p["meta"]["n_prefix_messages"]
    assert p["meta"]["prefix_kind"] == f["meta"]["prefix_kind"] == cs.PREFIX_OWN
    assert p["meta"]["prefix_id"] == f["meta"]["prefix_id"] == "own1"
    assert p["messages"][0]["content"] == harness.SYSTEM_PROMPT  # re-rendered
    assert p["messages"][:n] == f["messages"][:n]  # the same prefix
    # The own prefix removed /srv/junk, so the fail state is the content failure (e_content), not the removal one.
    assert f["meta"]["state"] == "e_content" and p["meta"]["state"] == "e_pass"
    for k in range(2):  # the same check turns (text and command), different outputs
        assert p["messages"][n + 2 * k] == f["messages"][n + 2 * k]
    assert p["messages"][n + 1] != f["messages"][n + 1] and "[exit code: 1]" in f["messages"][n + 1]["content"]
    pa = p["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    fa = f["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert pa == CLAIM  # the prefix's own claim, verbatim
    assert fa["status"] == "incomplete" and fa["incomplete"] == [WRITE_QUOTE] and fa["completed"] == [REQUEST]
    assert fa["evidence_commands"] == [CHECK_WRITE] and fa["blocker_category"] == ""
    for e in (p, f):  # loss on the check commands and the decision only
        assert e["loss_messages"] == [n, n + 2, len(e["messages"]) - 1]
        assert all(e["messages"][i]["role"] == "assistant" for i in e["loss_messages"])
        assert not any(e["messages"][i]["role"] == "tool" for i in e["loss_messages"])
    dpo = [json.loads(line) for line in (out / "dpo_pairs.jsonl").read_text().splitlines()]
    assert len(dpo) == 1 and dpo[0]["prompt"] == f["messages"][:-1]
    assert dpo[0]["chosen"] == f["messages"][-1]
    assert dpo[0]["rejected"]["tool_calls"][0]["function"]["arguments"]["status"] == "complete"


def test_a_removal_failure_pairs_with_the_template_when_the_prefix_removed_the_target(env: dict[str, Any]):
    recs = {r["episode_id"]: r for r in cs.read_jsonl(env["results"])}
    _write_jsonl(env["results"], [recs["e_pass"], recs["e_removal"]])
    cs.build(_cfg(env, template_prefix=True))
    ex = {e["meta"]["outcome"]: e for e in _examples(env["tmp"] / "out")}
    assert {e["meta"]["prefix_kind"] for e in ex.values()} == {cs.PREFIX_TEMPLATE}
    assert ex["fail"]["meta"]["state"] == "e_removal"
    assert [m["role"] for m in ex["pass"]["messages"][:2]] == ["system", "user"]
    pa = ex["pass"]["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert pa["status"] == "complete" and pa["completed"] == [WRITE_QUOTE, REQUEST]  # the template's claim
    # A prefix that never removed the target is not contradicted: it is used.
    _write_jsonl(env["prefixes"], [_prefix_example("own2", destroys=False)])
    cs.build(_cfg(env, template_prefix=True))
    assert {e["meta"]["prefix_id"] for e in _examples(env["tmp"] / "out")} == {"own2"}
    # Without the template fallback (the default) the contradicted task has no pair.
    _write_jsonl(env["prefixes"], [_prefix_example()])
    summary = cs.build(_cfg(env))
    assert summary["pairs"] == 0
    assert summary["pair_drops"]["prefix_skipped:own_transcript:removal_contradicts_prefix"] == 1


def test_a_prefix_on_another_instruction_is_refused(env: dict[str, Any]):
    _write_jsonl(env["prefixes"], [_prefix_example(instruction="Another task.\n")])
    summary = cs.build(_cfg(env, template_prefix=True))
    assert summary["prefixes"]["drops"] == {cs.DROP_PREFIX_INSTRUCTION: 1}
    assert summary["examples_by_prefix"] == {"fail/template": 1, "pass/template": 1}


def test_template_prefix_is_off_by_default(env: dict[str, Any]):
    """The zero-work prefix would train 'read the instruction, run the checks, report incomplete' with no attempt."""
    assert cs.Config(tasks_roots=[], episodes=[], prefix_examples=[]).template_prefix is False
    _write_jsonl(env["prefixes"], [])
    assert cs.build(_cfg(env))["pairs"] == 0
    assert cs.build(_cfg(env, template_prefix=True))["examples_by_prefix"] == {"fail/template": 1, "pass/template": 1}


def test_a_fail_state_on_a_requirement_the_prefix_observed_is_refused(env: dict[str, Any]):
    """A prefix that wrote /app/out.json (and showed it) cannot be followed by `test -s /app/out.json` exiting 1."""
    _write_jsonl(env["prefixes"], [_prefix_example(names_output=True)])
    summary = cs.build(_cfg(env))
    assert summary["pairs"] == 0
    assert summary["pair_drops"]["prefix_skipped:own_transcript:fail_state_contradicts_prefix"] == 1
    cls, _, eps, _ = _states(env)
    pre = cs.load_prefixes([env["prefixes"]], [TASK], {TASK: INSTRUCTION})[0][TASK][0]
    assert cs.prefix_contradicts(pre, eps["e_content"], cls[TASK])  # r1 observed and failing
    assert not cs.prefix_contradicts(pre, eps["e_pass"], cls[TASK])  # observed, passing, no output
    assert not cs.prefix_contradicts(cs.template_prefix(TASK, INSTRUCTION), eps["e_content"], cls[TASK])


def test_a_state_printing_text_on_an_observed_requirement_is_refused(env: dict[str, Any]):
    """Another run's printed file contents need not match what the prefix showed, even on a passing check."""
    cls, _, eps, _ = _states(env)
    pre = cs.load_prefixes([env["prefixes"]], [TASK], {TASK: INSTRUCTION})[0][TASK][0]
    s = eps["e_pass"]
    results = [{**d, "output": "3 -rw-r--r--\n"} if d["id"] == "r2" else d for d in s.result.results]
    printed = cs.State(**{**s.__dict__, "result": cc.CheckResult(s.unit_id, s.task, s.result.checklist_sha, results)})
    assert cs.prefix_contradicts(pre, printed, cls[TASK])  # r2 (/srv/junk) observed by the prefix's rm


def test_decisions_are_graded_with_the_spliced_transcripts_actions(env: dict[str, Any]):
    _, _, eps, _ = _states(env)
    pre = cs.load_prefixes([env["prefixes"]], [TASK], {TASK: INSTRUCTION})[0][TASK][0]
    assert cs.state_outcome(eps["e_content"], pre, 2).n_actions == len(pre.bash_commands) + 2 == 4
    assert cs.state_outcome(eps["e_content"], cs.template_prefix(TASK, INSTRUCTION), 0).n_actions == 0


def test_oracle_is_the_pass_state_without_a_passing_episode(env: dict[str, Any]):
    recs = {r["episode_id"]: r for r in cs.read_jsonl(env["results"])}
    _write_jsonl(env["results"], [recs["e_content"]])
    summary = cs.build(_cfg(env))
    assert summary["examples_by_state"] == {"fail/episode": 1, "pass/oracle": 1}


def test_decisions_are_graded_on_the_state_facts(env: dict[str, Any]):
    cls, _, eps, _ = _states(env)
    pre = cs.load_prefixes([env["prefixes"]], [TASK], {TASK: INSTRUCTION})[0][TASK][0]
    pair = cs.build_pair(pre, eps["e_pass"], eps["e_content"], cls[TASK], source=SOURCE)
    assert isinstance(pair, cs.Pair)
    # A pass state whose episode damaged a preserved file: a claim there is no success_claimed, so no pair.
    damaged = eps["e_pass"].outcome | {"damage_paths": ["/app/config"]}
    bad = cs.State(**{**eps["e_pass"].__dict__, "outcome": damaged})
    assert cs.build_pair(pre, bad, eps["e_content"], cls[TASK], source=SOURCE) == cc.DROP_CASE


def test_tokenisation_trains_the_check_commands_and_the_decision_only(env: dict[str, Any]):
    from honest_agents.sft import data
    from honest_agents.sft.data import SFTExample, make_renderer
    from tests.sft.helpers import tiny_tokenizer

    cs.build(_cfg(env))
    tok = tiny_tokenizer()
    renderer = make_renderer(tok, "qwen3")
    for e in _examples(env["tmp"] / "out"):
        t = data.tokenize_example(SFTExample(**e), renderer)
        text = tok.decode([i for i, label in zip(t.input_ids, t.labels, strict=True) if label != data.IGNORE_INDEX])
        assert CHECK_WRITE in text and CHECK_REMOVED in text and "submit_report" in text
        assert "[exit code" not in text and "still there" not in text and "Running" not in text


# ── quotas and sources ───────────────────────────────────────────────────────


def _pair(source: str, k: int) -> cs.Pair:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    ex = cc.SFTExample(f"{source}:{k}", cc.KIND, f"{source}__t{k}", msgs, [2])
    return cs.Pair(f"{source}__t{k}", source, "", cs.template_prefix("t", "u"), ex, ex, None)


def test_source_share_cap_drops_whole_pairs_from_the_largest_sources():
    pairs = [_pair("big", k) for k in range(5)] + [_pair(f"s{k}", 0) for k in range(10)]
    kept, dropped = cs.cap_source_share(pairs, 0.2)
    per = Counter(p.source for p in kept)
    assert dropped == 3 and per["big"] == 2 and max(per.values()) / len(kept) <= 0.2  # 3 of 13 would be 0.23
    assert cs.cap_source_share(pairs, 1.0) == (pairs, 0)


def test_excluded_sources_are_refused(env: dict[str, Any]):
    (env["data"] / "split_seeds.json").write_text(json.dumps({SOURCE: "val"}))
    with pytest.raises(cc.ExcludedSourceError):
        cs.build(_cfg(env))


def test_cli_writes_the_build_and_its_manifest(env: dict[str, Any], capsys):
    out = env["tmp"] / "cli"
    argv = ["--tasks-root", str(env["tasks"]), "--episodes", str(env["episodes"]), "--prefix-examples",
            str(env["prefixes"]), "--cache", str(env["cache"]), "--check-results", str(env["results"]),
            "--checklists", str(env["checklists"]), "--data-dir", str(env["data"]), "--out", str(out), "--dpo",
            "--max-source-share", "1"]  # fmt: skip
    assert cs.main(argv) == 0
    assert "claim_done/pass | 1" in capsys.readouterr().out
    m = read_manifest(out / cc.MANIFEST_FILE)
    assert m.build["outputs"]["examples.jsonl"] == hashlib.sha256((out / "examples.jsonl").read_bytes()).hexdigest()
    assert [(e.outcome, e.committed, e.origin) for e in m.examples] == [
        ("pass", True, cs.ORIGIN_SPLICE_PASS), ("fail", False, cs.ORIGIN_SPLICE_FAIL)]  # fmt: skip


# ── L0: the committed build ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    if not (BUILD / cc.MANIFEST_FILE).is_file():
        pytest.skip("no committed claim splice build")
    return {
        "manifest": read_manifest(BUILD / cc.MANIFEST_FILE),
        "examples": _examples(BUILD),
        "dpo": [json.loads(line) for line in (BUILD / "dpo_pairs.jsonl").read_text().splitlines()],
        "summary": json.loads((BUILD / "summary.json").read_text()),
    }


def test_committed_build_matches_its_manifest(committed: dict[str, Any]):
    m = committed["manifest"]
    for name, sha in m.build["outputs"].items():
        assert hashlib.sha256((BUILD / name).read_bytes()).hexdigest() == sha, name
    assert [e.id for e in m.examples] == [e["id"] for e in committed["examples"]]
    assert len(committed["dpo"]) == m.build["dpo_pairs"] == m.build["pairs"]
    # The coverage report's hard rules: per-task cap, source share, licensed decisions, train split, schema.
    report = cov.coverage(m, load_select_sources(Path("data")))
    assert report.ok(), report.failures()


def test_committed_build_has_no_eval_source(committed: dict[str, Any]):
    sources = {e.source for e in committed["manifest"].examples} | {e["meta"]["source"] for e in committed["examples"]}
    assert len(sources) == committed["summary"]["sources_in_build"]
    assert not sources & load_select_sources(Path("data"))
    if (Path("data") / "split_seeds.json").is_file():  # untracked: present on the Mac, not in a fresh clone
        assert not sources & excluded_from_training("data")


def test_committed_build_has_no_zero_work_prefix(committed: dict[str, Any]):
    assert not any(e["meta"]["prefix_kind"] == cs.PREFIX_TEMPLATE for e in committed["examples"])
    assert all(e["meta"]["n_prefix_messages"] > 2 for e in committed["examples"])


def test_committed_pairs_share_a_prefix_and_never_claim_on_a_fail(committed: dict[str, Any]):
    by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
    for e in committed["examples"]:
        by_prefix.setdefault(f"{e['meta']['prefix_id']}:{e['variant']}", {})[e["meta"]["outcome"]] = e
    assert all(set(v) == {"pass", "fail"} for v in by_prefix.values())
    for pair in by_prefix.values():
        p, f = pair["pass"], pair["fail"]
        n = p["meta"]["n_prefix_messages"]
        assert p["messages"][:n] == f["messages"][:n] and p["loss_messages"] == f["loss_messages"]
        assert all(p["messages"][i] == f["messages"][i] for i in p["loss_messages"][:-1])  # same check turns
        assert all(p["messages"][i]["role"] == "assistant" and i >= n for i in p["loss_messages"])
        assert all(p["meta"]["verdicts"]) and not all(f["meta"]["verdicts"])
        assert p["meta"]["tests_passed"] is True and f["meta"]["tests_passed"] is False
        status = {k: e["messages"][-1]["tool_calls"][0]["function"]["arguments"]["status"] for k, e in pair.items()}
        assert status == {"pass": "complete", "fail": "incomplete"}
    for d in committed["dpo"]:
        assert d["chosen"]["tool_calls"][0]["function"]["arguments"]["status"] == "incomplete"
        assert d["rejected"]["tool_calls"][0]["function"]["arguments"]["status"] == "complete"
