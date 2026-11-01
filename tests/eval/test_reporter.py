"""The reporter (eval.reporter): the cut, the threshold rule, and withdrawing claims on real archived runs."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from honest_agents.eval import reporter as rp

STAGED = Path.home() / "Documents" / "honest-agents-archive" / "staging" / "runs" / "harbor"


def _call(status: str, name: str = "submit_report") -> dict:
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps({"status": status})}}


def test_the_cut_ends_at_the_last_claim_and_skips_non_claims():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [_call("x", name="bash")]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done", "tool_calls": [_call("complete")]},
            {"role": "tool", "content": "recorded"}]  # fmt: skip
    cut = rp.status_cut(msgs)
    assert len(cut) == 5 and cut[-1]["content"] == "done"
    assert cut[-1]["tool_calls"][0]["function"]["arguments"] == {"status": rp.cb_sentinel()}
    no_claim = [*msgs[:4], {"role": "assistant", "content": "", "tool_calls": [_call("incomplete")]}]
    assert rp.status_cut(no_claim) is None


def test_the_threshold_maximises_the_payoff_of_the_kept_claims():
    # three passing claims (ΔL +0.9) above two failing ones (−0.386): withdraw exactly the two failures
    res = rp.choose_threshold([0.9, 0.8, 0.7, 0.3, 0.2], [0.9, 0.9, 0.9, -0.386, -0.386])
    assert res["threshold"] == pytest.approx(0.7) and res["kept"] == 3
    assert res["utility"] == pytest.approx(2.7) and res["utility_keep_all"] == pytest.approx(2.7 - 0.772)
    # all passing: keep everything (t = 0)
    assert rp.choose_threshold([0.4, 0.6], [0.9, 0.9])["threshold"] == 0.0


def test_withdraw_changes_only_rejected_claims():
    claim = {"status": "complete", "completed": ["x"], "reason": "r"}
    assert rp.withdraw(claim, 0.2, 0.5) == ({**claim, "status": "incomplete", "blocker": None, "reason": "",
                                             "completed": []}, True)  # fmt: skip
    assert rp.withdraw(claim, 0.6, 0.5) == (claim, False)
    assert rp.withdraw({"status": "incomplete"}, 0.0, 0.5)[1] is False
    rows = [{"episode_id": "a/b", "outcome": {"report": claim}}, {"episode_id": "c", "outcome": {"report": claim}}]
    out, n = rp.withdraw_episodes(rows, {"a_b": 0.1, "c": 0.9}, 0.5)
    assert n == 1 and out[0]["outcome"]["report"]["status"] == "incomplete" and out[1]["outcome"]["report"] == claim


def test_withdrawing_on_a_real_tb2_job_reproduces_its_units_and_removes_false_claims():
    jobs = sorted(glob.glob(str(STAGED / "it19a_g2_tb2" / "tb2__*")))
    if not jobs:
        pytest.skip("the it19a TB2 job is not staged")
    from honest_agents.eval.final_benchmark import tb2_units
    from honest_agents.eval.harbor_results import load_harbor_job

    job = Path(jobs[0])
    names = [Path(p).parents[1].name for p in glob.glob(str(job / "*" / "agent" / "honest_transcript.json"))]
    original = tb2_units(load_harbor_job(job))
    kept, n0 = rp.withdraw_harbor(job, dict.fromkeys(names, 1.0), 0.5, "terminal_bench_2")
    assert n0 == 0 and kept == original
    gone, n1 = rp.withdraw_harbor(job, dict.fromkeys(names, 0.0), 0.5, "terminal_bench_2")
    assert n1 > 0 and sum(u["false_claim"] for u in gone) == 0
    assert [u["success"] for u in gone] == [u["success"] for u in original]  # tests do not move


def test_a_real_harbor_claim_renders_to_the_status_cut():
    """A real TB2 claim from the it19a G2 job, rendered with the served Qwen3.5 template, ends where the claim decision
    is made, as the training items do."""
    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec
    from honest_agents.sft import confidence_bandit as cb

    files = sorted(glob.glob(str(STAGED / "it19a_g2_tb2" / "tb2__*" / "*" / "agent" / "honest_transcript.json")))
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")
    cut = next((c for f in files if (c := rp.status_cut(json.loads(Path(f).read_text()))) is not None), None)
    if cut is None:
        pytest.skip("no staged claiming TB2 transcript")
    prompt = cb.render_prefix(cut, tok, tool_schemas(), dict(family_spec("qwen3_5").chat_template_kwargs))
    assert prompt.endswith("<parameter=status>\n") and "submit_report" in prompt[-300:]
