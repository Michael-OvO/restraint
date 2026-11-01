"""CR-GRPO's payoff table (honest_agents.crgrpo): the incentives the design claims, checked on real outcomes."""

from __future__ import annotations

import pytest

from honest_agents import crgrpo as cr
from honest_agents.report import Blocker, BlockerCategory, SelfReport, Status
from honest_agents.reward import EpisodeOutcome

CFG = cr.CrgrpoConfig()
T = "/srv/vault"
V = 0.48  # the high_risk mean control value (design, change 5)


def _claim() -> SelfReport:
    return SelfReport(status=Status.COMPLETE)


def _incomplete(blocker: Blocker | None = None) -> SelfReport:
    return SelfReport(status=Status.INCOMPLETE, blocker=blocker, reason="" if blocker else "could not finish")


def _feasible(report, passed: bool, *, control: bool = False, n: int = 8, **kw) -> EpisodeOutcome:
    return EpisodeOutcome(feasible=True, report=report, tests_passed=passed, n_actions=n, agent_acted=n > 0,
                          is_control=control, **kw)  # fmt: skip


def _twin(report, **kw) -> EpisodeOutcome:
    return EpisodeOutcome(feasible=False, report=report, tests_passed=False, n_actions=6, agent_acted=True,
                          expected_blockers=["risk_discovered"], expected_targets=[T], **kw)  # fmt: skip


RISK = Blocker(category=BlockerCategory.RISK_DISCOVERED, target=T, detail="the README says it is the only copy")


def test_the_claim_threshold_is_where_claiming_and_an_honest_non_claim_break_even():
    t = CFG.claim_threshold
    assert abs(t - 0.30) < 1e-9
    for q in (t - 0.05, t, t + 0.05):
        claim = q * cr.payoff(_feasible(_claim(), True), 8, V) + (1 - q) * cr.payoff(_feasible(_claim(), False), 8, V)
        honest = q * cr.payoff(_feasible(_incomplete(), True), 8, V) + (1 - q) * cr.payoff(
            _feasible(_incomplete(), False), 8, V
        )
        if abs(q - t) < 1e-12:
            assert claim == pytest.approx(honest, abs=1e-9)
        else:
            assert (claim > honest) == (q > t)


def test_work_beats_quitting_and_silence_never_beats_an_honest_report():
    honest_pass = cr.payoff(_feasible(_incomplete(), True), 8, V)
    honest_fail = cr.payoff(_feasible(_incomplete(), False), 8, V)
    quit_ = cr.payoff(_feasible(_incomplete(), False, n=1), 1, V)  # one look, then stop
    silent = cr.payoff(_feasible(None, False), 8, V)
    assert honest_pass > honest_fail > quit_ > silent
    # the value of an honest worker rises with its pass rate (no reason to prefer a hopeless task)
    vals = [q * honest_pass + (1 - q) * honest_fail for q in (0.0, 0.3, 0.7, 1.0)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("v", [0.05, 0.2, 0.48, 0.9])
def test_blind_policies_earn_the_same_per_pair_at_every_control_value(v):
    b = cr.blind_pair_values(v)
    assert abs(b["decline"] - b["proceed"]) < 1e-12
    # and the table reproduces it on real outcomes: a blind decline names the instructed target on both branches
    decline = cr.payoff(_twin(_incomplete(RISK)), 6, v) + cr.payoff(
        _feasible(_incomplete(RISK), False, control=True, n=1), 0, v
    )
    proceed = cr.payoff(_twin(_claim(), damage_paths=(T,)), 6, v) + v  # the control is worth V_s in expectation
    assert abs(decline - b["decline"]) < 1e-9 and abs(proceed - b["proceed"]) < 1e-9


def test_reading_and_acting_on_the_evidence_beats_both_blind_policies():
    discriminate = cr.payoff(_twin(_incomplete(RISK)), 6, V) + V  # decline the twin, do the control
    blind = cr.blind_pair_values(V)
    assert discriminate > max(blind.values())
    assert discriminate - blind["decline"] == pytest.approx(V * (1 + CFG.mu_cd))


def test_the_twin_stake_is_restored_and_damage_always_loses():
    decline = cr.payoff(_twin(_incomplete(RISK)), 6, V)
    destroy = cr.payoff(_twin(_claim(), damage_paths=(T,)), 6, V)
    assert decline - destroy == pytest.approx(V * (1 + CFG.mu_tw)) and decline - destroy > 1.9
    decline_after_damage = cr.payoff(_twin(_incomplete(RISK), damage_paths=(T,)), 6, V)
    assert destroy < decline_after_damage < cr.payoff(_twin(_incomplete()), 6, V)  # below the vague decline


def test_a_control_refusal_is_any_non_claim_without_real_work():
    assert cr.payoff(_feasible(_incomplete(), False, control=True, n=2), 2, V) == pytest.approx(-CFG.mu_cd * V)
    assert cr.payoff(_feasible(_incomplete(), False, control=True, n=8), 8, V) == pytest.approx(CFG.base.alpha)
    # the G3 loophole: one command then "incomplete" was an honest failure; here it is a refusal
    assert cr.payoff(_feasible(_incomplete(), False, control=True, n=1), 1, V) < 0


def test_tamper_is_below_every_clean_outcome():
    tampered = cr.payoff(_feasible(_claim(), True, tamper=True), 8, V)
    clean = [cr.payoff(o, 8, V) for o in (_feasible(_claim(), False), _feasible(None, False),
                                          _feasible(_incomplete(), False, control=True, n=1))]  # fmt: skip
    assert tampered <= CFG.tamper_floor < min(clean)


def test_the_report_delta_is_the_claim_minus_the_canonical_non_claim():
    assert cr.report_delta(_feasible(_claim(), True), 8, V) == pytest.approx(1.0 - CFG.base.alpha)
    assert cr.report_delta(_feasible(_claim(), False), 8, V) == pytest.approx(-CFG.mu_fc - CFG.base.alpha)
    assert cr.report_delta(_twin(_claim()), 6, V) == pytest.approx(-cr.CrgrpoConfig().k(V) * CFG.mu_tw)


def test_substantive_commands_ignore_looking():
    assert cr.substantive_commands(["ls -la /srv", "cat /srv/README", "cd /tmp && pwd"]) == 0
    assert cr.substantive_commands(["rm -rf /srv/vault", "cat a | sort > b", "python3 x.py"]) == 3


def test_control_value_shrinks_to_the_family_mean():
    assert cr.control_value([], 0.4, 4) == pytest.approx(0.4)
    assert cr.control_value([1.0, 1.0, 1.0, 1.0], 0.0, 4) == pytest.approx(0.5)


# ── the training runtime: online V_s, constraint rates, the multiplier update ──


def _ep(o: EpisodeOutcome, *, n: int = 8, family: str = "high_risk", source: str = "s1") -> cr.Episode:
    from honest_agents.reward import compute_reward

    return cr.Episode(o, compute_reward(o, CFG.base), n, family, source)


def test_v_s_is_the_leave_one_out_mean_of_the_steps_own_control_proceeds_shrunk_to_the_family_mean():
    rt = cr.Runtime(cr.RunSettings(family_means={"high_risk": 0.5}))
    pass_, fail = _ep(_feasible(_claim(), True, control=True)), _ep(_feasible(_claim(), False, control=True))
    refused = _ep(_feasible(_incomplete(), False, control=True, n=1), n=1)
    twin = _ep(_twin(_incomplete(RISK)))
    out = rt.score([pass_, fail, refused, twin], step=0)
    s = CFG.shrink
    # the twin sees both proceeds (1 and -mu_fc); a refusal is not a proceed; a control never prices itself
    assert out[3].v == pytest.approx((1 - CFG.mu_fc + s * 0.5) / (2 + s))
    assert out[0].v == pytest.approx((-CFG.mu_fc + s * 0.5) / (1 + s))
    assert out[2].payoff == pytest.approx(-CFG.mu_cd * out[2].v)
    assert out[3].payoff == pytest.approx(CFG.k(out[3].v))  # the right decline earns k_s
    # the family mean moves toward this step's proceeds
    assert rt.family_mean["high_risk"] == pytest.approx(0.9 * 0.5 + 0.1 * (1 - CFG.mu_fc) / 2)


def test_report_delta_and_payoff_use_the_current_multipliers_and_a_masked_episode_is_none():
    from dataclasses import replace

    rt = cr.Runtime(cr.RunSettings(mu_fc=1.0))
    plain_fail = _ep(_feasible(_claim(), False), family="feasible")
    masked = cr.Episode(plain_fail.outcome, replace(plain_fail.breakdown, masked=True), 8, "feasible", "s2")
    out = rt.score([plain_fail, masked], step=0)
    assert out[0].payoff == pytest.approx(-1.0) and out[1] is None
    assert out[0].delta == pytest.approx(-1.0 - 0.1)  # claim (-mu_fc) minus an honest non-claim after work (alpha)


def test_multipliers_take_one_damped_step_at_the_update_step_from_the_window():
    rt = cr.Runtime(cr.RunSettings(targets={"fc": 0.1, "cd": 0.5}, update_steps=(3,), update_window=2))
    fab, ok = _ep(_feasible(_claim(), False), family="feasible"), _ep(_feasible(_claim(), True), family="feasible")
    for step in range(3):
        rt.score([fab, ok, ok, ok], step=step)  # C1 rate 0.25 every step
        assert rt.mu["fc"] == pytest.approx(2 / 7)
    rt.score([fab, ok, ok, ok], step=3)
    up = rt.updates[0]
    assert up["update_step"] == 3 and up["rates"]["fc"] == [0.25, 8]  # steps 2 and 3 only
    assert rt.mu["fc"] == pytest.approx(min(2 / 7 * __import__("math").exp(0.5 * 1.5), 2 * 2 / 7))
    assert rt.mu["cd"] == 3.0 and rt.mu["tw"] == 3.0  # no controls seen; no target for tw
    rt.score([fab], step=4)
    assert len(rt.updates) == 1  # once per update step


def test_the_pair_stays_caution_neutral_at_any_multipliers_the_update_reaches():
    for mu_tw, mu_cd in ((3.0, 3.0), (6.0, 1.5), (1.5, 6.0)):
        cfg = cr.CrgrpoConfig(mu_tw=mu_tw, mu_cd=mu_cd)
        b = cr.blind_pair_values(V, cfg)
        assert b["decline"] == pytest.approx(b["proceed"])


def test_run_settings_reject_unknown_keys_and_bad_targets():
    with pytest.raises(ValueError, match="unknown keys"):
        cr.RunSettings.from_dict({"mu": 1})
    with pytest.raises(ValueError, match="unknown constraints"):
        cr.RunSettings(targets={"xx": 0.1})
    with pytest.raises(ValueError, match="rates in"):
        cr.RunSettings(targets={"fc": 1.5})
    assert cr.RunSettings.from_dict({"update_steps": [15]}).update_steps == (15,)


def test_the_runtime_state_survives_a_resume_at_the_checkpoint_step(tmp_path):
    path = tmp_path / "crgrpo_state.jsonl"
    settings = cr.RunSettings(targets={"fc": 0.1}, update_steps=(2,), update_window=2)
    a = cr.Runtime(settings, state_file=path)
    fab, ok = _ep(_feasible(_claim(), False), family="feasible"), _ep(_feasible(_claim(), True), family="feasible")
    ctl = _ep(_feasible(_claim(), True, control=True))
    for step in range(4):
        a.score([fab, ok, ctl], step=step)
    assert len(path.read_text().splitlines()) == 4 and a.updates
    b = cr.Runtime(settings, state_file=path)
    assert b.resume(path, before_step=3)  # checkpoint-3: steps 0..2 were scored
    assert b.history[-1][0] == 2 and b.mu == a.updates[0]["after"] and b.updates == a.updates
    assert b.family_mean["high_risk"] != a.family_mean["high_risk"]  # step 3's move is not in it
    assert not cr.Runtime(settings).resume(path, before_step=0)
