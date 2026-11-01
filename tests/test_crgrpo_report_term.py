"""CR-GRPO's report term in the trainer (honest_agents.trainer): finding the status token in a real Qwen3.5 rollout,
the binary log-odds and its gradient, and the loss against the analytic natural step."""

from __future__ import annotations

import math

import pytest
import torch

from honest_agents import trainer as tr

C, I = 14311, 258  # Qwen3.5: `complete`, first token of `incomplete`


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")


def _rollout(tok, status: str, *, earlier: str = "") -> list[int]:
    """A completion as the policy writes it: a bash call, then the report call (Qwen3.5's XML tool format)."""
    text = (
        "Let me check.\n<tool_call>\n<function=bash>\n<parameter=command>\nls /srv\n</parameter>\n</function>\n"
        "</tool_call>" + earlier + "Done.\n<tool_call>\n<function=submit_report>\n<parameter=status>\n" + status
        + "\n</parameter>\n<parameter=completed>\n[\"built it\"]\n</parameter>\n</function>\n</tool_call>"
    )  # fmt: skip
    return tok(text, add_special_tokens=False).input_ids


def _decode(tok):
    return lambda ids: tok.decode(ids, skip_special_tokens=False)


@pytest.mark.parametrize(("status", "first", "claimed"), [("complete", C, True), ("incomplete", I, False)])
def test_the_status_token_of_the_report_call_is_found_in_a_real_rollout(tok, status, first, claimed):
    ids = _rollout(tok, status)
    j = tr.status_position(ids, None, C, I, _decode(tok), claimed)
    assert j > 0 and ids[j] == first and tok.decode(ids[:j]).endswith(tr.STATUS_MARKER)


def test_a_status_the_grader_did_not_see_or_a_tool_output_is_never_used(tok):
    ids = _rollout(tok, "complete")
    assert tr.status_position(ids, None, C, I, _decode(tok), claimed=False) == -1  # disagrees with the graded report
    j = tr.status_position(ids, None, C, I, _decode(tok), claimed=None)
    masked = [1] * len(ids)
    masked[j] = 0  # the token came from a tool result, not the model
    assert tr.status_position(ids, masked, C, I, _decode(tok), claimed=True) == -1
    plain = tok("The word complete appears here", add_special_tokens=False).input_ids
    assert tr.status_position(plain, None, C, I, _decode(tok), None) == -1


def test_the_last_report_call_is_the_one_used(tok):
    earlier = (
        "\n<tool_call>\n<function=submit_report>\n<parameter=status>\ncomplete\n</parameter>\n</function>\n</tool_call>"
    )
    ids = _rollout(tok, "incomplete", earlier=earlier)
    j = tr.status_position(ids, None, C, I, _decode(tok), claimed=False)
    assert ids[j] == I and j > len(ids) // 2


def test_binary_log_odds_value_and_capped_gradient():
    lp = torch.tensor([math.log(0.8), math.log(0.8), math.log(0.999)], requires_grad=True)
    sampled_c = torch.tensor([True, False, True])
    z = tr.status_log_odds(lp, sampled_c, grad_clip=20.0)
    assert z[0].item() == pytest.approx(math.log(0.8 / 0.2)) and z[1].item() == pytest.approx(math.log(0.2 / 0.8))
    z.sum().backward()
    assert lp.grad[0].item() == pytest.approx(1 / 0.2) and lp.grad[1].item() == pytest.approx(-1 / 0.2)
    assert lp.grad[2].item() == pytest.approx(20.0)  # 1 / 0.001 capped


def test_the_report_term_is_the_natural_step_on_the_status_logit():
    """A two-option toy: the status logit is a parameter; d l_rep / d z = -c * delta / N exactly (the natural gradient
    of E_r[L] w.r.t. z), whichever option was sampled, and nothing moves once z passes z_max."""
    for sampled_complete, delta in ((True, 0.7), (False, 0.7), (True, -1.4), (False, -1.4)):
        theta = torch.tensor(0.3, requires_grad=True)  # z itself: log p(c) - log p(i)
        logp_c, logp_i = torch.nn.functional.logsigmoid(theta), torch.nn.functional.logsigmoid(-theta)
        logp = (logp_c if sampled_complete else logp_i).reshape(1, 1)
        ids = torch.tensor([[C if sampled_complete else I]])
        loss, stats = tr.report_term(logp, ids, torch.tensor([0]), torch.tensor([delta]), C,
                                     c_rep=2.0, z_max=6.0, grad_clip=1e6, denominator=4.0)  # fmt: skip
        loss.backward()
        assert theta.grad.item() == pytest.approx(-2.0 * delta / 4.0, rel=1e-4)
        assert stats[tr.METRIC_REP_N] == 1.0
    far = torch.tensor([[math.log(1 - 1e-4)]], requires_grad=True)  # z about +9.2, delta > 0: frozen
    loss, stats = tr.report_term(far, torch.tensor([[C]]), torch.tensor([0]), torch.tensor([1.0]), C,
                                 c_rep=1.0, z_max=6.0, grad_clip=20.0, denominator=1.0)  # fmt: skip
    loss.backward()
    assert far.grad.item() == 0.0 and stats[tr.METRIC_REP_FROZEN] == 1.0


def test_rows_without_a_status_token_get_no_report_term():
    logp = torch.zeros(2, 3, requires_grad=True)
    loss, stats = tr.report_term(logp, torch.zeros(2, 3, dtype=torch.long), torch.tensor([-1, -1]),
                                 torch.tensor([1.0, -1.0]), C, c_rep=1.0, z_max=6.0, grad_clip=20.0, denominator=2.0)  # fmt: skip
    assert loss.item() == 0.0 and stats[tr.METRIC_REP_N] == 0.0


# ── the trainer hooks on the real TRL loss (tiny Qwen3, the padding-trimmer harness of test_trainer_loop) ──


def _crgrpo_round(seed: int = 0):
    from tests.test_trainer_loop import _round

    r = _round(seed=seed)
    rows = r["advantages"].numel()
    lengths = r["completion_mask"].sum(1)
    pos = torch.full((rows,), -1, dtype=torch.long)
    delta = torch.zeros(rows)
    for i in range(rows):
        if i % 3 == 2 or lengths[i] < 2:
            continue  # rows without a report
        j = int(lengths[i]) - 1
        r["completion_ids"][i, j] = 5 if i % 2 else 6  # the sampled status token
        r["tool_mask"][i, j] = 0  # as `_attach_report_signal` leaves it: out of the main loss
        pos[i], delta[i] = j, (0.7 if i % 2 else -1.3)
    r["crgrpo_pos"], r["crgrpo_delta"] = pos, delta
    return r


def test_the_loss_is_trls_plus_the_report_term_in_value_and_gradient():
    from honest_agents import crgrpo as cr
    from tests.test_trainer_loop import _loss_stub, _round_loss_and_grads, _tiny_policy

    model = _tiny_policy()
    round_ = _crgrpo_round()
    settings = cr.RunSettings(c_rep=1.5)
    ours = _loss_stub(model, trim=True, sort=True)
    ours.crgrpo, ours.status_tokens = settings, (5, 6)
    total, grads = _round_loss_and_grads(ours, model, round_)

    stock = _loss_stub(model, trim=True, sort=True)  # crgrpo None: TRL's loss, the extra keys ignored
    main, main_grads = _round_loss_and_grads(stock, model, round_)

    model.zero_grad()  # the report term alone, on the same forward
    from trl.trainer.utils import split_tensor_dict

    from tests.test_trainer_loop import trainer_mod

    rep_total = 0.0
    for batch in trainer_mod.sort_micro_batches(split_tensor_dict(dict(round_), 4)):
        ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
        mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1)
        logps, _, _ = stock._get_per_token_logps_and_entropies(model, ids, mask, batch["completion_ids"].size(1))
        rep, _ = tr.report_term(logps, batch["completion_ids"], batch["crgrpo_pos"], batch["crgrpo_delta"], 5,
                                c_rep=1.5, z_max=6.0, grad_clip=20.0, denominator=float(2 * 4))  # fmt: skip
        rep.backward()
        rep_total += float(rep.detach())
    rep_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert rep_total != 0.0
    assert total == pytest.approx(main + rep_total, rel=1e-5)
    for name, g in grads.items():
        torch.testing.assert_close(g, main_grads[name] + rep_grads[name], rtol=1e-4, atol=1e-6)
    assert ours._metrics["train"][tr.METRIC_REP_N]  # logged per micro-batch


def test_attach_masks_the_found_status_token_and_consumes_the_signal(tok):
    from collections import defaultdict
    from types import SimpleNamespace

    from honest_agents.report import SelfReport, Status

    claim, not_claim = _rollout(tok, "complete"), _rollout(tok, "incomplete")
    width = max(len(claim), len(not_claim))
    ids = torch.zeros(3, width, dtype=torch.long)
    cmask = torch.zeros(3, width, dtype=torch.long)
    for i, row in enumerate((claim, not_claim, claim)):
        ids[i, : len(row)], cmask[i, : len(row)] = torch.tensor(row), 1
    envs = [SimpleNamespace(_crgrpo={"delta": 0.9, "v": 0.5}, _report=SelfReport(status=Status.COMPLETE)),
            SimpleNamespace(_crgrpo={"delta": -1.2, "v": 0.5}, _report=SelfReport(status=Status.INCOMPLETE)),
            SimpleNamespace(_crgrpo=None, _report=None)]  # fmt: skip
    stub = object.__new__(tr.HonestGRPOTrainer)
    stub.__dict__.update(environments=envs, processing_class=tok, status_tokens=(C, I),
                         model=SimpleNamespace(training=True), _metrics={"train": defaultdict(list)})  # fmt: skip
    out = {"completion_ids": ids, "completion_mask": cmask, "tool_mask": cmask.clone()}
    stub._attach_report_signal(out)
    pos, delta = out["crgrpo_pos"], out["crgrpo_delta"]
    assert ids[0, pos[0]] == C and ids[1, pos[1]] == I and pos[2] == -1
    assert delta.tolist() == pytest.approx([0.9, -1.2, 0.0])
    assert (
        out["tool_mask"][0, pos[0]] == 0
        and out["tool_mask"][1, pos[1]] == 0
        and out["tool_mask"].sum() == cmask.sum() - 2
    )
    assert all(e._crgrpo is None for e in envs)  # consumed
    assert stub._metrics["train"]["crgrpo/rep/found_share"] == [pytest.approx(2 / 3)]
