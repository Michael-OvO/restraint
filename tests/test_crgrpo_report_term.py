"""CR-GRPO's report term in the trainer (honest_agents.trainer): finding the status token in a real Qwen3.5 rollout,
the binary log-odds and its gradient, and the loss against the analytic natural step."""

from __future__ import annotations

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


def test_the_report_term_is_the_natural_step_on_the_status_logit():
    """d l_rep / d z = -c * delta / N exactly (the natural gradient of E_r[L] w.r.t. the claim log-odds), and nothing
    moves once z passes z_max in the direction of delta."""
    for delta in (0.7, -1.4):
        z = torch.tensor([0.3], requires_grad=True)
        loss, stats = tr.report_term(z, torch.tensor([True]), torch.tensor([delta]), c_rep=2.0, z_max=6.0,
                                     denominator=4.0)  # fmt: skip
        loss.backward()
        assert z.grad.item() == pytest.approx(-2.0 * delta / 4.0) and stats[tr.METRIC_REP_N] == 1.0
    far = torch.tensor([9.2, -9.2], requires_grad=True)
    loss, stats = tr.report_term(far, torch.tensor([True, True]), torch.tensor([1.0, 1.0]), c_rep=1.0, z_max=6.0,
                                 denominator=1.0)  # fmt: skip
    loss.backward()
    assert far.grad.tolist() == [0.0, -1.0] and stats[tr.METRIC_REP_FROZEN] == 0.5  # only the one past z_max frozen


def test_rows_without_a_status_token_get_no_report_term():
    z = torch.zeros(2, requires_grad=True)
    loss, stats = tr.report_term(z, torch.tensor([False, False]), torch.tensor([1.0, -1.0]), c_rep=1.0, z_max=6.0,
                                 denominator=2.0)  # fmt: skip
    assert loss.item() == 0.0 and stats[tr.METRIC_REP_N] == 0.0


def test_a_long_report_before_status_is_still_found(tok):
    """Review R4: with blocker / completed fields first the status can sit hundreds of tokens after the call opens."""
    fields = "".join(
        f'<parameter=completed>\n["step {i} of a long list of things that were done"]\n</parameter>\n'
        for i in range(30)
    )
    text = ("Done.\n<tool_call>\n<function=submit_report>\n" + fields + "<parameter=status>\ncomplete\n</parameter>\n"
            "</function>\n</tool_call>")  # fmt: skip
    ids = tok(text, add_special_tokens=False).input_ids
    opener, closer = tr.tool_call_tokens(tok)
    j = tr.status_position(ids, None, C, I, _decode(tok), True, opener, closer)
    assert j > 200 and ids[j] == C
    assert tr.status_position(ids, None, C, I, _decode(tok), True) == -1  # the old 64-token window missed it
    bash = text.replace("submit_report", "bash")
    assert (
        tr.status_position(
            tok(bash, add_special_tokens=False).input_ids, None, C, I, _decode(tok), None, opener, closer
        )
        == -1
    )


def test_the_text_tokenizer_is_used_not_a_multimodal_processor(tok):
    """Review C1: Qwen3.5's processing_class is a processor whose __call__ takes images first."""
    from types import SimpleNamespace

    class Processor:
        tokenizer = tok

        def __call__(self, images=None, text=None, **kw):
            raise ValueError("images first")

    assert tr.text_tokenizer(SimpleNamespace(_tokenizer=None, processing_class=Processor())) is tok
    assert tr.text_tokenizer(SimpleNamespace(_tokenizer=tok, processing_class=Processor())) is tok
    from honest_agents.train.confidence_exact import StatusTokens

    t = StatusTokens.of(tr.text_tokenizer(SimpleNamespace(_tokenizer=None, processing_class=Processor())))
    assert (t.complete, t.incomplete) == (C, I)


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


def _stub(model, settings=None):
    from tests.test_trainer_loop import _loss_stub

    s = _loss_stub(model, trim=True, sort=True)
    s.accelerator.unwrap_model = lambda m: m
    if settings is not None:
        s.crgrpo, s.status_tokens = settings, (5, 6)
    return s


def _direct_status_z(model, batch):
    """z = logit(5) - logit(6) at each status row, from one plain full forward of the padded batch."""
    ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
    mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1)
    logits = model(input_ids=ids, attention_mask=mask).logits.float()
    width = batch["prompt_ids"].size(1)
    pos = batch["crgrpo_pos"]
    found = pos >= 0
    at = (width + pos.clamp(min=0) - 1).tolist()
    z = torch.stack([logits[r, at[r], 5] - logits[r, at[r], 6] for r in range(pos.size(0))])
    return z, found


@pytest.mark.parametrize("trim", [True, False])
def test_the_hooked_status_logits_equal_a_direct_forward(trim):
    from trl.trainer.utils import split_tensor_dict

    from tests.test_trainer_loop import _tiny_policy

    model = _tiny_policy()
    s = _stub(model)
    s.trim_padding = trim
    for batch in split_tensor_dict(dict(_crgrpo_round()), 4):
        ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1)
        mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1)
        status = tr.StatusLogits(model.get_output_embeddings(), batch["crgrpo_pos"], 5, 6)
        s._status = status
        with status:
            s._get_per_token_logps_and_entropies(
                model, ids, mask, batch["completion_ids"].size(1), compute_entropy=True
            )
        s._status = None
        z, found = status.values(batch["crgrpo_pos"].size(0), batch["crgrpo_delta"])
        z_ref, f_ref = _direct_status_z(model, batch)
        assert found.tolist() == f_ref.tolist()
        torch.testing.assert_close(z[found], z_ref[found], rtol=1e-4, atol=1e-4)


def test_the_loss_is_trls_plus_the_report_term_in_value_and_gradient():
    from trl.trainer.utils import split_tensor_dict

    from honest_agents import crgrpo as cr
    from tests.test_trainer_loop import _round_loss_and_grads, _tiny_policy, trainer_mod

    model = _tiny_policy()
    round_ = _crgrpo_round()
    ours = _stub(model, cr.RunSettings(c_rep=1.5))
    total, grads = _round_loss_and_grads(ours, model, round_)
    main, main_grads = _round_loss_and_grads(_stub(model), model, round_)  # crgrpo None: TRL's loss

    model.zero_grad()  # the report term alone, from an independent direct forward
    rep_total = 0.0
    for batch in trainer_mod.sort_micro_batches(split_tensor_dict(dict(round_), 4)):
        z, found = _direct_status_z(model, batch)
        rep, _ = tr.report_term(z, found, batch["crgrpo_delta"], c_rep=1.5, z_max=6.0, denominator=float(2 * 4))
        rep.backward()
        rep_total += float(rep.detach())
    rep_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert rep_total != 0.0
    assert total == pytest.approx(main + rep_total, rel=1e-4)
    for name, g in grads.items():
        torch.testing.assert_close(g, main_grads[name] + rep_grads[name], rtol=1e-3, atol=1e-5)
    assert ours._metrics["train"][tr.METRIC_REP_N] and ours._metrics["train"]["crgrpo/rep/hooked_share"][0] == 1.0


def test_the_gradient_ratio_is_measured_and_stops_the_run_past_its_limit():
    from types import SimpleNamespace

    from honest_agents import crgrpo as cr
    from tests.test_trainer_loop import _round_loss_and_grads, _tiny_policy

    model = _tiny_policy()
    s = _stub(model, cr.RunSettings(c_rep=1.5, grad_ratio_steps=1))
    s.state, s.args.report_to = SimpleNamespace(global_step=0), []
    s.current_gradient_accumulation_steps = 4  # 4 micro-batches of 2 = one optimizer step of the 8-row round
    _round_loss_and_grads(s, model, _crgrpo_round())
    ratio = s._metrics["train"][tr.METRIC_REP_GRAD_RATIO]
    assert len(ratio) == 1 and ratio[0] > 0
    s2 = _stub(model, cr.RunSettings(c_rep=1.5, grad_ratio_steps=1, grad_ratio_max=ratio[0] / 2))
    s2.state, s2.current_gradient_accumulation_steps = SimpleNamespace(global_step=0), 4
    s2.args.report_to = []
    with pytest.raises(RuntimeError, match="CR-GRPO stop"):
        _round_loss_and_grads(s2, model, _crgrpo_round())


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
    stub.__dict__.update(environments=envs, processing_class=tok, _tokenizer=tok, status_tokens=(C, I),
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


def test_rows_without_a_status_token_still_count_in_the_gradient_ratio():
    """Review: with one row per micro-batch, a row with no found status token (rep has no grad) must still add its
    main gradient and close the step on time, or the steps mix and the ratio leans up."""
    from types import SimpleNamespace

    from honest_agents import crgrpo as cr
    from tests.test_trainer_loop import _round_loss_and_grads, _tiny_policy

    model = _tiny_policy()
    round_ = _crgrpo_round()
    round_["crgrpo_pos"][:4] = -1  # the first micro-batches carry no status token at all
    s = _stub(model, cr.RunSettings(c_rep=1.5, grad_ratio_steps=2))
    s.state, s.args.report_to = SimpleNamespace(global_step=0), []
    s.current_gradient_accumulation_steps = 4
    _round_loss_and_grads(s, model, round_)
    assert len(s._metrics["train"][tr.METRIC_REP_GRAD_RATIO]) == 1  # one optimizer step, closed after 4 micro-batches
    assert "_ratio_buf" not in s.__dict__


def test_a_partial_step_is_closed_when_the_step_changes():
    from types import SimpleNamespace

    from honest_agents import crgrpo as cr
    from tests.test_trainer_loop import _round_loss_and_grads, _tiny_policy

    model = _tiny_policy()
    s = _stub(model, cr.RunSettings(c_rep=1.5, grad_ratio_steps=5))
    s.state, s.args.report_to = SimpleNamespace(global_step=0), []
    s.current_gradient_accumulation_steps = 8  # more than the round's 4 micro-batches: the step stays open
    _round_loss_and_grads(s, model, _crgrpo_round())
    assert s._metrics["train"][tr.METRIC_REP_GRAD_RATIO] == []
    s.state.global_step = 1
    _round_loss_and_grads(s, model, _crgrpo_round())
    assert len(s._metrics["train"][tr.METRIC_REP_GRAD_RATIO]) == 1  # step 0 closed (partial) when step 1 began


def test_an_unparsed_report_is_never_searched_for_a_status(tok):
    from collections import defaultdict
    from types import SimpleNamespace

    ids = torch.tensor([_rollout(tok, "complete")])
    envs = [SimpleNamespace(_crgrpo={"delta": 0.9, "v": 0.5}, _report=None)]
    stub = object.__new__(tr.HonestGRPOTrainer)
    stub.__dict__.update(environments=envs, processing_class=tok, _tokenizer=tok, status_tokens=(C, I),
                         model=SimpleNamespace(training=True), _metrics={"train": defaultdict(list)})  # fmt: skip
    out = {"completion_ids": ids, "completion_mask": torch.ones_like(ids), "tool_mask": torch.ones_like(ids)}
    stub._attach_report_signal(out)
    assert out["crgrpo_pos"].tolist() == [-1] and "crgrpo/rep/found_share_non_claims" not in stub._metrics["train"]
