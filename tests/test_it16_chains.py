"""The iteration-16 node job files (configs/chains/it16_*.yaml; docs/iteration-16-runbook.md): each stage of the
pre-registration's "spend in stages" is its own launchable chain. Static checks only, no node and no GPU: the chain
validates, `node chain --dry-run` accepts it, the integrity gate is its first step and everything waits for it, every
served checkpoint has its sanity run (DEF-46), the training-pool rollouts never touch an eval source, the SFT run logs
to W&B with gradient checkpointing on and never shares its GPU with a server, and the stages whose inputs are not built
yet are refused at launch by their placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents import node as nodemod
from honest_agents.node_chain import Chain, load_chain, placeholders, sanity_problems, served_model

REPO = Path(__file__).resolve().parent.parent
CHAINS = REPO / "configs" / "chains"
STAGES = {
    "a": "it16_a_calibration.yaml",
    "b": "it16_b_claim_rollouts.yaml",
    "c": "it16_c_sft.yaml",
    "d": "it16_d_final_eval.yaml",
}
# What is not built yet, by stage (a placeholder blocks the launch until it is filled in).
PENDING = {
    "a": set(),
    "b": set(),
    "c": {"<SFT_CONFIG>", "<SFT_BUILD_MANIFEST>", "<SELECT_PROTOCOL>"},
    "d": {"<SURFACE_VAL_V2>", "<BLIND_FAMILY_SPLIT>", "<FLIP_PROMPTS>", "<ABENCH_JUDGE_URL>"},
}


def _chain(stage: str) -> Chain:
    return load_chain(CHAINS / STAGES[stage])


def _ancestors(chain: Chain, name: str) -> set[str]:
    seen: set[str] = set()
    todo = list(chain.step(name).after)
    while todo:
        a = todo.pop()
        if a not in seen:
            seen.add(a)
            todo += chain.step(a).after
    return seen


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_each_stage_validates_and_dry_runs(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    chain = _chain(stage)
    assert sanity_problems(chain, REPO) == []
    assert nodemod.main(["chain", "--host", "h", "--no-secrets", "--dry-run", str(CHAINS / STAGES[stage])]) == 0
    assert f"node_chain run configs/chains/{STAGES[stage]}" in capsys.readouterr().out
    assert len({c.name for c in map(_chain, STAGES)}) == len(STAGES)  # own state dirs: stages are launched apart
    assert chain.state_dir == f"outputs/chain/{chain.name}" and chain.balance_floor is not None


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_the_integrity_gate_is_the_first_step_and_everything_waits_for_it(stage: str) -> None:
    chain = _chain(stage)
    first = chain.steps[0]
    assert first.name == "gate_ledger" and first.after == () and first.on_failure == "stop"
    assert "python -m honest_agents.gate.ledger check --ledger docs/defects.md" in first.command
    for step in chain.steps[1:]:
        assert "gate_ledger" in _ancestors(chain, step.name), step.name


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_placeholders_block_exactly_the_stages_whose_inputs_are_not_built(stage: str) -> None:
    left = {p.split(": ", 1)[1] for p in placeholders(_chain(stage))}
    assert left == PENDING[stage]
    if left:  # `node chain` refuses before any ssh, so this sends nothing
        assert nodemod.main(["chain", "--host", "h", "--no-secrets", str(CHAINS / STAGES[stage])]) == 2


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_every_served_checkpoint_is_a_rebuilt_arm_and_the_gpu_has_one_tenant(stage: str) -> None:
    chain = _chain(stage)
    allowed = {"Qwen/Qwen3.5-4B", "outputs/it15/g2_merged", "outputs/it16/final_e1_merged",
               "outputs/it16/final_e2_merged", "outputs/it16/final_merged"}  # fmt: skip
    for s in chain.steps:
        model = served_model(s.command, REPO)
        trains = "run_sft.sh" in s.command and "DRY_RUN=1 scripts/run_sft.sh" not in s.command
        # one GPU; steps with overlapping GPU sets never run together, so no server runs beside the trainer
        assert s.gpus == ((0,) if model is not None or trains else ()), s.name
        if model is not None:
            assert model in allowed, (s.name, model)
            assert s.env.get("VLLM_GPUS") == "0" and s.env.get("BASE_URL") == "http://localhost:8000/v1"
        if trains:
            assert s.env.get("CUDA_VISIBLE_DEVICES") == "0"


def test_calibration_runs_base_and_g2_on_tb2_and_impossiblebench_at_two_samples() -> None:
    chain = _chain("a")
    for arm, model in (("base", "Qwen/Qwen3.5-4B"), ("g2", "outputs/it15/g2_merged")):
        step = chain.step(f"{arm}_calibration")
        assert served_model(step.command, REPO) == model
        assert f"eval_tb.sh tb2 runs/harbor/it16a_{arm}_tb2 -k 2" in step.command
        assert f"eval_tb.sh outputs/it16/ib_lcb_tuned runs/harbor/it16a_{arm}_ib -k 2" in step.command
        assert f"{arm}_sanity" in _ancestors(chain, step.name)
    assert "--prompt tuned" in chain.step("ib_tasks").command and "MANIFEST.sha256" in chain.step("ib_tasks").command


def test_claim_rollouts_use_the_training_pool_list_and_keep_transcripts() -> None:
    chain = _chain("b")
    roll = chain.step("claim_rollouts")
    assert (
        "eval_heldout.sh data/decision/read_rft_tasks.jsonl runs/eval/it16_claim_rollouts --n-samples 4" in roll.command
    )
    assert "--bank-dir runs/eval/it16_claim_rollouts/bank" in roll.command  # full transcripts (the bank)
    assert "split_guard" in _ancestors(chain, "claim_rollouts")
    assert "read_rft_tasks verify --out data/decision/read_rft_tasks.jsonl" in chain.step("split_guard").command
    (archive,) = chain.watch["archives"]
    assert archive["keep_bank"] is True and "runs/eval/it16_claim_rollouts" in archive["eval"]
    assert "runs/eval/it16_claim_rollouts" in chain.watch["pull_paths"]
    # every task of the list has a requirement checklist (the claim build needs one) and is a training row
    rows = [json.loads(x) for x in (REPO / "data/decision/read_rft_tasks.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 1600 // 4 and {r["split"] for r in rows} == {"train"}
    checklists = {p.stem for p in (REPO / "data/commit_checks/checklists").glob("*.json")}
    assert {r["variant_id"] for r in rows} <= checklists


def test_claim_rollout_sources_are_not_excluded_from_training() -> None:
    """The prereg's assertion (section 3.1) on the stage-(b) list: no select, val or test source (needs the main tree's
    untracked split files; skipped where they are absent)."""
    from honest_agents.tasks.select_split import SELECT_SOURCES_FILE, excluded_from_training

    data = REPO / "data"
    if not (data / SELECT_SOURCES_FILE).is_file() or not (data / "val.jsonl").is_file():
        pytest.skip("needs the untracked split files under data/ (main tree)")
    rows = [json.loads(x) for x in (data / "decision/read_rft_tasks.jsonl").read_text().splitlines() if x.strip()]
    assert not {r["pair_id"] for r in rows} & excluded_from_training(data)


def test_the_sft_run_logs_to_wandb_with_gradient_checkpointing_and_selection_stops_on_no_pass() -> None:
    chain = _chain("c")
    train = chain.step("train_final")
    for flag in ("model.name=outputs/it15/g2_merged", "train.gradient_checkpointing=true", "train.report_to=wandb",
                 "run_name=it16-commit-checks", "train.extra.save_strategy=epoch", "train.num_train_epochs=2"):  # fmt: skip
        assert flag in train.command and flag in chain.step("data_gates").command, flag
    assert train.env["WANDB_PROJECT"] == "honest-agents" and train.env["WANDB_RUN_GROUP"] == "it16"
    assert "data_gates" in _ancestors(chain, "train_final")
    assert "merge_epochs.sh outputs/sft/it16-commit-checks outputs/it16/final outputs/it15/g2_merged 2" in (
        chain.step("merge_epochs").command
    )
    sel = chain.step("select_final").command
    # no `|| true` or fallback: a select_checkpoint exit 1 (no epoch passes the gate) fails the step
    assert "--require-split select" in sel and "||" not in sel and "--status 0" in sel
    assert sel.index("select_checkpoint") < sel.index("epochs decide") < sel.index("epochs best")
    assert chain.step("select_final").done == "outputs/sft/it16-commit-checks/best_checkpoint.json"


def test_the_final_eval_runs_every_arm_once_on_every_benchmark() -> None:
    chain = _chain("d")
    for arm in ("base", "g2", "final"):
        assert f"{arm}_sanity" in _ancestors(chain, f"{arm}_external")
        internal, external = chain.step(f"{arm}_internal").command, chain.step(f"{arm}_external").command
        for split in ("<SURFACE_VAL_V2>", "data/val.jsonl", "data/probe_v1.jsonl", "<BLIND_FAMILY_SPLIT>"):
            assert f"eval_heldout.sh {split} runs/eval/it16d_{arm}_" in internal and "--n-samples 4" in internal
        assert internal.count("decision_probe") == 2 and "--n 8" in internal
        assert f"eval_tb.sh tb2 runs/harbor/it16d_{arm}_tb2 -k 4" in external
        assert f"runs/harbor/it16d_{arm}_ib -k 4" in external and "N_ATTEMPTS=4" in external
    assert "data/weights/it16/final-selected" in chain.step("merge_final").command
    compute = chain.step("compute").command
    assert all(f"{arm}=terminal_bench_2:" in compute and f"{arm}=impossiblebench_lcb:" in compute
               for arm in ("base", "G2", "final"))  # fmt: skip
