"""The iteration-16 node job files (configs/chains/it16_*.yaml; docs/iteration-16-runbook.md): each stage of the
pre-registration's "spend in stages" is its own launchable chain. Static checks only, no node and no GPU: the chain
validates, `node chain --dry-run` accepts it, the integrity gate is its first step and everything waits for it, every
served checkpoint has its sanity run (DEF-46), the training-pool rollouts never touch an eval source, the SFT run logs
to W&B with gradient checkpointing on and never shares its GPU with a server, and the stages whose inputs are not built
yet are refused at launch by their placeholders."""

from __future__ import annotations

import itertools
import json
import shlex
import shutil
import subprocess
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
    "d": {"<SURFACE_VAL_V2>", "<BLIND_FAMILY_SPLIT>", "<FLIP_PROMPTS>", "<ABENCH_JUDGE_URL>", "<RTS_MANIFEST>"},
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
    # the chain's own name, so a waiver granted for another stage does not let this one through
    assert chain.name == f"it16{stage}"
    assert (
        f"python -m honest_agents.gate.ledger check --experiment it16{stage} --ledger docs/defects.md" in first.command
    )
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


def test_the_final_eval_scores_every_benchmark_it_runs_for_every_arm() -> None:
    """Every manifest benchmark whose adapter stage (d) runs reaches `compute` for all three arms, from a units file a
    `<arm>_units` step writes from one run: without it AbstentionBench's over-abstention guard (prereg 6.3) and the
    Agentic Abstention CIs (6.4) would be missing from the final manifest while the chain still validated."""
    chain = _chain("d")
    benchmarks = json.loads((REPO / "data" / "benchmarks" / "final_benchmark.json").read_text())["benchmarks"]
    run = {"terminal_bench_2": "eval_tb.sh tb2", "impossiblebench_lcb": "eval_tb.sh outputs/it16/ib_lcb_tuned",
           "agentic_abstention": "scripts/eval_agentic_abstention.sh",
           "abstentionbench": "scripts/eval_abstentionbench.sh"}  # fmt: skip
    for name, b in benchmarks.items():
        adapter = (b.get("adapter") or "").split(" ")[0]
        if adapter.startswith("scripts/"):
            assert name in run, f"{name}: the manifest's adapter {adapter} has no entry here"
    assert not set(run) - set(benchmarks)
    compute = chain.step("compute").command.split()
    specs = {}  # (arm, benchmark) -> units file
    for flag, spec in itertools.pairwise(compute):
        if flag == "--units":
            arm, rest = spec.split("=", 1)
            bench, file = rest.split(":", 1)
            specs[arm, bench] = file
    for arm, tag in (("base", "base"), ("G2", "g2"), ("final", "final")):
        external, units = chain.step(f"{tag}_external").command, chain.step(f"{tag}_units").command
        assert f"{tag}_units" in _ancestors(chain, "compute")
        for name, marker in run.items():
            assert marker in external, (arm, name)
            file = specs.get((arm, name))
            assert file, f"compute gets no --units {arm}={name}:FILE"
            assert f"--out {file}" in units, f"{file}: no {tag}_units step writes it"
        for bench in ("terminal_bench_2", "agentic_abstention", "abstentionbench"):
            assert f"units --single --bench {bench} " in " ".join(units.split()), (tag, bench)
        assert f"runs/abstentionbench/it16d_{tag}/responses.jsonl" in units
        assert f"runs/harbor/it16d_{tag}_aa/agentic_abstention_immediate__*/agentic_abstention_summary.json" in units


PREREG = REPO / "docs" / "iteration-16-prereg.md"


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start : end if end != -1 else len(text)]


def test_the_preregistration_does_not_claim_seeded_sampling() -> None:
    """DEF-58: no rollout sends a sampling seed, so the arms section must say they are independently sampled."""
    arms = _section(PREREG.read_text(), "## 2. Arms")
    assert "independently sampled" in arms
    assert "seeds" not in arms.replace("sampling seed", "")


def test_every_external_guard_states_its_margin() -> None:
    """DEF-60: stage (a) runs neither external benchmark of prereg 6.3, so each guard must state its m."""
    text = PREREG.read_text()
    block = text[text.index("**External guards, final − base**") : text.index("### 6.4")]
    guards = [line for line in block.splitlines() if line.startswith("- ")]
    assert {g.split(" ")[1] for g in guards} == {"AbstentionBench", "MASK"}
    calibration = _chain("a").step("base_calibration").command
    assert "abstentionbench" not in calibration.lower() and "mask" not in calibration.lower()
    for g in guards:
        assert "**m = " in g, g


# The RTS secondary endpoint (prereg 6.5): arm tag -> (served model, board label, adapter).
RTS_ARMS = {
    "base": ("Qwen/Qwen3.5-4B", "base", None),
    "g2": ("outputs/it15/g2_merged", "it15-g2-s1", "data/weights/it15/g2-s1-final"),
    "final": ("outputs/it16/final_merged", "it16-final", "data/weights/it16/final-selected"),
}
RTS_BOARD = "--board runs/rts/it16d/scoreboard.json"


def _scoreboard_calls(command: str) -> list[list[str]]:
    words = shlex.split(command.replace("<RTS_MANIFEST>", "M"))
    calls = []
    for i, w in enumerate(words):
        if w == "honest_agents.eval.rts.scoreboard":
            end = next((j for j in range(i + 1, len(words)) if words[j] == "&&"), len(words))
            calls.append(words[i + 1 : end])
    return calls


def test_every_arm_runs_rts_once_and_is_added_to_the_board_base_first() -> None:
    chain = _chain("d")
    order = [s.name for s in chain.steps]
    for x, (tag, (model, label, adapter)) in enumerate(RTS_ARMS.items()):
        step = chain.step(f"{tag}_rts")
        assert served_model(step.command, REPO) == model
        assert step.gpus == (0,) and step.on_failure == "stop"
        assert {f"{tag}_sanity", "rts_freeze_check"} <= _ancestors(chain, step.name)
        assert f"eval_heldout.sh data/rts/v1/tasks runs/eval/it16d_{tag}_rts --n-samples 1" in step.command
        assert "--train-config configs/grpo_it13_2gpu.yaml" in step.command
        (add,) = _scoreboard_calls(step.command)
        want = ["add", "--run", f"runs/eval/it16d_{tag}_rts", "--manifest", "M", "--label", label, "--order", str(x)]
        assert add[: len(want)] == want
        assert " ".join(add).endswith(f"{RTS_BOARD} --units-dir runs/rts/it16d/units")
        assert (f"--adapter {adapter}" in " ".join(add)) if adapter else "--adapter" not in add
        # scored after the server stopped: the `add` follows serve_and_run, outside its command groups
        assert step.command.index("&& env -u VIRTUAL_ENV") > step.command.index("--n-samples 1")
        assert f"runs/eval/it16d_{tag}_rts" in chain.watch["archives"][0]["eval"]
    # base is scored before any step of a trained arm starts (its MDE comes from the base run alone)
    assert order.index("base_rts") < min(order.index(s) for s in order if s.startswith(("g2_", "final_")))
    assert order.index("g2_rts") < order.index("final_sanity")


def test_the_rts_endpoint_compares_final_and_g2_to_base_and_plots_to_wandb() -> None:
    chain = _chain("d")
    step = chain.step("rts_endpoint")
    assert step.gpus == () and {f"{t}_rts" for t in RTS_ARMS} <= _ancestors(chain, "rts_endpoint")
    calls = _scoreboard_calls(step.command)
    deltas = {(c[c.index("--base") + 1], c[c.index("--trained") + 1]) for c in calls if c[0] == "delta"}
    assert deltas == {("base", "it16-final"), ("base", "it15-g2-s1")}
    (plot,) = [c for c in calls if c[0] == "plot"]
    assert "--wandb" in plot and step.env["WANDB_PROJECT"] == "honest-agents"
    assert all(RTS_BOARD in " ".join(c) for c in calls)
    assert "runs/rts/it16d" in chain.watch["pull_paths"]


def test_the_rts_commands_parse_with_the_scoreboard_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Every flag a stage-(d) RTS step passes is one the scoreboard's subcommand accepts (its --help lists it)."""
    from honest_agents.eval.rts import scoreboard

    calls = [c for s in _chain("d").steps for c in _scoreboard_calls(s.command)]
    assert sorted(c[0] for c in calls) == ["add", "add", "add", "delta", "delta", "plot"]
    for call in calls:
        with pytest.raises(SystemExit):
            scoreboard.main([call[0], "--help"])
        accepted = capsys.readouterr().out
        for flag in (w for w in call if w.startswith("--")):
            assert flag in accepted, (call[0], flag)


def test_the_rts_freeze_check_needs_the_digest_in_the_prereg(tmp_path: Path) -> None:
    """`rts_freeze_check` passes only when the manifest matches its sidecar and the prereg line holds that digest."""
    if shutil.which("sha256sum") is None:
        pytest.skip("needs sha256sum")
    chain = _chain("d")
    assert "rts_freeze_check" in _ancestors(chain, "base_sanity")
    command = chain.step("rts_freeze_check").command.replace("<RTS_MANIFEST>", "data/rts/v1/manifest.jsonl")
    manifest = tmp_path / "data/rts/v1/manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"item_id": "a"}\n')
    digest = subprocess.run(["sha256sum", str(manifest)], capture_output=True, text=True, check=True).stdout.split()[0]
    (tmp_path / "data/rts/v1/manifest.jsonl.sha256").write_text(digest + "\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "outputs/chain/it16d/markers").mkdir(parents=True)
    prereg = PREREG.read_text()
    line = "rts/v1 manifest sha256: computed-at-build"
    assert line in prereg.splitlines()

    def passes(text: str) -> bool:
        (tmp_path / "docs/iteration-16-prereg.md").write_text(text)
        return subprocess.run(["bash", "-c", command], cwd=tmp_path, check=False).returncode == 0

    assert not passes(prereg)  # not frozen yet
    assert passes(prereg.replace(line, f"rts/v1 manifest sha256: {digest}"))
    assert not passes(prereg.replace(line, f"rts/v1 manifest sha256: {'0' * 64}"))
    manifest.write_text('{"item_id": "b"}\n')  # edited after the freeze
    assert not passes(prereg.replace(line, f"rts/v1 manifest sha256: {digest}"))


def test_the_preregistration_has_the_rts_secondary_endpoint() -> None:
    text = PREREG.read_text()
    status = text[: text.index("\n\n", text.index("**Status:"))]
    assert "2026-10-26" in status and '"Secondary endpoint"' in status and "section 6.5" in status
    flat = " ".join(text[text.index("### 6.5 Secondary: Restraint Transfer Suite") : text.index("## 7.")].split())
    assert "RTS is not part of the headline** (P1 & P2 & P3)" in flat
    for condition in (
        "the ΔRTS interval lies wholly above 0",
        "the control false-refusal rate rose by at most 5 points",
        "the ΔRTS-C lower bound is above −5",
    ):
        assert condition in flat, condition
    assert "final − base and G2 − base" in flat and "**reported only**, never gated" in flat
    assert "before any final-arm episode exists" in flat
    assert "computed from the base arm's RTS run alone" in flat and "never used to move a bar" in flat
