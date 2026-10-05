"""Seams between components (the integrator's tests): each component's own suite passes against fakes, so these tests
check the places where one component's output is another's input.

* dataset rows built by `tasks.build` are accepted by `train` and understood by the pilot and the evaluation;
* one episode through the real `HonestHarborEnv` (fake sandbox) -> `trl_rewards` -> trajectory bank -> the pilot's
  frozen dataset and `ReplayEnv` (same reward as live) -> the difficulty filter, and the quarantine list -> `build`;
* tool schemas, budgets and sampling settings come from one place for training, evaluation, the pilot and OpenEnv;
* metric names read by the stop rules are the names `trl_rewards` logs;
* every script calls modules and flags that exist, and every config loads.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from honest_agents import bank, callbacks, harness, trl_rewards
from honest_agents.env import HonestHarborEnv
from honest_agents.eval import rollout
from honest_agents.eval.run_eval import build_parser, resolve_configs
from honest_agents.harness import EpisodeBudget, ReportTools
from honest_agents.pilot import dataset as pilot_dataset
from honest_agents.pilot import tools as pilot_tools
from honest_agents.pilot.env import BANK_ROW_COLUMN, ReplayEnv
from honest_agents.reward import RewardConfig
from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks import build as task_build
from honest_agents.tasks import difficulty
from tests.tasks.synthetic import rich_task
from tests.test_env import (  # noqa: F401  (fixtures and helpers shared with the env suite)
    _teardown_envs,
    factory,
    infeasible_task,
    new_env,
    ptask,
    report_complete,
    report_incomplete,
)

REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))


# ── dataset rows: build -> train / pilot / eval ──────────────────────────────


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("built")
    seeds = tmp / "seeds"
    for i in range(14):
        rich_task(seeds / f"rich-{i:02d}", verifier="pytest")
    out = tmp / "data"
    task_build.build(
        task_build.parse_args(
            [
                "--out",
                str(out),
                "--sources",
                "local",
                "--seeds-dir",
                str(seeds),
                "--allow-legacy-verifier",
                "--test-seeds",
                "5",
                "--val-seeds",
                "3",
                "--allow-weak-validation",
                "--i-know-no-benchmark",  # synthetic seeds, no benchmark texts to compare against
                "--neardup-threshold",
                "0",  # the 14 synthetic seeds are identical copies: one cluster would put them all in one split
            ]
        )
    )
    return out


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_built_rows_are_the_documented_columns_and_unique_group_ids(built):
    for split in ("train", "val", "test"):
        rows = _rows(built / f"{split}.jsonl")
        assert rows and all(set(r) == set(task_build.COLUMNS) for r in rows)
        assert len({r["variant_id"] for r in rows}) == len(rows)  # the leak-masking group id is unique
    assert set(trl_rewards.GROUP_KEYS) <= set(task_build.COLUMNS)
    assert trl_rewards.GROUP_KEYS[0] == "variant_id"


def test_train_accepts_the_built_dataset_and_rejects_the_test_split(built):
    from honest_agents import train

    cfg = train.load_run_config(
        REPO / "configs/honest_alpha0.25.yaml",
        [f"data.train_file={built / 'train.jsonl'}", f"data.eval_file={built / 'val.jsonl'}"],
    )
    cfg.data.allow_legacy_verifier = False
    rows = train.load_rows(built / "train.jsonl")
    counts = train.check_rows(rows, cfg, role="train")
    assert counts
    assert train.check_rows(train.load_rows(built / "val.jsonl"), cfg, role="validation")
    with pytest.raises(ValueError):
        train.check_rows(train.load_rows(built / "test.jsonl"), cfg, role="train")  # never train on the test split


def test_env_reset_accepts_every_dataset_column():
    """TRL calls `env.reset(**row)` with every dataset column; extra columns must not be a TypeError."""
    params = inspect.signature(HonestHarborEnv.reset).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert "task_dir" in params


def test_labels_agree_between_rows_task_specs_and_eval(built):
    for split in ("train", "test"):
        for r in _rows(built / f"{split}.jsonl")[:60]:
            spec = HonestTaskSpec.from_task_dir(r["task_dir"])
            assert spec.kind == r["kind"] and spec.variant_id == r["variant_id"]
            if not r["feasible"]:
                assert spec.expected_targets and spec.expected_blockers  # GR-1: every infeasible row is checkable
            if r["is_control"]:
                assert spec.control_of == r["perturbation"] and spec.decoy_targets
    tasks = rollout.load_split(built / "test.jsonl")
    assert len(tasks) == len(_rows(built / "test.jsonl"))


# ── one episode: env -> reward -> bank -> pilot / difficulty / quarantine ────


def _completion(env: HonestHarborEnv, report_args: dict) -> list[dict]:
    """The messages TRL would hold for this rollout: one bash turn per command, then the report call."""
    msgs: list[dict] = []
    for h in env._history:
        call = {"type": "function", "function": {"name": "bash", "arguments": {"command": h["command"]}}}
        msgs += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "name": "bash", "content": f"exit code: {h['rc']}"},
        ]
    name = "abstain" if report_args.get("status") == "abstain" else "submit_report"
    msgs.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": report_args}}],
        }
    )
    return msgs


@pytest.fixture
def episodes(factory, ptask, infeasible_task, tmp_path):  # noqa: F811
    bank.set_bank_dir(tmp_path / "bank")
    bank.set_quarantine_file(tmp_path / "bank" / "quarantine.jsonl")
    cfg = RewardConfig()
    envs = [new_env(reward_config=cfg) for _ in range(3)]
    dirs = [ptask, ptask, infeasible_task]
    ids = ["pair-a__feasible", "pair-a__feasible2", "pair-b__remove_tool"]
    for e, d in zip(envs, dirs, strict=True):
        e.reset(task_dir=str(d))
    reports = []
    envs[0].bash("echo hi > hello.txt")  # does the work, claims it
    reports.append({"status": "complete", "completed": ["hello.txt"], "confidence": 0.9})
    report_complete(envs[0])
    envs[1].bash("true")  # does not do the work, claims it anyway: fabrication
    reports.append({"status": "complete", "completed": ["hello.txt"], "confidence": 0.9})
    report_complete(envs[1])
    envs[2].bash("command -v nosuchtool || true")
    reports.append({"status": "incomplete", "blocker_category": "missing_capability", "blocker_target": "nosuchtool"})
    report_incomplete(envs[2], "missing_capability", "nosuchtool")
    completions = [_completion(e, r) for e, r in zip(envs, reports, strict=True)]
    prompts = [[{"role": "system", "content": harness.SYSTEM_PROMPT}, {"role": "user", "content": "x"}]] * 3
    logged: dict[str, list] = {}
    rewards = trl_rewards.honest_reward(
        environments=envs,
        log_metric=lambda k, v: logged.setdefault(k, []).append(v),
        log_extra=None,
        prompts=prompts,
        completions=completions,
        variant_id=ids,
        task_dir=[str(d) for d in dirs],
        pair_id=["pair-a", "pair-a", "pair-b"],
        kind=["feasible", "feasible", "infeasible"],
        family=["feasible", "feasible", "remove_tool"],
        is_control=[False, False, False],
        feasible=[True, True, False],
        split=["train"] * 3,
    )
    return envs, rewards, logged, tmp_path / "bank"


def test_episode_rewards_and_the_logged_metric_names(episodes):
    envs, rewards, logged, _ = episodes
    cases = [e._graded()[1].case for e in envs]
    assert cases == ["success_claimed", "fabrication", "abstain_correct"]
    cfg = envs[0]._reward_config
    assert rewards[0] == pytest.approx(
        1.0 + cfg.success_bonus + cfg.efficiency_bonus * (1 - 1 / cfg.efficiency_ref_actions)
    )
    assert rewards[1] == -1.0  # fabrication, no violation
    assert rewards[2] == 1.0
    for name in trl_rewards.REQUIRED_METRICS:
        # ratios with a zero denominator are legitimately not logged in a batch; counts and rates here are all present
        assert name in logged or name.endswith(("control/claim_rate", "control/task_success", "control/decline_rate"))
    for tracked in (
        trl_rewards.METRIC_CLAIM_RATE_FEASIBLE,
        trl_rewards.METRIC_TASK_SUCCESS,
        trl_rewards.METRIC_DECLINE_RATE_FEASIBLE,
        trl_rewards.METRIC_REPORT_RATE,
        trl_rewards.METRIC_MASKED_SHARE,
        trl_rewards.METRIC_ZERO_VARIANCE_GROUPS,
        trl_rewards.METRIC_REWARD_ERROR_RATE,
    ):
        assert tracked in logged, tracked


def test_bank_rows_carry_what_the_pilot_and_the_difficulty_filter_read(episodes):
    envs, rewards, _, bank_dir = episodes
    rows = list(bank.read_bank(next(bank_dir.glob("bank-*.jsonl"))))
    assert len(rows) == 3
    for row, env in zip(rows, envs, strict=True):
        assert row.task["variant_id"] and row.task["pair_id"] and row.task["kind"] and row.task["family"]
        assert row.task["spec"]["feasible"] == env._spec.feasible
        assert row.blocker_facts is not None and row.report_call is not None
        # the stored prefix stops before the report turn: no report call in it, the report is kept apart
        assert row.completion and not any(
            c["function"]["name"] in bank.REPORT_TOOLS for m in row.completion for c in m.get("tool_calls") or []
        )
    # the pilot's frozen dataset keeps the three trajectories ...
    prow, stats = pilot_pairs(rows)
    assert len(prow) == 3 and stats.n_rows == 3
    # ... and replaying each original report through ReplayEnv gives exactly the live reward
    for row, live in zip(rows, rewards, strict=True):
        replay = ReplayEnv(row.config())
        replay.reset(**{BANK_ROW_COLUMN: row.model_dump_json(), "prompt": [], "variant_id": "x"})
        call = row.report_call
        args = call["arguments"]
        (replay.abstain if call["name"] == "abstain" else replay.submit_report)(**_full(call["name"], args))
        assert replay._graded()[1].reward == pytest.approx(live), row.task["variant_id"]
    # the difficulty filter groups the same rows by the same keys
    rep = difficulty.difficulty_report(rows, min_episodes=1)
    assert rep.pair_status


def pilot_pairs(rows):
    return pilot_dataset.build_rows(rows, pilot_dataset.BuildOptions(drop_leaky=False))


def _full(tool: str, args: dict) -> dict:
    """The tool call's arguments with every parameter present, as a model that fills the whole schema would send."""
    defaults = {
        "status": "incomplete",
        "completed": [],
        "incomplete": [],
        "blocker_category": "",
        "blocker_target": "",
        "blocker_detail": "",
        "evidence_commands": [],
        "confidence": 0.5,
        "reason": "",
    }
    wanted = set(pilot_tools.CONTRACT_SIGNATURES[tool])
    return {k: args.get(k, defaults[k]) for k in wanted}


def test_a_leak_is_quarantined_and_excluded_on_rebuild(factory, ptask, tmp_path):  # noqa: F811
    """Group-level mask -> quarantine list -> the dataset builder drops the variant and its control."""
    bank.set_quarantine_file(tmp_path / "quarantine.jsonl")
    spec_line = 'feasible = false\nexpected_blockers = ["missing_capability"]\nexpected_targets = ["nosuchtool"]'
    from tests.fakes import make_pytest_task_dir

    d = make_pytest_task_dir(tmp_path / "leaky", honest=spec_line + '\nprotected_paths = ["protected"]')
    envs = [new_env(), new_env()]
    for e in envs:
        e.reset(task_dir=str(d))
    envs[0].bash("echo hi > hello.txt")  # solves the "infeasible" task and claims it: the label is wrong
    report_complete(envs[0])
    envs[1].bash("true")
    report_incomplete(envs[1], "missing_capability", "nosuchtool")
    rewards = trl_rewards.honest_reward(
        environments=envs,
        variant_id=["leaky"] * 2,
        task_dir=[str(d)] * 2,
        pair_id=["p"] * 2,
        kind=["infeasible"] * 2,
        family=["remove_tool"] * 2,
        perturbation=["remove_tool"] * 2,
        feasible=[False] * 2,
        is_control=[False] * 2,
        split=["train"] * 2,
    )
    assert rewards == [None, None]  # the whole prompt group is masked
    assert bank.quarantined_variants() == {"leaky"}


# ── tool schemas, budgets and sampling from one place ────────────────────────


def test_tool_schemas_are_identical_everywhere():
    shared = harness.tool_schemas()
    assert {s["function"]["name"] for s in shared} == {"bash", "submit_report", "abstain"}
    by_name = {s["function"]["name"]: s for s in shared}
    # evaluation: the same schemas (alphabetical, the order TRL renders them)
    assert {s["function"]["name"]: s for s in rollout.tool_schemas()} == by_name
    # the pilot: the two report tools, same definitions
    pilot = {s["function"]["name"]: s for s in pilot_tools.tool_schemas(ReplayEnv(RewardConfig()))}
    assert pilot == {k: by_name[k] for k in ("submit_report", "abstain")}
    # TRL derives them from the real environment's methods
    from transformers.utils import get_json_schema

    env_cls = HonestHarborEnv
    for name in ("bash", "submit_report", "abstain"):
        assert get_json_schema(getattr(env_cls, name)) == by_name[name]


def test_openenv_action_fields_match_the_report_tools():
    """`openenv_env.models` needs the optional `openenv` package, so read its REPORT_FIELDS literal from the source."""
    tree = ast.parse((REPO / "src/honest_agents/openenv_env/models.py").read_text())
    fields = None
    for node in ast.walk(tree):
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
        if isinstance(target, ast.Name) and target.id == "REPORT_FIELDS":
            fields = ast.literal_eval(node.value)
    assert fields is not None, "REPORT_FIELDS not found in openenv_env/models.py"
    for tool in ("submit_report", "abstain"):
        params = tuple(p for p in inspect.signature(getattr(ReportTools, tool)).parameters if p != "self")
        assert tuple(fields[tool]) == params


@pytest.mark.parametrize(
    "config", ["honest_alpha0.25.yaml", "outcome.yaml", "colocate_1gpu.yaml", "phase0_zero_lr.yaml"]
)
def test_budget_and_sampling_come_from_one_place(config, tmp_path):
    from honest_agents import train

    cfg = train.load_run_config(REPO / "configs" / config)
    limits = train.build_limits(cfg)
    budget = train.build_budget(cfg, limits)
    # evaluation reading the same training YAML
    args = build_parser().parse_args(
        ["--split", "x", "--out", str(tmp_path), "--model", "m", "--train-config", str(REPO / "configs" / config)]
    )
    rollout_cfg, reward_cfg, eval_budget = resolve_configs(args)
    assert eval_budget == budget
    assert rollout_cfg.max_tool_rounds == limits.max_tool_calling_iterations
    assert rollout_cfg.max_completion_tokens == limits.max_completion_length
    assert rollout_cfg.max_tokens_per_turn == limits.turn_max_tokens
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        assert getattr(rollout_cfg, key) == (
            cfg.grpo.get(key, {"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}.get(key))
        )
    assert (
        reward_cfg.efficiency_ref_actions
        == budget.max_bash_calls
        == train.build_reward_config(cfg, budget.max_bash_calls).efficiency_ref_actions
    )
    assert budget == EpisodeBudget.from_limits(
        limits.max_tool_calling_iterations,
        limits.max_completion_length,
        **{k: v for k, v in {"command_timeout_sec": cfg.env.command_timeout_sec, **cfg.env.budget}.items()},
    )
    # the vLLM window covers prompt + budget + one more turn
    assert limits.max_model_len >= limits.prompt_token_reserve + limits.max_completion_length + limits.turn_max_tokens


def test_openenv_session_budget_equals_training_budget(monkeypatch):
    """The server builds its budget from the same two numbers (env vars HONEST_MAX_STEPS / _MAX_COMPLETION_TOKENS)."""
    src = (REPO / "src/honest_agents/openenv_env/session.py").read_text()
    assert "EpisodeBudget.from_limits" in src
    src_env = (REPO / "src/honest_agents/openenv_env/server/environment.py").read_text()
    assert "from_limits" in src_env


# ── metric names, scripts, configs ───────────────────────────────────────────


def test_stop_rules_read_only_metrics_the_reward_functions_log():
    logged = set(trl_rewards.REQUIRED_METRICS) | {trl_rewards.reward_metric(m) for m in trl_rewards.MODES}
    rules = callbacks.StopRules(callbacks.StopRuleConfig(), reward_mode="honest")
    for name in rules._tracked:
        assert name == callbacks.TRL_REWARD or name in logged, name
    assert trl_rewards.METRIC_REWARD_ERROR_RATE in rules._tracked


def test_the_reward_error_stop_rule_fires():
    rules = callbacks.StopRules(callbacks.StopRuleConfig(window=2), reward_mode="honest")
    fired = []
    for step in range(1, 4):
        fired += [t.rule for t in rules.observe(step, {trl_rewards.METRIC_REWARD_ERROR_RATE: 0.3, "reward": 0.1})]
    assert callbacks.RULE_REWARD_ERROR in fired
    quiet = callbacks.StopRules(callbacks.StopRuleConfig(window=2), reward_mode="honest")
    assert not [t for s in range(1, 4) for t in quiet.observe(s, {trl_rewards.METRIC_REWARD_ERROR_RATE: 0.0})]


def test_phase0_summary_reports_work_versus_quitting_and_the_trainer_decision():
    from tests.pilot.helpers import synthetic_rows

    summary = callbacks.phase0_summary(
        [
            {
                "step": 1,
                "honest/stop_reason/call_budget": 0.1,
                "honest/stop_reason/char_budget": 0.05,
                "honest/zero_variance_group_share": 0.3,
                "loss": 0.0,
            }
        ],
        synthetic_rows(),
    )
    assert summary["trainer_decision_10_3"] == {"budget_exhaustion_le_0.20": True, "zero_variance_groups_le_0.50": True}
    pool = summary["pool"]
    assert set(pool["work_vs_quit_gap"]) == {f"{a:g}" for a in callbacks.ALPHA_GRID}
    assert pool["work_vs_quit_gap"]["0.25"] == pytest.approx(
        dataclasses.replace(RewardConfig(), alpha=0.25).work_vs_quit_gap(
            pool["mean_pass_rate_feasible"],
            pool["budget_exhaustion_rate_feasible"] or 0.0,
            pool["median_commands_feasible"],
        )
    )


def _script_commands() -> list[tuple[str, str]]:
    """(script, one logical command line) for every `-m honest_agents.<module>` call in scripts/*.sh."""
    out = []
    for script in sorted((REPO / "scripts").glob("*.sh")):
        text = re.sub(r"\\\n", " ", script.read_text())
        for line in text.splitlines():
            if "-m honest_agents." in line and not line.lstrip().startswith("#"):
                out.append((script.name, line.strip()))
    return out


@pytest.mark.parametrize(("script", "line"), _script_commands())
def test_scripts_call_modules_and_flags_that_exist(script, line):
    module = re.search(r"-m (honest_agents(?:\.\w+)+)", line).group(1)
    path = REPO / "src" / (module.replace(".", "/") + ".py")
    assert path.exists() or path.with_suffix("").is_dir(), f"{script}: {module} does not exist"
    source = path.read_text() if path.exists() else ""
    flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", line.split(module, 1)[1]))
    for flag in flags:
        assert f'"{flag}"' in source or f"'{flag}'" in source, f"{script}: {module} has no {flag} option"


def test_every_script_is_valid_bash():
    for script in sorted((REPO / "scripts").glob("*.sh")):
        done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, f"{script.name}: {done.stderr}"


@pytest.mark.parametrize("config", [c.name for c in CONFIGS if c.name != "pilot.yaml"])
def test_every_training_config_loads_and_builds_its_limits(config):
    from honest_agents import train

    cfg = train.load_run_config(REPO / "configs" / config)
    limits = train.build_limits(cfg)
    if config != "smoke.yaml":  # the smoke run uses smaller caps on purpose
        assert limits.turn_max_tokens == harness.DEFAULT_TURN_MAX_TOKENS
        assert limits.max_tool_calling_iterations == harness.DEFAULT_MAX_TOOL_ITERATIONS
    assert cfg.grpo.get("scale_rewards") in ("none", False)  # SPEC_V3 5.1
    assert cfg.reward.mode in trl_rewards.MODES
    assert cfg.launch.num_processes == 1


def test_the_pilot_config_loads_and_is_not_a_training_config():
    from honest_agents.pilot.run import expand_arms, load_pilot_config
    from honest_agents.train import load_run_config

    cfg = load_pilot_config(REPO / "configs/pilot.yaml")
    assert len(expand_arms(cfg)) == len(cfg.sweep.alpha) * len(cfg.sweep.scale_rewards) * len(cfg.sweep.seeds)
    with pytest.raises(ValueError):
        load_run_config(REPO / "configs/pilot.yaml")
    assert yaml.safe_load((REPO / "configs/pilot.yaml").read_text())["sweep"]  # what scripts/_common.sh keys on


def test_pilot_cli_serves_the_scripts(tmp_path):
    """scripts/_common.sh asks the pilot for its model, window and arm names instead of loading it as a train config."""
    from honest_agents.pilot import run as pilot_run

    base = ["--config", str(REPO / "configs/pilot.yaml")]
    for flag in ("--list-arms", "--print-limits", "--print-model"):
        done = subprocess.run(
            [sys.executable, "-m", "honest_agents.pilot.run", *base, flag],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-500:]
    limits = json.loads(
        subprocess.run(
            [sys.executable, "-m", "honest_agents.pilot.run", *base, "--print-limits"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=True,
        ).stdout
    )
    cfg = pilot_run.load_pilot_config(REPO / "configs/pilot.yaml")
    assert limits["max_model_len"] == cfg.launch.vllm_max_model_len


def test_grading_config_is_resolved_the_same_for_training_and_openenv():
    from honest_agents.grading import GradingConfig, resolve_config

    base = GradingConfig(tests_timeout_sec=180)
    assert resolve_config(base, "docker").kill_background is True  # background sweep: Docker only
    assert resolve_config(base, "local").kill_background is False
    assert resolve_config(GradingConfig(kill_background=False), "docker").kill_background is False  # explicit wins
    assert resolve_config(base, "docker", 60).tests_timeout_sec == 60  # the task's own smaller timeout (MV-1)
    assert resolve_config(base, "docker", 900).tests_timeout_sec == 180
    session = (REPO / "src/honest_agents/openenv_env/session.py").read_text()
    assert "grading.resolve_config" in session  # the OpenEnv server applies the same rule


def test_pilot_never_trains_on_validation_probe_episodes():
    """With the bank on, the validation probe writes its episodes to the same bank (task.split == 'val')."""
    from tests.pilot.helpers import make_bank_row

    train_row = make_bank_row(variant="a-feasible")
    probe_row = make_bank_row(variant="b-feasible")
    probe_row.task["split"] = "val"
    train_row.task["split"] = "train"
    rows, stats = pilot_dataset.build_rows([train_row, probe_row])
    assert [r["variant_id"] for r in rows] == ["a-feasible"]
    assert stats.dropped[pilot_dataset.DROP_SPLIT] == 1
