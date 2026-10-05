"""Leak-register regression tests 1, 2, 8 and 9: what the policy can see at runtime, and which rows may be trained on.

* The only row-specific text the policy sees is instruction.md; Docker names and every failure text are opaque.
* The training / probe / pilot / difficulty consumers refuse rows whose split is not the one they are for, including a
  missing split column; eval-sourced quarantine entries do not change a rebuilt training set.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from honest_agents import bank, harness, train
from honest_agents.bank import QuarantineEntry
from honest_agents.env import HonestHarborEnv
from honest_agents.pilot import dataset as pilot_dataset
from honest_agents.trainer import _tool_error_text
from tests.fakes import FakeFactory, install_fake_factory, make_pytest_task_dir
from tests.pilot.helpers import make_bank_row

REPO = Path(__file__).resolve().parent.parent
VARIANT = "ab12cd-conflicting_constraint-impossible__control"
LABEL_WORDS = (
    "ab12cd",
    "conflicting_constraint",
    "remove_input",
    "revoke_permission",
    "high_risk",
    "no_network",
    "remove_tool",
    "control",
    "impossible",
    "decoy",
    "twin",
    "abstain",
)
FEASIBLE = 'feasible = true\nprotected_paths = ["protected"]'


def _no_labels(text: str) -> None:
    low = text.lower()
    assert not [w for w in LABEL_WORDS if w in low], text


def seed(root: Path) -> None:
    (root / "protected").mkdir(exist_ok=True)
    (root / "protected" / "tool").write_text("real tool")


@pytest.fixture
def factory(monkeypatch, tmp_path, grading_paths) -> FakeFactory:
    return install_fake_factory(monkeypatch, FakeFactory(tmp_path / "sandboxes", lambda i: {"on_start": seed}))


@pytest.fixture
def task(tmp_path) -> Path:
    return make_pytest_task_dir(tmp_path / VARIANT, honest=FEASIBLE, instruction="Write `hi` to hello.txt.\n")


@pytest.fixture
def env():
    """Stop the env while its loop thread is alive (TRL's `HarborEnv.__del__` would block on it at shutdown)."""
    e = HonestHarborEnv(command_timeout_sec=20)
    yield e
    with contextlib.suppress(Exception):
        if e._boot is not None:
            e._boot.result(timeout=10)
    with contextlib.suppress(Exception):
        asyncio.run_coroutine_threadsafe(e._stop(), e._loop).result(timeout=10)
    e._run = lambda coro: coro.close()
    e._loop.call_soon_threadsafe(e._loop.stop)


# ── regression 2: opaque names and generic failure text (L3) ──────────────────


@pytest.mark.parametrize("environment_type", ["docker", "modal"])
def test_harbor_session_id_is_opaque_and_unique(environment_type):
    t = SimpleNamespace(short_name=VARIANT)
    names = [harness.harbor_names(t, environment_type) for _ in range(50)]
    assert len({sid for _, sid in names}) == 50
    for env_name, sid in names:
        _no_labels(sid)
        assert sid.startswith("ep") and len(sid) == 18
        if environment_type == "docker":
            _no_labels(env_name)


def test_docker_names_reach_harbor_opaque(factory, task, env):
    env.reset(task_dir=str(task))
    env._wait_boot()
    (fake,) = factory.created
    for key in ("environment_name", "session_id"):
        _no_labels(fake.factory_kwargs[key])


def test_the_build_lock_stays_keyed_on_the_real_task_name_and_boots_still_build_once(task):
    """Harbor tags images `hb__<content hash>`, so opaque names do not rebuild per episode; the host-side lock and the
    warm set are keyed on the task name (never shown to the sandbox)."""
    started: list[str] = []

    class _Env:
        async def start(self, force_build):
            started.append("start")

    t = SimpleNamespace(
        short_name=VARIANT,
        paths=SimpleNamespace(environment_dir=task / "environment"),
        config=SimpleNamespace(environment=SimpleNamespace(docker_image=None)),
    )
    harness._WARM.discard(VARIANT)
    asyncio.run(harness.start_sandbox(_Env(), t, "docker"))
    asyncio.run(harness.start_sandbox(_Env(), t, "docker"))
    assert started == ["start", "start"] and VARIANT in harness._WARM and VARIANT in harness._BUILD_LOCKS
    harness._WARM.discard(VARIANT)


def test_boot_failure_text_names_nothing(factory, task, env, caplog):
    leaky = f"Docker compose command failed for environment {VARIANT}. Command: docker compose -p {VARIANT}__abc up"
    factory.make_kwargs = lambda i: {"fail_start": RuntimeError(leaky)}
    with caplog.at_level(logging.ERROR):
        env.reset(task_dir=str(task))
        out = env.bash("true")
    assert out == harness.SANDBOX_UNAVAILABLE
    assert VARIANT in caplog.text  # the real text is kept host-side


def test_exec_failure_and_timeout_text_name_nothing(factory, task, env):
    env.reset(task_dir=str(task))
    env._wait_boot()
    (fake,) = factory.created
    for exc in (
        RuntimeError(f"container {VARIANT}-main-1 is not running; /tmp/honest_x/{VARIANT}/environment"),
        TimeoutError(f"Command timed out after 60 seconds: docker compose -p {VARIANT}"),
    ):

        async def dead(*a, _e=exc, **k):
            raise _e

        fake.exec = dead
        out = env.bash("true")
        assert harness.SANDBOX_UNAVAILABLE in out
        _no_labels(out)
        assert VARIANT not in out and "/tmp" not in out and "docker" not in out.lower()


def test_report_paths_return_fixed_text(factory, task, env):
    env.reset(task_dir=str(task))
    out = env.abstain("missing", "missing_capability", "x", 0.5)
    assert out == harness.REPORT_RECEIVED
    assert env.bash("true") == harness.EPISODE_OVER
    _no_labels(out + harness.EPISODE_OVER)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(f"compose up failed for {VARIANT}"),
        OSError(f"/tmp/honest_x/{VARIANT}/environment/Dockerfile: no such file"),
        ValueError(f"docker compose -p {VARIANT} exec"),
        TypeError(f"{VARIANT} got an unexpected argument"),
    ],
)
def test_trainer_tool_errors_are_fixed_strings(exc):
    text = _tool_error_text(exc)
    assert text in (harness.SANDBOX_UNAVAILABLE, "[invalid tool arguments]")
    _no_labels(text)


# ── regression 1: the prompt column is constant, reset() returns only instruction.md ──────────────────────────────


def test_reset_returns_only_the_instruction_and_the_prompt_column_is_constant(factory, tmp_path, env):
    t1 = make_pytest_task_dir(tmp_path / VARIANT, honest=FEASIBLE, instruction="Write `hi` to hello.txt.\n")
    assert (
        env.reset(task_dir=str(t1), pair_id="p", variant_id=VARIANT, kind="control")
        == (t1 / "instruction.md").read_text()
    )

    path = tmp_path / "rows.jsonl"
    rows = [{"task_dir": str(t1), "variant_id": f"v{i}", "kind": k, "split": "train"} for i, k in enumerate(("a", "b"))]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded = train.load_rows(path)
    assert loaded[0]["prompt"] == loaded[1]["prompt"] == train._constant_prompt()
    assert loaded[0]["prompt"][1]["content"] == ""


def _cfg() -> train.TrainRunConfig:
    cfg = train.load_run_config(REPO / "configs/honest_alpha0.25.yaml", [])
    cfg.data.check_tasks = False
    return cfg


def _row(split, **kw) -> dict:
    r = {"task_dir": "/x", "variant_id": "v", "prompt": train._constant_prompt()} | kw
    if split is not None:
        r["split"] = split
    return r


def test_check_rows_refuses_a_row_specific_prompt():
    with pytest.raises(ValueError, match="constant"):
        train.check_rows(
            [
                _row(
                    "train",
                    prompt=[{"role": "system", "content": harness.SYSTEM_PROMPT}, {"role": "user", "content": "x"}],
                )
            ],
            _cfg(),
            role="train",
        )


# ── regression 8: split guards ────────────────────────────────────────────────


@pytest.mark.parametrize("split", [None, "", "test", "validation", "val", "held_out"])
def test_check_rows_train_requires_split_train(split):
    with pytest.raises(ValueError, match="split"):
        train.check_rows([_row(split)], _cfg(), role="train")
    assert train.check_rows([_row("train")], _cfg(), role="train")["rows"] == 1


@pytest.mark.parametrize("split", [None, "", "train", "test"])
def test_check_rows_probe_requires_validation_split(split):
    with pytest.raises(ValueError, match="split"):
        train.check_rows([_row(split)], _cfg(), role="validation")
    assert train.check_rows([_row("validation")], _cfg(), role="validation")["rows"] == 1


@pytest.mark.parametrize("split", [None, "test", "validation", "val"])
def test_pilot_refuses_non_train_and_missing_splits(split):
    row = make_bank_row(variant="x")
    task = {k: v for k, v in row.task.items() if k != "split"}
    if split is not None:
        task["split"] = split
    bad = row.model_copy(update={"task": task})
    ok = make_bank_row(variant="y")
    ok = ok.model_copy(update={"task": {**ok.task, "split": "train"}})
    built, stats = pilot_dataset.build_rows([bad, ok])
    assert [r["variant_id"] for r in built] == ["y"]
    assert stats.dropped[pilot_dataset.DROP_SPLIT] == 1


def test_difficulty_counts_only_train_rows():
    """The gate skips (and counts) validation and test rows, and refuses a row with no split column."""
    from honest_agents.tasks import difficulty

    def mk(v, split):
        r = make_bank_row(variant=v, passed=True)
        task = {k: x for k, x in r.task.items() if k != "split"}
        if split is not None:
            task["split"] = split
        return r.model_copy(update={"task": task})

    report = difficulty.difficulty_report([mk("t0", "train"), mk("e0", "test")], n_rollouts=1, band=(0, 1))
    text = json.dumps(report.to_dict(), default=str)
    assert "t0" in text and "e0" not in text
    assert report.non_train_rows_skipped == {"test": 1}
    with pytest.raises(difficulty.MissingSplitError):
        difficulty.difficulty_report([mk("t0", "train"), mk("m0", None)], n_rollouts=1, band=(0, 1))


# ── regression 9: eval-sourced quarantine entries leave the training set unchanged ────────────────────────────────


def test_eval_quarantine_entries_are_ignored_by_default(tmp_path):
    q = tmp_path / "quarantine.jsonl"
    bank.append_quarantine(
        [
            QuarantineEntry(variant_id="from_eval", source="eval"),
            QuarantineEntry(variant_id="from_train", source="train"),
            QuarantineEntry(variant_id="from_val", source="validation"),
            QuarantineEntry(variant_id="eval_unclaimed", source="eval", kind=bank.QUARANTINE_UNCLAIMED_PASS),
        ],
        q,
    )
    assert bank.quarantined_variants(q) == {"from_train", "from_val"}
    assert bank.quarantined_variants(q, exclude_sources=()) == {"from_eval", "from_train", "from_val"}
    both = (bank.QUARANTINE_LEAK, bank.QUARANTINE_UNCLAIMED_PASS)
    assert "eval_unclaimed" not in bank.quarantined_variants(q, kinds=both)
    # the same entries, plus or minus the eval ones, leave the set a rebuild would exclude unchanged
    only_train = tmp_path / "q2.jsonl"
    bank.append_quarantine([QuarantineEntry(variant_id="from_train", source="train")], only_train)
    assert bank.quarantined_variants(q) >= bank.quarantined_variants(only_train)
    q_no_eval = tmp_path / "q3.jsonl"
    bank.append_quarantine(
        [e for e in bank.read_quarantine(q) if e.source != "eval"],
        q_no_eval,
    )
    assert bank.quarantined_variants(q) == bank.quarantined_variants(q_no_eval)


def test_held_out_and_openenv_quarantine_sources_are_ignored_by_default(tmp_path):
    """`trl_rewards` / `difficulty` record the row's split as `source`, and OpenEnv episodes may run on test tasks."""
    q = tmp_path / "quarantine.jsonl"
    bank.append_quarantine(
        [QuarantineEntry(variant_id=f"v_{s}", source=s) for s in ("test", "openenv", "eval", "train", "validation")], q
    )
    assert bank.quarantined_variants(q) == {"v_train", "v_validation"}
    assert len(bank.quarantined_variants(q, exclude_sources=())) == 5
