"""Conversion of seed tasks to the hermetic pytest-verifier contract (docs/design.md section 3)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from harbor.models.task.task import Task

from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks import hermetic
from honest_agents.tasks.hermetic import (
    HERMETIC_MARKER,
    convert_many,
    hermetic_problems,
    is_hermetic,
    to_pytest_verifier,
)
from honest_agents.tasks.taskfs import read_task_toml, write_task_toml
from honest_agents.tasks.validate import task_hash
from tests.tasks.hermetic_fixtures import ET_DOCKERFILE, ET_TEST_PY, et_task
from tests.tasks.synthetic import TASK_TOML

NETWORK_WORDS = ("apt-get", "curl", "uvx", "uv ", "pip install")


def convert(tmp_path: Path, **kw):
    d = et_task(tmp_path / "t", **kw)
    return d, to_pytest_verifier(d)


def refused(tmp_path: Path, reason_part: str, **kw) -> None:
    """The task cannot be converted: a reason is given and nothing on disk changed."""
    d = et_task(tmp_path / "t", **kw)
    before = task_hash(d)
    rep = to_pytest_verifier(d)
    assert not rep.ok and not rep.changed, rep
    assert reason_part in rep.reason, rep.reason
    assert task_hash(d) == before
    assert not is_hermetic(d)


# ── the real Endless Terminals shape ─────────────────────────────────────────


def test_endless_terminals_task_is_converted(tmp_path):
    d, rep = convert(tmp_path)
    assert rep.ok and rep.changed and not rep.already_hermetic and not rep.reasons
    assert rep.test_files == ["test_final_state.py"] and rep.n_tests == 3 and rep.cwd == "/home/user"
    assert not rep.verifier_site and not rep.site_modules and not rep.warnings
    sh = (d / "tests" / "test.sh").read_text()
    assert HERMETIC_MARKER in sh
    assert not any(w in sh for w in NETWORK_WORDS), sh
    assert "cd /home/user" in sh and "/logs/verifier/reward.txt" in sh
    assert "python3 -m pytest /tests" in sh and "--junitxml=/logs/verifier/junit.xml" in sh
    assert os.access(d / "tests" / "test.sh", os.X_OK)
    assert subprocess.run(["bash", "-n", str(d / "tests" / "test.sh")], check=False).returncode == 0
    # the test files are untouched
    assert (d / "tests" / "test_final_state.py").read_text() == ET_TEST_PY


def test_converted_labels_and_audit_record(tmp_path):
    d, _ = convert(tmp_path)
    spec = HonestTaskSpec.from_task_dir(d)
    assert spec.verifier == "pytest" and spec.verifier_site is False and spec.feasible
    cfg = read_task_toml(d)
    assert cfg["verifier"]["network_mode"] == "no-network"
    assert cfg["verifier"]["timeout_sec"] == 300.0  # existing verifier settings survive
    h = cfg["metadata"]["hermetic"]
    assert h["cwd"] == "/home/user" and h["test_files"] == ["test_final_state.py"] and h["n_tests"] == 3
    assert len(h["original_test_sh_sha256"]) == 64


def test_converted_task_is_a_valid_harbor_task_with_offline_verifier(tmp_path):
    from harbor.models.task.config import NetworkMode

    d, _ = convert(tmp_path)
    task = Task(d)
    assert task.config.verifier.network_mode == NetworkMode.NO_NETWORK
    assert task.config.verifier.explicit_phase_policy().network_mode == NetworkMode.NO_NETWORK
    assert task.config.metadata["honest"]["verifier"] == "pytest"


def test_conversion_is_idempotent(tmp_path):
    d, first = convert(tmp_path)
    before = task_hash(d)
    again = to_pytest_verifier(d)
    assert again.ok and again.already_hermetic and not again.changed and again.cwd == "/home/user"
    assert task_hash(d) == before
    assert first.changed


def test_conversion_keeps_existing_honest_labels_and_variant_detail(tmp_path):
    d = et_task(tmp_path / "t")
    cfg = read_task_toml(d)
    cfg["metadata"] = {
        "honest": {
            "feasible": False,
            "perturbation": "remove_input",
            "expected_blockers": ["missing_capability"],
            "expected_targets": ["/home/user/data/input.txt"],
            "pair_id": "p1",
            "irreversible_paths": ["/srv/x"],
        },
        "honest_variant": {"path": "/home/user/data/input.txt"},
    }
    write_task_toml(d, cfg)
    assert to_pytest_verifier(d).ok
    spec = HonestTaskSpec.from_task_dir(d)
    assert (spec.feasible, spec.perturbation, spec.expected_targets, spec.pair_id) == (
        False,
        "remove_input",
        ["/home/user/data/input.txt"],
        "p1",
    )
    assert spec.irreversible_paths == ["/srv/x"] and spec.verifier == "pytest"
    assert read_task_toml(d)["metadata"]["honest_variant"] == {"path": "/home/user/data/input.txt"}


# ── what is hermetic ─────────────────────────────────────────────────────────


def test_unconverted_task_is_not_hermetic_and_says_why(tmp_path):
    d = et_task(tmp_path / "t")
    problems = hermetic_problems(d)
    assert any("verifier is not" in p for p in problems)
    assert any("network_mode" in p for p in problems)
    assert any("uses the network" in p for p in problems)
    assert not is_hermetic(d)


@pytest.mark.parametrize(
    "mutate,part",
    [
        (
            lambda d: (d / "tests" / "test.sh").write_text("#!/bin/bash\ncurl x\necho 1 > /logs/verifier/reward.txt\n"),
            "network",
        ),
        (
            lambda d: (d / "tests" / "test.sh").write_text("#!/bin/bash\npytest /tests\n"),
            "does not write",
        ),
        (lambda d: (d / "tests" / "test.sh").unlink(), "test.sh is missing"),
        (
            lambda d: (d / "tests" / "test_final_state.py").write_text("x = 1\n"),
            "no collectable pytest test",
        ),
        (
            lambda d: (d / "tests" / "test_final_state.py").unlink(),
            "no collectable pytest test",
        ),
        (
            lambda d: (d / "tests" / "test_final_state.py").write_text("def test_a(:\n"),
            "does not parse",
        ),
        (
            lambda d: (d / "tests" / "test_final_state.py").write_text(
                "import yaml\n\n\ndef test_a():\n    assert yaml\n"
            ),
            "verifier_site is false",
        ),
    ],
)
def test_each_contract_violation_is_detected(tmp_path, mutate, part):
    d, rep = convert(tmp_path)
    assert rep.ok and is_hermetic(d)
    mutate(d)
    problems = hermetic_problems(d)
    assert any(part in p for p in problems), problems
    assert not is_hermetic(d)


def test_dropping_the_offline_verifier_policy_is_detected(tmp_path):
    d, _ = convert(tmp_path)
    cfg = read_task_toml(d)
    cfg["verifier"].pop("network_mode")
    write_task_toml(d, cfg)
    assert any("offline" in p for p in hermetic_problems(d))


def test_task_without_honest_metadata_is_not_hermetic(tmp_path):
    assert not is_hermetic(et_task(tmp_path / "t"))
    assert not is_hermetic(tmp_path / "does-not-exist")


# ── imports: standard library, site-packages, local helpers ──────────────────


def test_standard_library_and_local_helpers_do_not_need_site(tmp_path):
    tests = {
        "test_final_state.py": (
            "import os, re, json, stat, subprocess, distutils\nfrom pathlib import Path\nimport pytest\n"
            "import helper\nfrom pkg.mod import x\n\n\ndef test_a():\n    assert helper.ok and x\n"
        ),
        "helper.py": "ok = True\n",
        "pkg/mod.py": "x = 1\n",
    }
    d = tmp_path / "t"
    et_task(
        d,
        test_py={
            "test_final_state.py": tests["test_final_state.py"],
            "helper.py": "ok = True\n",
        },
    )
    (d / "tests" / "pkg").mkdir()
    (d / "tests" / "pkg" / "mod.py").write_text("x = 1\n")
    rep = to_pytest_verifier(d)
    assert rep.ok and not rep.verifier_site and rep.site_modules == [], rep


def test_third_party_import_sets_verifier_site(tmp_path):
    d, rep = convert(
        tmp_path,
        test_py="import yaml\nimport requests.adapters\nimport pytest\n\n\ndef test_a():\n    assert yaml and requests\n",
    )
    assert rep.ok and rep.verifier_site and rep.site_modules == ["requests", "yaml"]
    assert HonestTaskSpec.from_task_dir(d).verifier_site is True
    assert is_hermetic(d)


def test_module_newer_than_the_floor_is_a_warning(tmp_path):
    _, rep = convert(tmp_path, test_py="import tomllib\n\n\ndef test_a():\n    assert tomllib\n")
    assert rep.ok and any("tomllib" in w for w in rep.warnings)


# ── refusals ────────────────────────────────────────────────────────────────


def test_refuses_task_without_pytest_files(tmp_path):
    refused(
        tmp_path,
        "no pytest test files",
        test_py={"check.py": "def test_a():\n    pass\n"},
    )


def test_refuses_test_file_without_test_functions(tmp_path):
    refused(tmp_path, "no pytest test files", test_py="VALUE = 1\n")


def test_refuses_unparseable_or_too_new_syntax(tmp_path):
    refused(tmp_path, "does not parse", test_py="def test_a(:\n    pass\n")
    refused(
        tmp_path / "b",
        "does not parse",
        test_py="def test_a():\n    match 1:\n        case 1:\n            pass\n",
    )


def test_refuses_shell_checks_that_pytest_would_not_reproduce(tmp_path):
    sh = "#!/bin/bash\ntest -f /home/user/out.txt && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n"
    refused(tmp_path, "does not run pytest", test_sh=sh)


def test_refuses_filtered_runs(tmp_path):
    sh = "#!/bin/bash\ncd /home/user\npython3 -m pytest /tests -k 'not slow' -q\necho 1 > /logs/verifier/reward.txt\n"
    refused(tmp_path, "filters the tests", test_sh=sh)


@pytest.mark.parametrize(
    "flag",
    [
        "-k'not slow'",
        "-mslow",
        "-m 'not slow'",
        "--deselect=test_a.py::test_x",
        "-ofoo=1",
    ],
)
def test_refuses_every_spelling_of_a_filter(tmp_path, flag):
    sh = f"#!/bin/bash\ncd /home/user\npython3 -m pytest /tests {flag} -q\necho 1 > /logs/verifier/reward.txt\n"
    refused(tmp_path, "filters the tests", test_sh=sh)


def test_refuses_runs_of_only_some_test_files(tmp_path):
    sh = "#!/bin/bash\ncd /home/user\npytest /tests/test_a.py\necho 1 > /logs/verifier/reward.txt\n"
    refused(
        tmp_path,
        "runs only",
        test_sh=sh,
        test_py={
            "test_a.py": "def test_a():\n    pass\n",
            "test_b.py": "def test_b():\n    assert False\n",
        },
    )


def test_refuses_named_file_a_directory_run_would_not_collect(tmp_path):
    sh = "#!/bin/bash\ncd /home/user\npytest /tests/check_state.py\necho 1 > /logs/verifier/reward.txt\n"
    refused(
        tmp_path,
        "would not collect",
        test_sh=sh,
        test_py={
            "check_state.py": "def test_a():\n    pass\n",
            "test_x.py": "def test_x():\n    pass\n",
        },
    )


def test_refuses_unknown_pytest_plugins_but_not_ctrf(tmp_path):
    base = "#!/bin/bash\ncd /home/user\nuvx --python 3.13 --with pytest==8.4.1 --with {plugin} pytest /tests/test_final_state.py\n"
    base += "echo 1 > /logs/verifier/reward.txt\n"
    refused(tmp_path, "pytest plugins", test_sh=base.format(plugin="pytest-asyncio==0.23"))
    rep = to_pytest_verifier(et_task(tmp_path / "ok", test_sh=base.format(plugin="pytest-json-ctrf==0.3.5")))
    assert rep.ok, rep.reason


def test_refuses_images_without_a_usable_python(tmp_path):
    refused(
        tmp_path,
        "does not install python3",
        dockerfile="FROM alpine:3.19\nRUN apk add curl\n",
    )
    refused(
        tmp_path / "b",
        "older than 3.9",
        dockerfile="FROM ubuntu:20.04\nRUN apt-get install -y python3\n",
    )
    refused(tmp_path / "c", "older than 3.9", dockerfile="FROM python:3.7\n")
    refused(tmp_path / "d", "no environment/Dockerfile", dockerfile=None)


def test_old_base_with_explicit_newer_python_is_accepted(tmp_path):
    df = "FROM ubuntu:20.04\nRUN apt-get install -y python3.10 && pip install pytest\n"
    assert to_pytest_verifier(et_task(tmp_path / "t", dockerfile=df)).ok


def test_refuses_missing_test_sh(tmp_path):
    d = et_task(tmp_path / "t")
    (d / "tests" / "test.sh").unlink()
    rep = to_pytest_verifier(d)
    assert not rep.ok and "no tests/test.sh" in rep.reason


def test_unreadable_task_toml_is_reported(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "task.toml").write_text("not = [toml")
    rep = to_pytest_verifier(d)
    assert not rep.ok and "unreadable" in rep.reason


# ── warnings and cwd ─────────────────────────────────────────────────────────


def test_image_without_pytest_warns_but_converts(tmp_path):
    _, rep = convert(tmp_path, dockerfile="FROM ubuntu:22.04\nRUN apt-get install -y python3\n")
    assert rep.ok and any("does not install pytest" in w for w in rep.warnings)


def test_cwd_comes_from_the_original_cd_then_workdir(tmp_path):
    sh = "#!/bin/bash\npytest /tests/test_final_state.py\necho 1 > /logs/verifier/reward.txt\n"
    d, rep = convert(tmp_path, test_sh=sh, dockerfile=ET_DOCKERFILE + "WORKDIR /app\n")
    assert rep.cwd == "/app" and any("no cd" in w for w in rep.warnings)
    assert "cd /app" in (d / "tests" / "test.sh").read_text()
    d2, rep2 = convert(tmp_path / "x", test_sh=sh)
    assert rep2.cwd == "/" and "cd / 2>/dev/null || cd /" in (d2 / "tests" / "test.sh").read_text()


def test_cwd_dependent_tests_warn(tmp_path):
    _, rep = convert(tmp_path, test_py="import os\n\n\ndef test_a():\n    assert os.getcwd()\n")
    assert rep.ok and any("working directory" in w for w in rep.warnings)


def test_convert_many_never_raises(tmp_path):
    good = et_task(tmp_path / "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    reps = convert_many([good, bad, tmp_path / "missing"])
    assert [r.ok for r in reps] == [True, False, False]
    assert reps[1].reasons and reps[2].reasons


def test_report_serialises(tmp_path):
    _, rep = convert(tmp_path)
    d = rep.to_dict()
    assert d["ok"] is True and d["task_dir"] == str(rep.task_dir) and d["reason"] == ""


def test_cli_converts_a_directory(tmp_path, capsys):
    et_task(tmp_path / "tasks" / "a")
    et_task(tmp_path / "tasks" / "b", test_py="X = 1\n")
    hermetic.main(["--tasks-dir", str(tmp_path / "tasks")])
    out = capsys.readouterr().out
    assert "1 of 2 tasks hermetic, 1 not convertible" in out
    assert is_hermetic(tmp_path / "tasks" / "a") and not is_hermetic(tmp_path / "tasks" / "b")


def test_default_toml_task_has_legacy_version_key():
    """Guards the fixture: the synthetic seeds keep Endless Terminals' legacy `version = "0.1"`."""
    assert 'version = "0.1"' in TASK_TOML


# ── real data (optional) ────────────────────────────────────────────────────

SAMPLE_DIR = os.environ.get("HONEST_ET_SAMPLES")


@pytest.mark.skipif(
    not SAMPLE_DIR,
    reason="set HONEST_ET_SAMPLES to a dir of raw Endless Terminals tasks",
)
def test_real_endless_terminals_samples_convert(tmp_path):
    from honest_agents.tasks.sources import normalize_task

    raws = sorted(p for p in Path(SAMPLE_DIR).glob("task_*") if (p / "task.toml").exists())
    assert raws
    for raw in raws:
        st = normalize_task(raw, tmp_path / raw.name, source="endless-terminals", task_id=raw.name)
        assert st is not None and st.hermetic and is_hermetic(st.task_dir), raw.name
        assert "uvx" not in (st.task_dir / "tests" / "test.sh").read_text()


# ── L4: nothing that grades the task may sit in the image or its build context ───────────────────────────────────────


def test_expected_state_files_in_the_build_context_fail_the_contract_and_are_scrubbed_on_conversion(tmp_path):
    d = et_task(tmp_path / "t")
    for name in ("task.json", "test_initial_state.py", "container.def"):
        (d / "environment" / name).write_text("expected")
    assert any("task.json" in p for p in hermetic.image_leak_problems(d))
    rep = to_pytest_verifier(d)
    assert rep.ok and any("removed" in c for c in rep.changes)
    assert sorted(p.name for p in (d / "environment").iterdir()) == ["Dockerfile"]
    assert is_hermetic(d)


def test_a_dockerfile_that_copies_tests_is_refused_and_not_hermetic(tmp_path):
    df = ET_DOCKERFILE + "COPY tests/test_final_state.py /app/\n"
    refused(tmp_path, "COPY/ADDs", dockerfile=df)
    d = et_task(tmp_path / "u", dockerfile=df)
    assert any("COPY/ADDs" in p for p in hermetic_problems(d))
