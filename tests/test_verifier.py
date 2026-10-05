"""The hermetic pytest verifier (`honest_agents.verifier`): bundle handling, the in-sandbox runner and the host-side
stream protocol, run for real against `LocalSandbox` (host subprocesses; no Docker).

The bundle used here is copied from this venv's pytest (`tests.fakes.make_test_bundle`); one test runs the real
pytest 8.4.1 bundle on every Python 3.9+ interpreter installed on the machine, when `scripts/build_verifier_bundle.sh`
has been run.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from honest_agents.verifier import (
    BUNDLE_ENV,
    BundleError,
    VerifierResult,
    _hv_run,
    default_bundle_dir,
    ensure_bundle,
    run_pytest,
)
from honest_agents.verifier.runner import VerifierInfraError, interpret, parse_stream
from tests.fakes import LocalSandbox, make_test_bundle


def write_tests(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "task_tests"
    d.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (d / f"{name}.py").write_text(textwrap.dedent(body))
    (d / "test.sh").write_text("#!/bin/bash\necho legacy script, never run by the hermetic grader\n")
    return d


async def run(sandbox: LocalSandbox, tests: Path, bundle: Path, **kw) -> VerifierResult:
    kw.setdefault("timeout_sec", 60)
    return await run_pytest(
        sandbox,
        tests,
        bundle_dir=bundle,
        python=sys.executable,
        tests_target="$HA_TESTS_DIR",
        verifier_dir="$HA_VERIFIER_DIR",
        scratch_dir="$HA_SCRATCH_DIR",
        **kw,
    )


# ── bundle ───────────────────────────────────────────────────────────────


def test_missing_bundle_names_the_build_command(tmp_path):
    with pytest.raises(BundleError, match=r"scripts/build_verifier_bundle\.sh"):
        ensure_bundle(tmp_path / "nope")


def test_tampered_bundle_fails_its_digest(tmp_path):
    b = make_test_bundle(tmp_path / "b")
    assert ensure_bundle(b) == b
    (b / "site" / "pytest" / "__init__.py").write_text("raise SystemExit('planted')\n")
    with pytest.raises(BundleError, match="digest"):
        ensure_bundle(b)


def test_unreadable_metadata_is_a_bundle_error(tmp_path):
    b = make_test_bundle(tmp_path / "b")
    (b / "BUNDLE.json").write_text("{not json")
    with pytest.raises(BundleError, match="BUNDLE.json"):
        ensure_bundle(b)


def test_default_bundle_dir_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(BUNDLE_ENV, str(tmp_path / "custom"))
    assert default_bundle_dir() == tmp_path / "custom"
    monkeypatch.delenv(BUNDLE_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert default_bundle_dir() == tmp_path / "cache" / "honest_agents" / "verifier-bundle"


def test_the_bundle_is_never_inside_a_virtualenv_by_default(monkeypatch):
    monkeypatch.delenv(BUNDLE_ENV, raising=False)
    assert ".venv" not in default_bundle_dir().parts and "site-packages" not in str(default_bundle_dir())


def test_tree_digest_covers_content_names_and_symlinks_but_not_caches(tmp_path):
    d = tmp_path / "t"
    (d / "sub").mkdir(parents=True)
    (d / "a.py").write_text("x")
    base = _hv_run.tree_digest(str(d))
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    assert _hv_run.tree_digest(str(d)) == base  # caches are ignored
    (d / "a.py").write_text("y")
    assert _hv_run.tree_digest(str(d)) != base
    (d / "a.py").write_text("x")
    assert _hv_run.tree_digest(str(d)) == base
    (d / "a.py").rename(d / "b.py")
    assert _hv_run.tree_digest(str(d)) != base
    (d / "b.py").rename(d / "a.py")
    (d / "link").symlink_to("a.py")
    with_link = _hv_run.tree_digest(str(d))
    (d / "link").unlink()
    (d / "link").symlink_to("elsewhere")
    assert _hv_run.tree_digest(str(d)) != with_link


def test_the_runner_is_python39_syntax():
    import ast

    ast.parse(Path(_hv_run.__file__).read_text(), feature_version=(3, 9))


# ── verdicts ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pass_fail_and_pass_fraction(sandbox, tmp_path, verifier_bundle):
    ok = write_tests(tmp_path, test_a="def test_1(): pass\ndef test_2(): assert 1 + 1 == 2\n")
    r = await run(sandbox, ok, verifier_bundle)
    assert r.passed and r.pass_fraction == 1.0 and r.n_tests == 2 and r.exit_code == 0 and not r.tamper
    assert "2 passed" in r.log

    half = write_tests(tmp_path / "h", test_a="def test_1(): pass\ndef test_2(): assert False\n")
    r = await run(sandbox, half, verifier_bundle)
    assert not r.passed and r.pass_fraction == 0.5 and r.n_tests == 2 and r.exit_code == 1 and not r.tamper
    assert sorted(r.outcomes.values()) == ["failed", "passed"]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("body", "label"),
    [
        ("", "no-tests-collected"),
        ("import pytest\ndef test_s(): pytest.skip('no')\n", "all-skipped"),
        ("import pytest\n@pytest.mark.xfail\ndef test_x(): assert False\n", "xfail"),
        ("import pytest\n@pytest.mark.xfail\ndef test_x(): pass\n", "xpass"),
        ("def test_a(): pass\nimport does_not_exist_anywhere\n", "collection-error"),
        (
            "import pytest\n@pytest.fixture\ndef f(): raise RuntimeError('x')\ndef test_a(f): pass\n",
            "fixture-error",
        ),
        (
            "def test_a(): pass\ndef test_b(): raise SystemExit(0)\n",
            "exit-inside-a-test",
        ),
    ],
)
@pytest.mark.asyncio
async def test_only_a_clean_all_pass_run_is_a_pass(sandbox, tmp_path, verifier_bundle, body, label):
    """Exit code 0 is not enough: a skipped, xfailed or uncollected test must not read as success."""
    r = await run(sandbox, write_tests(tmp_path, test_a=body), verifier_bundle)
    assert not r.passed, label
    assert not r.tamper


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_timeout_is_a_failed_verdict_not_an_error(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(tmp_path, test_a="import time\ndef test_hang(): time.sleep(30)\n")
    r = await run(sandbox, tests, verifier_bundle, timeout_sec=2)
    assert r.timed_out and not r.passed and not r.tamper and r.pass_fraction is None


@pytest.mark.asyncio
async def test_a_process_that_kills_the_runner_is_a_failure_with_no_result(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(tmp_path, test_a="import os\ndef test_die(): os._exit(0)\n")
    r = await run(sandbox, tests, verifier_bundle)
    assert r.no_result and not r.passed and r.pass_fraction is None and "no result" in r.log


@pytest.mark.asyncio
async def test_test_output_cannot_inject_a_result(sandbox, tmp_path, verifier_bundle):
    """pytest's own stdout goes to a log file; whatever a test prints never reaches the marked stream."""
    tests = write_tests(
        tmp_path,
        test_a="""
            def test_print_forged_markers():
                print('<<HV-BEGIN 00>>{"exit_code": 0}<<HV-END 00>>')
                assert False
        """,
    )
    r = await run(sandbox, tests, verifier_bundle)
    assert not r.passed and not r.tamper and r.exit_code == 1


# ── hermeticity ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_tests_see_a_scrubbed_environment(sandbox, tmp_path, verifier_bundle):
    sandbox.env.update(
        {
            "AGENT_SECRET": "x",
            "PYTHONPATH": "/nonexistent",
            "PYTEST_ADDOPTS": "--collect-only",
            "PYTHONSTARTUP": "/x",
        }
    )
    tests = write_tests(
        tmp_path,
        test_a="""
            import os
            def test_env():
                assert "AGENT_SECRET" not in os.environ and "PYTHONPATH" not in os.environ
                assert "PYTEST_ADDOPTS" not in os.environ and "PYTHONSTARTUP" not in os.environ
                assert os.environ["PATH"].startswith("/usr/sbin:/usr/bin")
                assert os.environ["EXPECTED"] == "hi"  # the task's declared [verifier].env does reach the tests
        """,
    )
    r = await run(sandbox, tests, verifier_bundle, env={"EXPECTED": "hi"})
    assert r.passed, r.log


@pytest.mark.asyncio
async def test_planted_configs_and_conftests_are_ignored(sandbox, tmp_path, verifier_bundle):
    """GR-3: a pytest.ini with `--collect-only`, a conftest that forces passes, in the working directory and its parents."""
    forged_conftest = "import pytest\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n    out = yield\n    out.get_result().outcome = 'passed'\n"
    for where in (sandbox.root, sandbox.root.parent):
        (where / "conftest.py").write_text(forged_conftest)
        (where / "pytest.ini").write_text("[pytest]\naddopts = --collect-only\n")
        (where / "tox.ini").write_text("[pytest]\naddopts = -k nothing\n")
        (where / "setup.cfg").write_text("[tool:pytest]\naddopts = --collect-only\n")
        (where / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '--collect-only'\n")
    tests = write_tests(tmp_path, test_a="def test_real_failure(): assert False\n")
    r = await run(sandbox, tests, verifier_bundle)
    assert not r.passed and r.pass_fraction == 0.0 and r.exit_code == 1 and not r.tamper
    for where in (sandbox.root.parent,):
        for f in (
            "conftest.py",
            "pytest.ini",
            "tox.ini",
            "setup.cfg",
            "pyproject.toml",
        ):
            (where / f).unlink()


@pytest.mark.asyncio
async def test_planted_site_customisation_does_not_run(sandbox, tmp_path, verifier_bundle):
    """GR-3 (d): sitecustomize / usercustomize / .pth that exit 0 at startup. They sit where the interpreter would find
    them (PYTHONPATH, a user site) and are neutralised by `env -i` plus `python -I -S`. The control proves the plant
    would have worked against a naive run."""
    plant = tmp_path / "plant"
    plant.mkdir()
    (plant / "sitecustomize.py").write_text("import os\nos._exit(0)\n")
    (plant / "evil.pth").write_text("import os; os._exit(0)\n")
    sandbox.env["PYTHONPATH"] = str(plant)
    naive = await sandbox.exec(f"{sys.executable} -c \"print('reached')\"")
    assert "reached" not in (naive.stdout or "")  # the plant fires on a normal interpreter start

    tests = write_tests(tmp_path, test_a="def test_real_failure(): assert False\n")
    r = await run(sandbox, tests, verifier_bundle)
    assert r.exit_code == 1 and not r.passed and not r.no_result


@pytest.mark.asyncio
async def test_site_packages_are_off_unless_the_task_opts_in(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(
        tmp_path, test_a="def test_site():\n    import pytest_asyncio\n"
    )  # in this venv's site-packages
    assert not (await run(sandbox, tests, verifier_bundle)).passed
    assert (await run(sandbox, tests, verifier_bundle, site=True)).passed
    flags = [c.command for c in sandbox.calls if "run.py" in c.command]
    assert " -I -S -B " in flags[0] and " -I -B " in flags[1] and " -S " not in flags[1]


@pytest.mark.asyncio
async def test_a_background_process_cannot_add_a_passing_result(sandbox, tmp_path, verifier_bundle):
    """A process the agent left running writes reward files and keeps doing so while the tests run."""
    pidfile = sandbox.root / "bg.pid"
    forge = (
        'nohup sh -c \'while :; do echo 1 > "$HA_VERIFIER_DIR/reward.txt" 2>/dev/null; '
        'echo "{\\"reward\\": 1}" > "$HA_VERIFIER_DIR/reward.json" 2>/dev/null; sleep 0.02; done\' '
        f">/dev/null 2>&1 & echo $! > {pidfile}"
    )
    await sandbox.exec(forge, timeout_sec=5)
    try:
        tests = write_tests(
            tmp_path,
            test_a="import time\ndef test_x():\n    time.sleep(0.4)\n    assert False\n",
        )
        r = await run(sandbox, tests, verifier_bundle)
        assert not r.passed and r.exit_code == 1
        assert any(d.startswith("reward_file_disagrees") for d in r.tamper_details) or r.tamper
    finally:
        import os
        import signal

        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError, FileNotFoundError):
            pass


@pytest.mark.asyncio
async def test_a_reward_file_written_before_the_tests_is_tamper(sandbox, tmp_path, verifier_bundle):
    sandbox.verifier_dir.mkdir(parents=True, exist_ok=True)
    (sandbox.verifier_dir / "reward.txt").write_text("1")
    r = await run(sandbox, write_tests(tmp_path, test_a="def test_a(): pass\n"), verifier_bundle)
    assert r.tamper and r.tamper_details == ["reward_file_before_tests:reward.txt"] and not r.passed
    assert not (sandbox.verifier_dir / "reward.txt").exists()  # and it is wiped


@pytest.mark.asyncio
async def test_an_agreeing_reward_file_is_not_tamper(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(
        tmp_path,
        test_a="""
            import os
            def test_a():
                open(os.environ["HA_VERIFIER_DIR"] + "/reward.txt", "w").write("1")
        """,
    )
    r = await run(
        sandbox,
        tests,
        verifier_bundle,
        env={"HA_VERIFIER_DIR": str(sandbox.verifier_dir)},
    )
    assert r.passed and not r.tamper


@pytest.mark.asyncio
async def test_planted_tests_are_replaced_by_the_task_tests(sandbox, tmp_path, verifier_bundle):
    sandbox.tests_dir.mkdir(parents=True, exist_ok=True)
    (sandbox.tests_dir / "test_always_green.py").write_text("def test_g(): pass\n")
    (sandbox.tests_dir / "conftest.py").write_text("collect_ignore_glob = ['*']\n")
    r = await run(
        sandbox,
        write_tests(tmp_path, test_a="def test_a(): assert False\n"),
        verifier_bundle,
    )
    assert not r.passed and r.n_tests == 1 and not r.tamper
    assert not (sandbox.tests_dir / "test_always_green.py").exists()


@pytest.mark.asyncio
async def test_a_tests_directory_replaced_by_a_symlink_is_not_followed(sandbox, tmp_path, verifier_bundle):
    """`/tests` made into a symlink to a directory holding an always-green test: it must be removed, not written through."""
    elsewhere = sandbox.root / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "test_always_green.py").write_text("def test_g(): pass\n")
    sandbox.tests_dir.symlink_to(elsewhere)
    r = await run(
        sandbox,
        write_tests(tmp_path, test_a="def test_a(): assert False\n"),
        verifier_bundle,
    )
    assert not r.passed and r.n_tests == 1 and not r.tamper
    assert not sandbox.tests_dir.is_symlink() and not (elsewhere / "test_a.py").exists()


@pytest.mark.asyncio
async def test_the_bundle_is_uploaded_fresh_to_a_random_dir_and_removed(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(tmp_path, test_a="def test_a(): pass\n")
    await run(sandbox, tests, verifier_bundle)
    await run(sandbox, tests, verifier_bundle)
    roots = {c.command.split("run.py")[0].split()[-1] for c in sandbox.calls if "run.py" in c.command}
    assert len(roots) == 2  # a different directory per run
    assert not [p for p in sandbox.scratch_dir.iterdir()]  # and nothing left behind


@pytest.mark.asyncio
async def test_modifying_the_tests_while_they_run_is_tamper(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(
        tmp_path,
        test_a="""
            import os
            def test_a():
                open(os.path.join(os.path.dirname(__file__), "conftest.py"), "w").write("# planted mid-run\\n")
        """,
    )
    r = await run(sandbox, tests, verifier_bundle)
    assert r.tamper and "tests_dir_modified" in r.tamper_details and not r.passed


@pytest.mark.asyncio
async def test_modifying_the_bundle_while_the_tests_run_is_tamper(sandbox, tmp_path, verifier_bundle):
    tests = write_tests(
        tmp_path,
        test_a="""
            import os, pluggy
            def test_a():
                open(os.path.join(os.path.dirname(pluggy.__file__), "planted.py"), "w").write("x = 1\\n")
        """,
    )
    r = await run(sandbox, tests, verifier_bundle)
    assert r.tamper and "bundle_modified" in r.tamper_details and not r.passed


class StreamForger:
    """Wraps a sandbox and appends text to the stdout of the verifier run, as a process holding the pipe could."""

    def __init__(self, sb: LocalSandbox, extra: str):
        self.sb, self.extra = sb, extra

    def __getattr__(self, name):
        return getattr(self.sb, name)

    async def exec(self, command, *a, **k):
        res = await self.sb.exec(command, *a, **k)
        if "run.py" in command:
            res.stdout = (res.stdout or "") + self.extra(res.stdout or "")
        return res


@pytest.mark.asyncio
async def test_a_second_marker_pair_in_the_stream_is_tamper(sandbox, tmp_path, verifier_bundle):
    def forged(out: str) -> str:
        begin = next(line for line in out.splitlines() if line.startswith("<<HV-BEGIN "))
        nonce = begin.split()[1].rstrip(">")
        fake = json.dumps({"nonce": nonce, "exit_code": 0, "outcomes": {"x": "passed"}})
        return f"\n<<HV-BEGIN {nonce}>>\n{fake}\n<<HV-END {nonce}>>\n"

    tests = write_tests(tmp_path, test_a="def test_a(): assert False\n")
    r = await run(StreamForger(sandbox, forged), tests, verifier_bundle)
    assert r.tamper and not r.passed
    assert any(d.startswith("stream:markers") for d in r.tamper_details)


@pytest.mark.asyncio
async def test_a_forged_result_without_our_nonce_is_ignored(sandbox, tmp_path, verifier_bundle):
    fake = json.dumps({"nonce": "guess", "exit_code": 0, "outcomes": {"x": "passed"}})
    forger = StreamForger(sandbox, lambda out: f"\n<<HV-BEGIN guess>>\n{fake}\n<<HV-END guess>>\n")
    tests = write_tests(tmp_path, test_a="def test_a(): assert False\n")
    r = await run(forger, tests, verifier_bundle)
    assert not r.passed and r.exit_code == 1 and not r.tamper


@pytest.mark.asyncio
async def test_infra_failures_raise_a_typed_error(sandbox, tmp_path, verifier_bundle, monkeypatch):
    tests = write_tests(tmp_path, test_a="def test_a(): pass\n")

    async def no_upload(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sandbox, "upload_dir", no_upload)
    with pytest.raises(VerifierInfraError, match="disk full"):
        await run(sandbox, tests, verifier_bundle)


@pytest.mark.asyncio
async def test_prepare_step_failure_raises(sandbox, tmp_path, verifier_bundle, monkeypatch):
    tests = write_tests(tmp_path, test_a="def test_a(): pass\n")
    real = sandbox.exec

    async def broken(command, *a, **k):
        if "PREP_OK" in command:
            raise RuntimeError("container is not running")
        return await real(command, *a, **k)

    monkeypatch.setattr(sandbox, "exec", broken)
    with pytest.raises(VerifierInfraError, match="prepare"):
        await run(sandbox, tests, verifier_bundle)


@pytest.mark.asyncio
async def test_prep_runs_as_root_and_the_tests_as_the_requested_user(sandbox, tmp_path, verifier_bundle):
    await run(
        sandbox,
        write_tests(tmp_path, test_a="def test_a(): pass\n"),
        verifier_bundle,
        user="checker",
    )
    by_cmd = {("run" if "run.py" in c.command else "other"): c.user for c in sandbox.calls}
    assert by_cmd["run"] == "checker" and by_cmd["other"] == "root"
    run_cmd = next(c for c in sandbox.calls if "run.py" in c.command).command
    assert run_cmd.startswith(
        "timeout -k 5 60 env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin "
    )
    assert f"{sys.executable} -I -S -B " in run_cmd  # an absolute interpreter, never found through the PATH


# ── the stream protocol, without a sandbox ───────────────────────────────


def _payload(nonce="n", **kw):
    base = {
        "nonce": nonce, "exit_code": 0, "outcomes": {"a": "passed"}, "collect_errors": 0,
        "tests_before": "T", "tests_after": "T", "bundle": "B", "bundle_after": "B", "log_tail": "ok",
    }  # fmt: skip
    return base | kw


def test_parse_stream_requires_exactly_one_pair_with_our_nonce():
    ok = f"noise\n<<HV-BEGIN n>>\n{json.dumps(_payload())}\n<<HV-END n>>\nmore"
    data, tamper = parse_stream(ok, "n")
    assert data["exit_code"] == 0 and tamper == []
    assert parse_stream("nothing here", "n") == (None, [])
    assert parse_stream(ok + ok, "n")[1] == ["stream:markers x2/2"]
    assert parse_stream(ok, "other") == (None, [])
    assert parse_stream("<<HV-BEGIN n>>\n{broken\n<<HV-END n>>", "n")[1] == ["stream:payload is not JSON"]
    swapped = f"<<HV-END n>>\n<<HV-BEGIN n>>\n{json.dumps(_payload())}"
    assert parse_stream(swapped, "n")[1]
    wrong = f"<<HV-BEGIN n>>\n{json.dumps(_payload('x'))}\n<<HV-END n>>"
    assert parse_stream(wrong, "n")[1] == ["stream:payload nonce mismatch"]


def test_interpret_flags_every_disagreement():
    ok = interpret(_payload(), "T", "B")
    assert ok.passed and ok.pass_fraction == 1.0 and not ok.tamper
    assert interpret(_payload(tests_after="X"), "T", "B").tamper_details == ["tests_dir_modified"]
    assert interpret(_payload(tests_before="X"), "T", "B").tamper_details == ["tests_dir_modified"]
    assert interpret(_payload(bundle_after="X"), "T", "B").tamper_details == ["bundle_modified"]
    # pytest says 0 although a test did not pass: nothing honest produces that
    liar = interpret(_payload(outcomes={"a": "passed", "b": "failed"}), "T", "B")
    assert "exit_code_disagrees_with_outcomes" in liar.tamper_details and not liar.passed
    assert not interpret(_payload(exit_code=1, outcomes={"a": "passed"}), "T", "B").passed
    assert not interpret(_payload(outcomes={}), "T", "B").passed
    assert interpret(_payload(outcomes={}), "T", "B").pass_fraction is None
    assert not interpret(_payload(collect_errors=1), "T", "B").passed


# ── the real pytest 8.4.1 bundle on every installed Python ───────────────


def _interpreters() -> list[str]:
    import subprocess

    found = {}
    for cand in (
        "/usr/bin/python3",
        *sorted(str(p) for p in Path.home().glob(".local/share/uv/python/*/bin/python3.*")),
    ):
        if not Path(cand).is_file() or cand.endswith("-config"):
            continue
        try:
            out = subprocess.run(
                [cand, "-c", "import sys;print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip() not in found:
            found[out.stdout.strip()] = cand
    return [v for k, v in found.items() if eval(k) >= (3, 9)]


@pytest.mark.parametrize(
    "python",
    _interpreters() or [pytest.param("", marks=pytest.mark.skip("no other Python found"))],
)
@pytest.mark.asyncio
async def test_real_bundle_runs_on_this_python(sandbox, tmp_path, python, monkeypatch):
    """pytest 8.4.1 plus its backports (what `scripts/build_verifier_bundle.sh` builds) works on Python 3.9 to 3.14."""
    monkeypatch.delenv(BUNDLE_ENV)  # the fixture points it at the offline copy; this test wants the real build
    bundle = default_bundle_dir()
    try:
        ensure_bundle(bundle)
    except BundleError:
        pytest.skip(f"real bundle not built (run scripts/build_verifier_bundle.sh): {bundle}")
    tests = write_tests(tmp_path, test_a="def test_a(): pass\ndef test_b(): assert False\n")
    r = await run_pytest(
        sandbox, tests, bundle_dir=bundle, python=python, timeout_sec=60, tests_target="$HA_TESTS_DIR",
        verifier_dir="$HA_VERIFIER_DIR", scratch_dir="$HA_SCRATCH_DIR",
    )  # fmt: skip
    assert r.pass_fraction == 0.5 and r.exit_code == 1 and not r.tamper and not r.no_result, r.log
    assert json.loads((bundle / "BUNDLE.json").read_text())["pytest"] == "8.4.1"


# ── compiled caches ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_planted_pyc_with_a_matching_header_is_never_loaded(sandbox, tmp_path, verifier_bundle):
    """A `.pyc` under a `__pycache__` is trusted when its header matches the source's mtime and size, which a root agent
    can forge. The stdlib's own `__pycache__` cannot be hashed (python runs write there), so the run points
    `-X pycache_prefix` at a directory that does not exist and no cached bytecode is ever read."""
    import importlib.util
    import marshal
    import os
    import struct
    import subprocess

    d = write_tests(
        tmp_path,
        helper="VALUE = 'the source'\n",
        test_a="import helper\n\n\ndef test_helper_comes_from_the_source():\n    assert helper.VALUE == 'the source'\n",
    )
    src = d / "helper.py"
    st = os.stat(src)
    code = compile("VALUE = 'a planted pyc'\n", str(src), "exec")
    pyc = Path(importlib.util.cache_from_source(str(src)))
    pyc.parent.mkdir(exist_ok=True)
    pyc.write_bytes(
        importlib.util.MAGIC_NUMBER
        + struct.pack("<III", 0, int(st.st_mtime) & 0xFFFFFFFF, st.st_size & 0xFFFFFFFF)
        + marshal.dumps(code)
    )
    # the control: a plain isolated interpreter does load the forged cache
    plain = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(d)!r}); import helper; print(helper.VALUE)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.stdout.strip() == "a planted pyc"
    r = await run(sandbox, d, verifier_bundle)
    assert r.passed and not r.tamper, r.log  # the cache directory is not part of the tests digest, and it was not read


@pytest.mark.asyncio
async def test_the_nonce_is_gone_from_disk_before_the_first_test_runs(sandbox, tmp_path, verifier_bundle):
    """The one-shot config is deleted by the runner as soon as it has read it, so a process that polls the verifier's
    directory (even a non-root one that can read it) has no window while the tests run."""
    d = write_tests(
        tmp_path,
        test_a="""
        import glob, os


        def test_no_config_file_is_left_for_anyone_to_read():
            assert glob.glob(os.environ["SCRATCH"] + "/.hv-*/out/config.json") == []
            assert glob.glob(os.environ["SCRATCH"] + "/.hv-*/ctl/config.json") == []
        """,
    )
    r = await run(sandbox, d, verifier_bundle, env={"SCRATCH": str(sandbox.scratch_dir)})
    assert r.passed and not r.tamper, r.log


@pytest.mark.asyncio
@pytest.mark.parametrize("rc", [143, 130, 1, 255])
async def test_a_forged_pair_then_a_signal_to_the_runner_is_not_a_pass(sandbox, tmp_path, verifier_bundle, rc):
    """A process the tests started (an agent's script) forges a pair through /proc and SIGTERMs the runner before it can
    print its own: one pair, our nonce, and an exit status that is not one of the timeout codes. The runner always exits
    0 after printing, so any other status means the pair is not its own."""

    class KillRunner:
        def __init__(self, sb):
            self.sb = sb

        def __getattr__(self, name):
            return getattr(self.sb, name)

        async def exec(self, command, *a, **k):
            res = await self.sb.exec(command, *a, **k)
            if "run.py" in command:
                begin = next(ln for ln in res.stdout.splitlines() if ln.startswith("<<HV-BEGIN "))
                nonce = begin.split()[1].rstrip(">")
                real = res.stdout.split(begin, 1)[1].split("<<HV-END", 1)[0]
                payload = json.loads(real)  # the real digests: a forger can compute them
                payload.update(exit_code=0, outcomes={"t::x": "passed"}, collect_errors=0)
                res.stdout = f"\n<<HV-BEGIN {nonce}>>\n{json.dumps(payload)}\n<<HV-END {nonce}>>\n"
                res.return_code = rc
            return res

    tests = write_tests(tmp_path, test_a="def test_a(): assert False\n")
    r = await run(KillRunner(sandbox), tests, verifier_bundle)
    assert not r.passed and r.no_result
    assert any(d.startswith("stream:result_with_runner_exit_") for d in r.tamper_details)


# ── hardening of the runner process ──────────────────────────────────────


@pytest.mark.asyncio
async def test_the_result_says_whether_the_runner_made_itself_non_dumpable(sandbox, tmp_path, verifier_bundle):
    """On Linux the runner sets PR_SET_DUMPABLE=0, so a script the tests run cannot read its memory (the nonce) or open
    its stdout pipe through /proc, root included unless it holds CAP_SYS_PTRACE. macOS has no `prctl`: the flag says so."""
    d = write_tests(tmp_path, test_a="def test_a(): pass\n")
    r = await run(sandbox, d, verifier_bundle)
    assert r.passed and isinstance(r.hardened, bool)
    if sys.platform == "linux":
        assert r.hardened is True
    else:
        assert r.hardened is False  # reported, not hidden


@pytest.mark.skipif(sys.platform != "linux", reason="prctl(PR_SET_DUMPABLE) is Linux only")
def test_undumpable_really_clears_the_dumpable_flag():
    import subprocess

    code = (
        "import ctypes; from honest_agents.verifier import _hv_run;"
        "libc = ctypes.CDLL(None); assert libc.prctl(3, 0, 0, 0, 0) == 1;"  # PR_GET_DUMPABLE: dumpable by default
        "assert _hv_run.undumpable(); assert libc.prctl(3, 0, 0, 0, 0) == 0; print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.stdout.strip() == "ok", out.stderr


def test_undumpable_never_raises_where_there_is_no_prctl(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ctypes(name, *a, **k):
        if name == "ctypes":
            raise ImportError("no ctypes in this interpreter")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_ctypes)
    assert _hv_run.undumpable() is False
