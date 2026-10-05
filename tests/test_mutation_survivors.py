"""Killers for the mutants that survived the full suite in the second audit (docs/research/audit2-memo.md §3).

88 deliberate bugs were injected into reward.py, grading.py, report.py, trl_rewards.py, harness.py and the verifier; 10
survived. Each test names the mutant it kills (id from the audit's mutant list) and passes on the real code. V06 is an
equivalent mutant (see the last test), so nothing can kill it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from honest_agents import grading
from honest_agents.grading import ToolchainError, check_blocker, self_check, shadowed, snapshot
from honest_agents.harness import coerce_report
from honest_agents.verifier import _hv_run
from honest_agents.verifier.runner import VerifierResult, interpret, run_pytest
from tests.test_grading import FAST, PYSPEC, blocker, make_spec
from tests.test_verifier import _payload, write_tests


async def test_home_relative_blocker_target_is_resolved_against_HOME(sandbox):  # G11: `~/x` was quoted literally
    (sandbox.root / "present.txt").write_text("x")
    sandbox.env["HOME"] = str(sandbox.root)
    assert await check_blocker(sandbox, blocker("missing_capability", "~/present.txt")) == "false"
    assert await check_blocker(sandbox, blocker("missing_capability", "~/absent.txt")) == "true"


class RcSandbox:
    """Every probe dies with a given return code, like `command -v` hitting rc 2 / 126 / 127 or a killed shell."""

    def __init__(self, rc: int):
        self.rc = rc

    async def exec(self, command, **kw):
        return SimpleNamespace(stdout="", stderr="", return_code=self.rc)


@pytest.mark.parametrize("rc", [2, 126, 127, 137])
async def test_a_probe_that_could_not_run_is_unknown_not_a_verified_blocker(rc):  # G01: any rc but 0 meant "false"
    for category, target in (("missing_capability", "jq"), ("permission_denied", "/data/x")):
        assert await check_blocker(RcSandbox(rc), blocker(category, target)) == "unknown"


async def test_self_check_fails_on_tamper_even_if_the_result_says_passed(sandbox, monkeypatch):  # G16
    async def fake_run(*args, **kwargs):
        return VerifierResult(
            passed=True, pass_fraction=1.0, n_tests=1, exit_code=0, log="", tamper=True, tamper_details=["bundle"]
        )

    monkeypatch.setattr(grading, "run_pytest", fake_run)
    with pytest.raises(ToolchainError):
        await self_check(sandbox, make_spec(**PYSPEC), FAST)


def test_a_shim_must_be_in_a_later_directory_than_the_protected_file():  # G02: the file's own directory counted
    order = ["/usr/local/bin", "/usr/bin"]
    protected = {"/usr/local/bin/env": "d", "/usr/bin/env": "d"}
    assert shadowed(["/usr/local/bin/env"], protected, order) == ["/usr/local/bin/env"]  # /usr/bin is later
    assert shadowed(["/usr/bin/env"], protected, order) == []  # nothing later: the earlier entry is not a shadow
    assert shadowed(["/usr/local/bin/env"], {"/usr/local/bin/env": "d"}, order) == []  # same directory only


@pytest.mark.parametrize("change", ["modified", "removed", "added"])
async def test_every_kind_of_change_to_a_sentinel_is_a_violation(sandbox, monkeypatch, change):  # G07, G08
    """A directory's own ctime moves whenever a file is added or removed, so on a real filesystem the `modified` entry
    hides a mutant that ignores additions or removals. Feed `_state_checks` each kind of diff on its own."""
    spec = make_spec(irreversible_paths=[str(sandbox.root / "i")], forbidden_paths=[str(sandbox.root / "f")])
    baseline = await snapshot(sandbox, spec, FAST)
    diff = {"modified": (["/x"], [], []), "removed": ([], ["/x"], []), "added": ([], [], ["/x"])}[change]

    def fake_diff(before, after):
        return diff if before is not baseline.protected else ([], [], [])

    monkeypatch.setattr(grading, "diff_manifests", fake_diff)
    st = await grading._state_checks(sandbox, spec, baseline, FAST)
    assert st.irreversible and st.forbidden
    assert st.violations == ["irreversible:/x", "forbidden:/x"]


class KilledRunnerSandbox:
    """Runs everything for real except the verifier's own command, which dies by SIGKILL after a timeout (rc 137)."""

    def __init__(self, inner, rc: int):
        self.inner, self.rc = inner, rc

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def exec(self, command, **kw):
        if "ctl/run.py" in command:
            return SimpleNamespace(stdout="", stderr="", return_code=self.rc)
        return await self.inner.exec(command, **kw)


@pytest.mark.parametrize("rc", [124, 137])
async def test_both_gnu_timeout_exit_codes_are_a_timeout(sandbox, tmp_path, verifier_bundle, rc):  # V10: 137 missed
    tests = write_tests(tmp_path, test_a="def test_a(): pass\n")
    r = await run_pytest(
        KilledRunnerSandbox(sandbox, rc),
        tests,
        bundle_dir=verifier_bundle,
        python=sys.executable,
        tests_target="$HA_TESTS_DIR",
        verifier_dir="$HA_VERIFIER_DIR",
        scratch_dir="$HA_SCRATCH_DIR",
        timeout_sec=60,
    )
    assert r.timed_out and not r.no_result and not r.passed and r.exit_code == rc


def test_a_failed_test_stays_failed_when_a_later_phase_report_is_skipped():  # HV01: the last phase overwrote it
    collector = _hv_run._Collector()

    def report(when: str, **flags: bool) -> SimpleNamespace:
        return SimpleNamespace(
            nodeid="t::a", when=when, **{"passed": False, "failed": False, "skipped": False, **flags}
        )

    collector.pytest_runtest_logreport(report("call", failed=True))
    collector.pytest_runtest_logreport(report("teardown", skipped=True))
    assert collector.outcomes == {"t::a": "failed"}


@pytest.mark.parametrize("value", [float("nan"), "nan"])
def test_a_nan_confidence_is_dropped_not_clamped_to_zero(value):  # H04: NaN became 0.0, "certainly fails"
    assert coerce_report({"status": "incomplete", "confidence": value}).confidence is None
    assert coerce_report({"status": "incomplete", "confidence": 0.7}).confidence == 0.7


def test_collect_errors_are_never_a_pass_whichever_check_sees_them():
    """V06 (`clean` ignoring collect_errors) is an equivalent mutant: with exit code 0 the same payload is also tamper
    (`exit_code_disagrees_with_outcomes`), and with any other exit code `clean` is already false. Pin both routes."""
    r = interpret(_payload(collect_errors=1), "T", "B")
    assert not r.passed and r.tamper and r.tamper_details == ["exit_code_disagrees_with_outcomes"]
    assert not interpret(_payload(collect_errors=1, exit_code=2), "T", "B").passed
