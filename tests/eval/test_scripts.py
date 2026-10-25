"""The shell entry points are valid bash and only pass flags that the Python CLIs accept."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from honest_agents.eval import report_file, run_eval

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
EVAL_SCRIPTS = sorted([*SCRIPTS.glob("eval_*.sh"), SCRIPTS / "h1_tmax_check.sh"])


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", EVAL_SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash(script):
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_h1_script_runs_the_two_models_through_run_eval_with_valid_flags():
    text = (SCRIPTS / "h1_tmax_check.sh").read_text()
    assert "Qwen/Qwen3.5-4B" in text and "allenai/tmax-4b" in text
    parser = run_eval.build_parser()
    known = {opt for a in parser._actions for opt in a.option_strings}
    # every `--flag` that follows `honest_agents.eval.run_eval` (continuation lines included) must exist
    blocks = re.findall(r"honest_agents\.eval\.run_eval((?:[^\n]*\\\n)*[^\n]*)", text)
    assert len(blocks) >= 3  # the rollouts, the paired differences and the comparison
    for block in blocks:
        flags = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", block))
        assert flags <= known, flags - known
    # the same split, same caps, same sampling for both arms: only the model and its tool parser differ
    assert text.count("run_one ") >= 4 and "--no-thinking" in text


def test_harness_script_uses_the_report_file_cli_and_harbor_flags():
    text = (SCRIPTS / "eval_harness.sh").read_text()
    for sub in ("link-split", "instructions", "path"):
        assert f"report_file {sub}" in text
    for flag in ("--extra-instruction-path", "--artifact", "-a", "-m", "-k"):
        assert flag in text
    # the CLI subcommands the script calls exist
    with pytest.raises(SystemExit) as e:
        report_file.main(["--help"])
    assert e.value.code == 0


@pytest.mark.parametrize("script", ["eval_harness.sh", "eval_tb.sh", "eval_agentic_abstention.sh"])
def test_harbor_eval_scripts_run_in_the_umask_pinned_docker_environment(script):
    """DEF-16: a stock `-e docker` sandbox keeps the daemon's umask (0000 under OrbStack) for every agent (terminus-2
    included) and for a separate verifier; the scripts pass the pinned `DockerEnvironment` subclass instead."""
    text = (SCRIPTS / script).read_text()
    assert "harbor run" in text
    assert re.search(r"(?<![\\w-])-e docker\b", text) is None
    assert "-e honest_agents.eval.harbor_env:UmaskDockerEnvironment " in text
