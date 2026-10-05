"""Documented commands cannot rot: every script parses, every config dry-runs, and every command, script, module, config,
flag and link that the docs and the script headers name exists. Runs on a Mac (no Docker, no GPU, no network)."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

from honest_agents.train import load_run_config
from tests.test_pipeline_integration import built  # noqa: F401  (module-scoped fixture: a small built dataset)

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", *(REPO / "docs" / f for f in ("node-runbook.md", "roadmap.md", "evaluation.md"))]
SHELL_SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))
# sft_ configs and the pilot have their own schemas and need a trajectory bank to dry-run (tests/sft, tests/pilot)
TRAIN_CONFIGS = sorted(
    p.name for p in (REPO / "configs").glob("*.yaml") if not p.name.startswith("sft_") and p.name != "pilot.yaml"
)
# paths the docs name that exist only after a run, or placeholders
GENERATED = ("data/", "outputs/", "runs/", "benchmarks", "<", "*", "/tmp", "$")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False, **kw)


# ── scripts parse ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_script_is_valid_bash_and_executable(script):
    done = _run(["bash", "-n", str(script)])
    assert done.returncode == 0, f"{script.name}: {done.stderr}"
    if script.name != "_common.sh" and script.name != "eval_common.sh":  # the two sourced libraries
        assert script.stat().st_mode & 0o111, f"{script.name} is not executable"
        assert script.read_text().startswith("#!/usr/bin/env bash"), f"{script.name}: no bash shebang"


@pytest.mark.parametrize("script", sorted((REPO / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_python_script_parses(script):
    ast.parse(script.read_text())


# ── configs dry-run ──────────────────────────────────────────────────────────


def _data_overrides(config: Path, data: Path) -> list[str]:
    """Point only the dataset files the config actually uses (phase0 has no validation file: giving it one turns on the probe)."""
    cfg = load_run_config(config)
    out = []
    if cfg.data.train_file:
        name = "train_pool.jsonl" if "pool" in str(cfg.data.train_file) else "train.jsonl"
        if "smoke" in str(cfg.data.train_file):
            name = "train.jsonl"
        out.append(f"data.train_file={data / name}")
    if cfg.data.eval_file:
        out.append(f"data.eval_file={data / 'val.jsonl'}")
    return out


@pytest.mark.parametrize("config", TRAIN_CONFIGS)
def test_every_training_config_dry_runs(config, built):  # noqa: F811
    path = REPO / "configs" / config
    done = _run(
        [sys.executable, "-m", "honest_agents.train", "--config", str(path), "--dry-run", *_data_overrides(path, built)]
    )
    assert done.returncode == 0, f"{config}: {done.stderr[-800:]}"


@pytest.mark.parametrize("config", ["base.yaml", "smoke.yaml", "colocate_1gpu.yaml"])
def test_training_configs_print_limits(config):
    """scripts/serve_vllm.sh and scripts/_common.sh read `--print-limits` to size the vLLM window."""
    done = _run(
        [sys.executable, "-m", "honest_agents.train", "--config", str(REPO / "configs" / config), "--print-limits"]
    )
    assert done.returncode == 0 and '"max_model_len"' in done.stdout, f"{config}: {done.stderr[-500:]}"


def test_the_colocated_phase0_config_differs_from_phase0_only_in_the_engine():
    a, b = load_run_config(REPO / "configs/phase0_zero_lr.yaml"), load_run_config(REPO / "configs/phase0_colocate.yaml")
    assert b.grpo["vllm_mode"] == "colocate" and a.grpo["vllm_mode"] == "server"
    assert (a.grpo["learning_rate"], a.grpo["max_steps"], a.data.train_file) == (
        b.grpo["learning_rate"],
        b.grpo["max_steps"],
        b.data.train_file,
    )


# ── docs name only things that exist ─────────────────────────────────────────


def _doc_text(path: Path) -> str:
    return re.sub(r"\\\n\s*", " ", path.read_text())  # join backslash continuations


def _command_lines() -> list[tuple[str, str]]:
    """(doc, command) for each line of a fenced code block, plus each header usage line of a script (`#   scripts/x.sh ...`)."""
    out = []
    for doc in DOCS:
        for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", _doc_text(doc), re.DOTALL):
            out += [(doc.name, ln.strip()) for ln in block.splitlines() if ln.strip() and not ln.startswith("#")]
    for script in sorted((REPO / "scripts").glob("*.sh")):
        for ln in _doc_text(script).splitlines():
            m = re.match(r"#\s{2,}((?:[A-Z_]+=\S+ )*(?:scripts/|python -m|uv run|env -u)\S.*)", ln)
            if m:
                out.append((script.name, m.group(1)))
    return out


COMMANDS = _command_lines()


def _paths(text: str, pattern: str) -> set[str]:
    return {p for p in re.findall(pattern, text) if not any(g in p for g in GENERATED)}


def test_the_docs_and_headers_have_commands_to_check():
    assert len(COMMANDS) > 60, "the extraction found too few commands: the doc or header format changed"
    assert {d for d, _ in COMMANDS} >= {"README.md", "node-runbook.md", "evaluation.md", "run_sft.sh"}


def test_every_script_and_config_a_command_names_exists():
    missing = []
    for doc, cmd in COMMANDS:
        for s in _paths(cmd, r"(?<![\w/.-])(scripts/[\w.-]+\.(?:sh|py))"):
            if not (REPO / s).exists():
                missing.append(f"{doc}: {s}")
        for c in _paths(cmd, r"(?<![\w/.-])(configs/[\w./-]+\.yaml)"):
            if not (REPO / c).exists():
                missing.append(f"{doc}: {c}")
    assert not missing, sorted(set(missing))


# scripts that hand their trailing flags to another program: vLLM, `harbor run`, or `honest_agents.eval.run_eval`
PASS_THROUGH = {"serve_vllm.sh", "eval_harness.sh", "eval_tb.sh", "run_pilot.sh", "sweep_alpha.sh"}
RUN_EVAL_ARGS = {"eval_heldout.sh", "h1_tmax_check.sh"}


def test_every_script_flag_a_command_names_is_parsed_by_the_script():
    run_eval_help = _help("honest_agents.eval.run_eval").stdout
    bad = []
    for doc, cmd in COMMANDS:
        m = re.search(r"(?<![\w/.-])(scripts/[\w.-]+\.sh)((?: +\S+)*)", cmd)
        if not m or not (REPO / m.group(1)).exists() or Path(m.group(1)).name in PASS_THROUGH:
            continue
        source = (REPO / m.group(1)).read_text()
        if Path(m.group(1)).name in RUN_EVAL_ARGS:
            source += run_eval_help
        own = m.group(2).split(" -- ")[0].split("|")[0]  # after `--` the flags belong to another program
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", own):
            if flag not in source:
                bad.append(f"{doc}: {m.group(1)} has no {flag}")
    assert not bad, sorted(set(bad))


def _modules() -> list[str]:
    mods = set()
    for doc, cmd in COMMANDS:
        mods |= set(re.findall(r"-m (honest_agents(?:\.\w+)+)", cmd))
    for script in SHELL_SCRIPTS:
        mods |= set(re.findall(r"-m (honest_agents(?:\.\w+)+)", script.read_text()))
    return sorted(mods)


@cache
def _help(module: str) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", module, "--help"])


@pytest.mark.parametrize("module", _modules())
def test_module_cli_help_works_and_knows_the_documented_flags(module):
    done = _help(module)
    assert done.returncode == 0, f"{module} --help: {done.stderr[-500:]}"
    if "{" in done.stdout.split("options:")[0]:  # sub-commands: their flags are in their own --help
        return
    unknown = []
    for doc, cmd in COMMANDS:
        if f"-m {module}" not in cmd:
            continue
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", cmd.split(f"-m {module}", 1)[1]):
            if flag not in done.stdout:
                unknown.append(f"{doc}: {flag}")
    assert not unknown, sorted(set(unknown))


def test_console_scripts_point_at_real_callables():
    text = (REPO / "pyproject.toml").read_text()
    for name, target in re.findall(r'^(honest-[\w-]+) = "([\w.]+:\w+)"', text, re.MULTILINE):
        module, func = target.split(":")
        mod = __import__(module, fromlist=[func])
        assert callable(getattr(mod, func)), f"{name} -> {target}"


def test_markdown_links_and_named_source_paths_resolve():
    bad = []
    for doc in DOCS:
        text = doc.read_text()
        for link in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if "://" in link or link.startswith("mailto:"):
                continue
            if not (doc.parent / link).exists():
                bad.append(f"{doc.name}: link {link}")
        for p in _paths(text, r"`((?:src/honest_agents|docker|docs|tests/test_)[\w./-]+?)`"):
            if not (REPO / p.rstrip("/")).exists():
                bad.append(f"{doc.name}: path {p}")
    assert not bad, sorted(set(bad))
