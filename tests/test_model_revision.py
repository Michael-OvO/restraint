"""DEF-51: the base model's Hub revision is pinned in one place (`models.PINNED_REVISIONS`), and every config, the serve
scripts, the merge and the colocated trainer resolve a pinned Hub id to it. L0: no network, no GPU (a fake `uv` stands in
for the vLLM server)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from honest_agents import models
from honest_agents.pilot.run import load_pilot_config
from honest_agents.sft.dpo import load_dpo_config
from honest_agents.sft.train_sft import load_config as load_sft_config
from honest_agents.train import load_run_config

REPO = Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
MERGED = "outputs/it14/sft_v2_e3_merged"


def _load(path: Path):
    """The config through the loader that owns it (the same split as tests/test_scripts.py and scripts/_common.sh)."""
    if path.name == "pilot.yaml":
        return load_pilot_config(path)
    if path.name.startswith("sft_dpo_"):
        return load_dpo_config(path).sft
    if path.name.startswith("sft_"):
        return load_sft_config(path)
    return load_run_config(path)


def test_the_pins_are_commits_and_the_benchmark_base_arm_uses_the_same_one():
    from honest_agents.eval.final_benchmark import arms

    assert models.PINNED_REVISIONS["Qwen/Qwen3.5-4B"] == models.QWEN3_5_4B_REVISION
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in models.PINNED_REVISIONS.values())
    base = arms()["base"]
    assert base["revision"] == models.PINNED_REVISIONS[base["model"]]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_every_config_resolves_its_hub_model_to_the_pin(config):
    cfg = _load(config)
    if cfg.model.name.startswith("outputs/"):  # a merged model: a local directory, no Hub revision
        assert cfg.model.revision is None
        return
    assert cfg.model.name in models.PINNED_REVISIONS, f"{config.name}: pin {cfg.model.name} in models.PINNED_REVISIONS"
    assert cfg.model.revision == models.PINNED_REVISIONS[cfg.model.name]


def test_a_merged_model_has_no_revision_and_a_conflicting_revision_is_refused():
    assert load_run_config(REPO / "configs/grpo_it13_2gpu.yaml", [f"model.name={MERGED}"]).model.revision is None
    assert models.pinned_revision(MERGED) is None
    with pytest.raises(ValueError, match="differs from its pin"):
        load_run_config(REPO / "configs/qwen35_4b.yaml", ["model.revision=main"])


def test_the_revision_cli_prints_the_pin():
    done = subprocess.run(
        [sys.executable, "-m", "honest_agents.models", "revision", "Qwen/Qwen3.5-4B"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert done.stdout.strip() == models.QWEN3_5_4B_REVISION


FAKE_UV = """#!{python}
# stands in for `uv run [--flags] CMD...`: runs python commands, prints a `vllm` command instead of serving
import json, os, sys
args = sys.argv[2:]
while args and args[0].startswith("--"):
    args = args[2:] if args[0] == "--extra" else args[1:]
if args[0] == "vllm":
    print("VLLM " + json.dumps(args))
    sys.exit(0)
if args[0] == "python":
    args[0] = sys.executable
os.execv(args[0], args)
"""


def _serve(tmp_path: Path, *args: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("SLOT", "VLLM_GPUS", "VLLM_PORT", "EVAL")}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    done = subprocess.run(
        ["bash", str(REPO / "scripts/serve_vllm.sh"), *args], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]
    (line,) = [ln for ln in done.stdout.splitlines() if ln.startswith("VLLM ")]
    return json.loads(line.removeprefix("VLLM "))


def _flag(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_the_serve_script_reads_the_revision_with_the_model():
    script = (REPO / "scripts/serve_vllm.sh").read_text()
    assert "read -r MODEL FAMILY REVISION < <(model_from_config" in script
    assert '--revision "$REVISION" --tokenizer-revision "$REVISION"' in script
    assert 'cfg.model.revision or ""' in (REPO / "scripts/_common.sh").read_text()


@pytest.mark.slow  # four Python start-ups through the fake uv (~7 s)
def test_the_vllm_server_serves_a_hub_base_at_its_pin_and_a_merged_model_without_one(tmp_path):
    cmd = _serve(tmp_path, "configs/grpo_it13_2gpu.yaml")
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3.5-4B"]
    assert _flag(cmd, "--revision") == _flag(cmd, "--tokenizer-revision") == models.QWEN3_5_4B_REVISION
    cmd = _serve(tmp_path, "configs/grpo_it13_2gpu.yaml", f"model.name={MERGED}")
    assert cmd[2] == MERGED and "--revision" not in cmd and "--tokenizer-revision" not in cmd


def test_the_pilot_prints_its_pin_for_the_serve_script(capsys):
    from honest_agents.pilot import run

    assert run.main(["--config", str(REPO / "configs/pilot.yaml"), "--print-model"]) == 0
    assert capsys.readouterr().out.split() == ["Qwen/Qwen3.5-4B", "qwen3_5", models.QWEN3_5_4B_REVISION]


def test_the_other_serve_and_download_scripts_use_the_pin():
    h1 = (REPO / "scripts/h1_tmax_check.sh").read_text()
    assert '"${BASE_REVISION:-$(pinned_revision "$BASE_MODEL")}"' in h1 and '--tokenizer-revision "$rev"' in h1
    setup = (REPO / "scripts/setup_node.sh").read_text()
    assert 'rev="$(pinned_revision "$m")"' in setup and '${rev:+--revision "$rev"}' in setup


def test_the_merge_downloads_a_hub_base_at_its_pin(tmp_path, monkeypatch):
    import huggingface_hub

    from honest_agents.sft import merge

    asked = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: asked.append((repo, kw)) or str(tmp_path)
    )
    (tmp_path / "out").mkdir()
    merge.restore_base_tensors("Qwen/Qwen3.5-4B", tmp_path / "out", "bfloat16")
    assert asked == [("Qwen/Qwen3.5-4B", {"revision": models.QWEN3_5_4B_REVISION})]


def test_the_colocated_engine_loads_the_pinned_snapshot(monkeypatch, tmp_path):
    """TRL builds the colocated vLLM engine from the model's `name_or_path` with no revision: the trainer gets the pinned
    snapshot directory in colocate mode, and the name in server mode or for a merged model."""
    import huggingface_hub

    from honest_agents.train.run import trainer_model

    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: asked.append((repo, kw)) or "/snap")
    colocate = load_run_config(REPO / "configs/grpo_colocate_qwen35.yaml")
    assert trainer_model(colocate) == "/snap"
    assert asked == [("Qwen/Qwen3.5-4B", {"revision": models.QWEN3_5_4B_REVISION})]
    assert trainer_model(load_run_config(REPO / "configs/grpo_it13_2gpu.yaml")) == "Qwen/Qwen3.5-4B"
    merged = tmp_path / "merged"
    merged.mkdir()
    assert trainer_model(load_run_config(REPO / "configs/grpo_colocate_qwen35.yaml", [f"model.name={merged}"])) == str(
        merged
    )
    assert len(asked) == 1
