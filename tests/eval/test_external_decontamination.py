"""Keeping the external benchmark texts out of training: the build decontaminates against them.

`scripts/fetch_benchmarks.sh` lays Agentic Abstention (terminal immediate rewrites and the TB2 sources they rewrite) and
AgentAbstain (all 526 task instructions) out as `<label>/<task>/instruction.md` next to terminal-bench-2 and TBLite;
`tasks.dedup` reads that directory and the build refuses to run without it. These tests pin the three links of the chain:
the fetch script writes the labels, the deduplicator catches their texts (identical and near-identical), and the build
stage drops the matching seeds (a pool made only of them fails before anything is emitted).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from honest_agents.tasks import build as task_build
from honest_agents.tasks.build.seeds import collect_seeds, dedup_seeds_report
from honest_agents.tasks.dedup import Deduplicator, load_benchmark_tasks
from tests.tasks.synthetic import RICH_INSTRUCTION, plain_task
from tests.test_data_leakage import _args, _make_seeds

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
EXTERNAL_LABELS = ("agentic-abstention", "agentabstain")
# What fetch_benchmarks.sh names a task dir of each external label (`<task>__<variant>`, `<pair>__<type>`).
LAYOUT = {
    "agentic-abstention": "log-summary__false_premise_or_contradiction",
    "agentabstain": "ambiguous_action_specification_preview_001__abstain",
}
NEAR_DUPLICATE_SUFFIX = " Also keep a copy of the result under the archive directory."


def _bench(root: Path, label: str, text: str) -> Path:
    d = root / label / LAYOUT[label]
    d.mkdir(parents=True)
    (d / "instruction.md").write_text(text)
    return root


def test_the_fetch_script_writes_both_external_benchmarks_as_instruction_dirs():
    text = (SCRIPTS / "fetch_benchmarks.sh").read_text()
    for label in EXTERNAL_LABELS:
        assert f'out / "{label}"' in text, f"fetch_benchmarks.sh no longer writes the {label} instructions"
    assert "instruction.md" in text and "terminal/data/immediate" in text and "tasks.jsonl" in text
    assert re.search(r"BENCH_DIR", (SCRIPTS / "build_dataset.sh").read_text())


@pytest.mark.parametrize("label", EXTERNAL_LABELS)
@pytest.mark.parametrize("suffix", ["", NEAR_DUPLICATE_SUFFIX], ids=["identical", "near-duplicate"])
def test_a_seed_that_reproduces_an_external_text_is_matched(tmp_path, label, suffix):
    bench = load_benchmark_tasks(_bench(tmp_path / "bench", label, RICH_INSTRUCTION + suffix))
    assert [b.benchmark for b in bench] == [label]
    hit = Deduplicator(bench, threshold=0.5).match("some-other-name", RICH_INSTRUCTION)
    assert hit is not None and hit[0].benchmark == label


def _config(tmp_path: Path, seeds: Path, bench: Path):
    return task_build.parse_args([*_args(tmp_path, seeds, "--benchmark-dir", str(bench)), "--neardup-threshold", "0"])


def test_the_build_stage_drops_external_texts_and_keeps_only_seeds_that_match_none_of_them(tmp_path):
    seeds = _make_seeds(tmp_path / "seeds", 3)  # three seeds that carry RICH_INSTRUCTION
    for name in ("other", "second", "third", "fourth"):  # seeds no external text contains
        plain_task(seeds / name)
    bench_root = tmp_path / "bench"
    for label in EXTERNAL_LABELS:  # the text appears in the two external benchmarks only, never in TB2 or TBLite
        _bench(bench_root, label, RICH_INSTRUCTION)
    cfg = _config(tmp_path, seeds, bench_root)
    kept, dropped, stats = dedup_seeds_report(cfg, collect_seeds(cfg, []))
    assert len(dropped) == 3 and {d["benchmark"] for d in dropped} <= set(EXTERNAL_LABELS)
    assert stats["n_dropped"] == 3 and set(stats["benchmarks"]) == set(EXTERNAL_LABELS)
    dd = Deduplicator(load_benchmark_tasks(bench_root), threshold=0.5)
    assert len(kept) == 4 and all(
        dd.match(s.task_id, (s.task_dir / "instruction.md").read_text()) is None for s in kept
    )


def test_a_build_whose_seeds_are_all_external_texts_fails_before_emitting_anything(tmp_path):
    seeds = _make_seeds(tmp_path / "seeds", 3)
    bench_root = tmp_path / "bench"
    for label in EXTERNAL_LABELS:
        _bench(bench_root, label, RICH_INSTRUCTION)
    with pytest.raises(ValueError, match="3 dropped as benchmark duplicates"):
        task_build.build(_config(tmp_path, seeds, bench_root))
    assert not list((tmp_path / "data").glob("*.jsonl")) and not (tmp_path / "data" / "tasks").exists()
