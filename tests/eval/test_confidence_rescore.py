"""`eval.confidence_rescore` against a fake completions server: rows, resume, and the prompt-token check."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from honest_agents.eval import confidence_rescore as cr
from tests.sft.helpers import make_row, write_bank


def _bank(tmp_path: Path) -> Path:
    rows = []
    for k in range(3):
        r = make_row(f"src__t{k}__feasible", passed=k != 1, episode_id=f"e{k}", spec={"pair_id": f"src/t{k}"},
                     original={"status": "complete", "completed": ["x"], "confidence": 1})  # fmt: skip
        rows.append(r.model_copy(update={"task": {**r.task, "pair_id": f"src/t{k}"}}))
    return write_bank(tmp_path / "bank", rows)


def _args(tmp_path: Path, **kw) -> argparse.Namespace:
    base = {"base_url": "x", "model": "policy", "tokenizer": "Qwen/Qwen3.5-4B", "family": "qwen3_5", "samples": 3,
            "temperature": 1.0, "seed": 0, "concurrency": 2}  # fmt: skip
    return argparse.Namespace(**(base | kw))


def _fake_client(monkeypatch, texts: list[str], prompt_tokens: int | None = None):
    seen: list[str] = []

    class Completions:
        async def create(self, *, prompt, n, **_):
            from transformers import AutoTokenizer

            seen.append(prompt)
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
            n_tok = prompt_tokens if prompt_tokens is not None else len(tok(prompt, add_special_tokens=False).input_ids)
            return SimpleNamespace(choices=[SimpleNamespace(index=i, text=texts[i % len(texts)]) for i in range(n)],
                                   usage=SimpleNamespace(prompt_tokens=n_tok))  # fmt: skip

    class Client:
        def __init__(self, **_):
            self.completions = Completions()

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    return seen


@pytest.fixture(autouse=True)
def _tokenizer():
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")


def test_rows_hold_the_samples_their_mean_and_the_label(tmp_path: Path, monkeypatch):
    seen = _fake_client(monkeypatch, ["0.8\n</parameter>", "x\n", "0.4\n"])
    items, dropped = cr.items_of([_bank(tmp_path)])
    out = tmp_path / "rows.jsonl"
    assert asyncio.run(cr._score(items, _args(tmp_path), out)) == 3 and dropped == {}
    rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["e1"]["y"] == 0 and rows["e0"]["y"] == 1
    assert rows["e0"]["samples"] == [0.8, None, 0.4] and rows["e0"]["mean"] == pytest.approx(0.6)
    assert rows["e0"]["parsed"] == 2 and rows["e0"]["original_confidence"] == 1.0
    assert all(p.endswith("<parameter=confidence>\n") for p in seen)


def test_a_prompt_the_server_tokenises_differently_is_refused(tmp_path: Path, monkeypatch):
    _fake_client(monkeypatch, ["0.5\n"], prompt_tokens=3)
    items, _ = cr.items_of([_bank(tmp_path)])
    with pytest.raises(cr.PromptMismatchError):
        asyncio.run(cr._score(items[:1], _args(tmp_path), tmp_path / "rows.jsonl"))


def test_a_heldout_bank_is_scored_on_held_out_sources_only(tmp_path: Path):
    from honest_agents.train.confidence import heldout

    items, dropped = cr.items_of([], [_bank(tmp_path)], fraction=0.5, seed="s")
    expected = [f"src/t{k}" for k in range(3) if heldout(f"src/t{k}", 0.5, "s")]
    assert sorted(i.source for i in items) == expected
    assert dropped.get("training_source", 0) == 3 - len(expected)

