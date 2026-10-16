"""Model soup on tiny synthetic safetensors checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from honest_agents.sft import soup as sp


def make_ck(path: Path, scale: float, sharded: bool = True) -> Path:
    path.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    t = {
        "w": (torch.randn(4, 4, generator=g) * scale).bfloat16(),
        "layers.0.A_log": torch.arange(4, dtype=torch.float32) * scale,
        "mtp.w": (torch.randn(3, generator=g) * scale).bfloat16(),
        "same": torch.randn(2, 2, generator=g).bfloat16(),
    }
    if sharded:
        shards = {"model-1.safetensors": ["w", "same"], "model-2.safetensors": ["layers.0.A_log", "mtp.w"]}
        for f, ks in shards.items():
            save_file({k: t[k] for k in ks}, str(path / f), metadata={"format": "pt"})
        wm = {k: f for f, ks in shards.items() for k in ks}
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    else:
        save_file(t, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps({"scale": scale}))
    return path


def load_all(d: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in d.glob("*.safetensors"):
        out.update(load_file(str(f)))
    return out


def test_identical_inputs_are_preserved_exactly(tmp_path):
    # The first input has `sft.merge`'s layout (one model.safetensors, no index); the second is sharded.
    a, b = make_ck(tmp_path / "a", 1.0, sharded=False), make_ck(tmp_path / "b", 1.0)
    rep = sp.soup([str(a), str(b)], str(tmp_path / "o"))
    o, ref = load_all(tmp_path / "o"), load_all(a)
    assert set(o) == set(ref)
    for k in ref:
        assert o[k].dtype == ref[k].dtype and torch.equal(o[k], ref[k])
    assert rep["tensors_identical_across_inputs"] == 4 and rep["tensors_differing"] == 0
    assert sorted(f.name for f in (tmp_path / "o").glob("*.safetensors*")) == ["model.safetensors"]  # a's layout


def test_average_weights_dtypes_and_files(tmp_path):
    a, b = make_ck(tmp_path / "a", 1.0), make_ck(tmp_path / "b", 3.0)
    (a / "merge_info.json").write_text("{}")  # the input's provenance stays with the input
    (tmp_path / "o").mkdir()
    save_file({"stale": torch.zeros(2)}, str(tmp_path / "o" / "model-9.safetensors"))  # an earlier layout
    rep = sp.soup([str(a), str(b)], str(tmp_path / "o"), weights=[3, 1])
    o, ta, tb = load_all(tmp_path / "o"), load_all(a), load_all(b)
    assert set(o) == set(ta)  # the stale shard is gone
    assert o["layers.0.A_log"].dtype == torch.float32 and o["mtp.w"].dtype == torch.bfloat16
    assert torch.allclose(o["layers.0.A_log"], 0.75 * ta["layers.0.A_log"] + 0.25 * tb["layers.0.A_log"])
    want = (0.75 * ta["w"].float() + 0.25 * tb["w"].float()).bfloat16()
    assert torch.equal(o["w"], want) and torch.equal(o["same"], ta["same"])
    assert json.loads((tmp_path / "o" / "config.json").read_text()) == {"scale": 1.0}
    assert not (tmp_path / "o" / "merge_info.json").exists()
    idx = json.loads((tmp_path / "o" / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(o)
    assert idx["weight_map"]["w"] == "model-1.safetensors" and idx["weight_map"]["mtp.w"] == "model-2.safetensors"
    assert idx["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in o.values())
    assert rep["weights"] == [0.75, 0.25] and "layers.0.A_log" in rep["max_abs_diff_between_inputs"]
    assert rep["tensors_total"] == 4 and rep["tensors_differing"] == 3
    assert (tmp_path / "o" / "soup_report.json").exists()


def test_extrapolation(tmp_path):
    base, a, b = (make_ck(tmp_path / n, s) for n, s in (("base", 0.0), ("a", 1.0), ("b", 3.0)))
    sp.soup([str(a), str(b)], str(tmp_path / "o"), base=str(base), factor=2.0)
    o = load_all(tmp_path / "o")
    # base A_log = 0, avg = 2 * arange, so out = 4 * arange
    assert torch.allclose(o["layers.0.A_log"], torch.arange(4, dtype=torch.float32) * 4)


def test_mismatches_are_refused(tmp_path):
    a = make_ck(tmp_path / "a", 1.0)
    b = make_ck(tmp_path / "b", 1.0)
    (tmp_path / "b" / "model-2.safetensors").unlink()
    (tmp_path / "b" / "model.safetensors.index.json").unlink()  # now only model-1: w, same
    with pytest.raises(SystemExit, match="tensor names differ"):
        sp.soup([str(a), str(b)], str(tmp_path / "o"))
    c = make_ck(tmp_path / "c", 1.0)
    d = make_ck(tmp_path / "d", 1.0)
    save_file(
        {"w": torch.zeros(5, 4).bfloat16(), "same": torch.zeros(2, 2).bfloat16()},
        str(tmp_path / "d" / "model-1.safetensors"),
    )
    with pytest.raises(SystemExit, match="shape of w"):
        sp.soup([str(c), str(d)], str(tmp_path / "o2"))
    with pytest.raises(SystemExit, match="--factor needs --base"):
        sp.soup([str(a), str(c)], str(tmp_path / "o3"), factor=2.0)
    e = make_ck(tmp_path / "e", 1.0)
    te = load_all(e)
    save_file(
        {"layers.0.A_log": te["layers.0.A_log"].bfloat16(), "mtp.w": te["mtp.w"]},
        str(e / "model-2.safetensors"),
        metadata={"format": "pt"},
    )
    with pytest.raises(SystemExit, match="dtype of layers.0.A_log differs"):
        sp.soup([str(a), str(e)], str(tmp_path / "o4"))
    assert not (tmp_path / "o4").exists()  # refused before anything was written
    with pytest.raises(SystemExit, match="is one of the inputs"):
        sp.soup([str(a), str(c)], str(a))
    assert set(load_all(a)) == {"w", "same", "layers.0.A_log", "mtp.w"}  # untouched
