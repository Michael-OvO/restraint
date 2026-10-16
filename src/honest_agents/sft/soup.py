"""Average full merged HF checkpoints (a model soup), optionally extrapolated along the task vector from a base.

    python -m honest_agents.sft.soup --models A B [C ...] [--weights w1 w2 ...] [--base BASE --factor f] --out OUT

Every input is a directory of safetensors weights (single file or sharded) with identical tensor names and shapes
(checked). Tensors are averaged in float32 and cast back to the dtype they have in the inputs, so float32 tensors that
`sft.merge` keeps from the Qwen3.5 base (`A_log`, norms) and tensors the model class does not load (the MTP head) all
stay, with their dtypes; identical inputs reproduce the tensor exactly. With `--base`, the result is
`base + factor * (avg - base)` (task-vector extrapolation; factor 1 is the plain average). Weights are normalised to sum
to 1. The first model's non-weight files (config, tokenizer, chat template, processor, generation config, ...) are
copied, except its own provenance (`merge_info.json`, `soup_report.json`). Inputs are streamed tensor by tensor from
open shard handles, and the output keeps the first model's shard layout, so peak memory is one output shard plus a
tensor. Stale `*.safetensors` in the output directory are removed first (vLLM loads every safetensors file it finds).
A `soup_report.json` lists the inputs, weights and per-tensor statistics; it is written last, so it marks completion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPORT_TENSORS = 5  # max-abs-diff is reported for this many tensors (the most different ones)
INDEX = "model.safetensors.index.json"
PROVENANCE_FILES = ("merge_info.json", "soup_report.json")  # describe the input, not the soup; not copied


def weight_map(model_dir: Path) -> dict[str, str]:
    """tensor name -> shard file name."""
    from safetensors import safe_open

    index = model_dir / INDEX
    if index.exists():
        return dict(json.loads(index.read_text())["weight_map"])
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {model_dir}")
    out: dict[str, str] = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            out.update({k: f.name for k in h.keys()})  # noqa: SIM118 - a safetensors handle, not a dict
    return out


class Checkpoint:
    """Lazy tensor access over one checkpoint's shards."""

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.map = weight_map(self.dir)
        self._handles: dict[str, Any] = {}

    def _handle(self, shard: str):
        from safetensors import safe_open

        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), "pt")
        return self._handles[shard]

    def get(self, name: str):
        return self._handle(self.map[name]).get_tensor(name)

    def meta(self, name: str) -> tuple[tuple[int, ...], str]:
        sl = self._handle(self.map[name]).get_slice(name)
        return tuple(sl.get_shape()), sl.get_dtype()


def check_compatible(cks: Sequence[Checkpoint], labels: Sequence[str]) -> None:
    keys = set(cks[0].map)
    for ck, label in zip(cks[1:], labels[1:], strict=True):
        other = set(ck.map)
        if other != keys:
            raise SystemExit(
                f"tensor names differ: {label} lacks {sorted(keys - other)[:5]}, has extra {sorted(other - keys)[:5]}"
            )
    for k in sorted(keys):
        shape0, dtype0 = cks[0].meta(k)
        for ck, label in zip(cks[1:], labels[1:], strict=True):
            shape, dtype = ck.meta(k)
            if shape != shape0:
                raise SystemExit(f"shape of {k} differs: {shape0} vs {shape} in {label}")
            if dtype != dtype0:
                raise SystemExit(f"dtype of {k} differs: {dtype0} vs {dtype} in {label}")


def soup(
    models: Sequence[str],
    out: str,
    weights: Sequence[float] | None = None,
    base: str | None = None,
    factor: float = 1.0,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    if len(models) < 2:
        raise SystemExit("need at least two models")
    if weights is not None and len(weights) != len(models):
        raise SystemExit(f"{len(weights)} weights for {len(models)} models")
    if base is None and factor != 1.0:
        raise SystemExit("--factor needs --base")
    w = list(weights) if weights is not None else [1.0] * len(models)
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise SystemExit("weights must be non-negative with a positive sum")
    w = [x / sum(w) for x in w]
    cks = [Checkpoint(m) for m in models]
    check_compatible(cks, list(models))
    base_ck = Checkpoint(base) if base else None
    if base_ck is not None:
        check_compatible([cks[0], base_ck], [models[0], base])

    out_dir = Path(out)
    if out_dir.resolve() in {Path(m).resolve() for m in (*models, *([base] if base else []))}:
        raise SystemExit(f"--out {out} is one of the inputs; the output overwrites shards while they are read")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.safetensors"):
        stale.unlink()  # a shard of an earlier layout would be loaded next to the new ones
    (out_dir / INDEX).unlink(missing_ok=True)
    first = cks[0]
    shards: dict[str, list[str]] = {}
    for name, shard in first.map.items():
        shards.setdefault(shard, []).append(name)

    identical: list[str] = []
    diffs: dict[str, float] = {}
    new_map: dict[str, str] = {}
    total_bytes = 0
    for shard, names in sorted(shards.items()):
        tensors = {}
        for name in sorted(names):
            ts = [ck.get(name) for ck in cks]
            dtype = ts[0].dtype
            same = all(torch.equal(ts[0], t) for t in ts[1:])
            if same and factor == 1.0:
                identical.append(name)
                tensors[name] = ts[0]  # exact: no float32 round trip
            else:
                if not ts[0].is_floating_point():
                    raise SystemExit(f"{name} is {dtype} and differs across inputs: only float tensors average")
                if same:
                    identical.append(name)
                    avg = ts[0].float()
                else:
                    avg = sum(wi * t.float() for wi, t in zip(w, ts, strict=True))
                    diffs[name] = max(float((t.float() - ts[0].float()).abs().max()) for t in ts[1:])
                if base_ck is not None:
                    b = base_ck.get(name).float()
                    avg = b + factor * (avg - b)
                tensors[name] = avg.to(dtype)
            new_map[name] = shard
            total_bytes += tensors[name].numel() * tensors[name].element_size()
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(out_dir / shard), metadata={"format": "pt"})
        del tensors
    if (first.dir / INDEX).exists():
        index = {"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(new_map.items()))}
        (out_dir / INDEX).write_text(json.dumps(index, indent=2))
    for f in first.dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in (INDEX, *PROVENANCE_FILES):
            shutil.copy(f, out_dir / f.name)
    top = sorted(diffs, key=lambda k: -diffs[k])[:REPORT_TENSORS]
    report = {
        "models": list(models), "weights": w, "base": base, "factor": factor,
        "tensors_total": len(new_map),
        "tensors_identical_across_inputs": len(identical), "tensors_differing": len(diffs),
        "identical_examples": sorted(identical)[:REPORT_TENSORS],
        "max_abs_diff_between_inputs": {k: diffs[k] for k in top},
    }  # fmt: skip
    (out_dir / "soup_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m honest_agents.sft.soup", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--base", default=None, help="base checkpoint for task-vector extrapolation")
    p.add_argument("--factor", type=float, default=1.0, help="out = base + factor * (avg - base)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    print(json.dumps(soup(args.models, args.out, args.weights, args.base, args.factor), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
