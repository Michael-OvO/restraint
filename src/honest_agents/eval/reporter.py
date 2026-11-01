"""The reporter (docs/iteration-19-design-reporter.md): a status-only adapter that decides whether G2's claim stands.

    python -m honest_agents.eval.reporter threshold --config C --adapter A --out t.json        (GPU)
    python -m honest_agents.eval.reporter score --config C --adapter A --transcripts DIR --out scores.jsonl  (GPU)
    python -m honest_agents.eval.reporter withdraw-episodes --episodes E --scores S --threshold-file T --out E2  (CPU)
    python -m honest_agents.eval.reporter withdraw-harbor --job DIR --scores S --threshold-file T --bench B --out U  (CPU)

G2 acts and writes its report; when the report claims `complete`, the reporter scores π(claim) on the transcript cut at
the report's status (`status_cut`, as the K2a items are cut) and the claim stands only if π(claim) ≥ t. A withdrawn
claim becomes `incomplete` with no blocker and no reason; a non-claim is never changed. Applied to archived episodes it
is exactly what a live reporter would have done, because the status comes after the last action.

`t` is chosen on the training items only (`choose_threshold`): the cut that maximises the CR-GRPO table's payoff of
the decision, Σ ΔL over the claims kept (ΔL = claim minus the canonical non-claim, `train.confidence_exact
.status_deltas`), so it is fixed before any evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPORT_TOOL = "submit_report"
WITHDRAWN_STATUS = "incomplete"


def _args(call: dict[str, Any]) -> dict[str, Any]:
    a = (call.get("function") or {}).get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return {}
    return a if isinstance(a, dict) else {}


def cb_sentinel() -> str:
    from honest_agents.sft import confidence_bandit as cb

    return cb.SENTINEL


def claim_index(messages: Sequence[dict[str, Any]]) -> int | None:
    """The index of the last assistant message that calls `submit_report` with status `complete`, else None."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if (c.get("function") or {}).get("name") == REPORT_TOOL and _args(c).get("status") == "complete":
                return i
    return None


def status_cut(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The transcript up to the claim, the claiming message reduced to `submit_report(status=<sentinel>)`: the prompt
    the reporter scores, as `confidence_bandit.status_messages` cuts a training item. None when nothing is claimed."""
    from honest_agents.sft import confidence_bandit as cb

    i = claim_index(messages)
    if i is None:
        return None
    call = next(c for c in messages[i]["tool_calls"] if (c.get("function") or {}).get("name") == REPORT_TOOL)
    cut = {**dict(call), "function": {**dict(call["function"]), "arguments": {"status": cb.SENTINEL}}}
    from honest_agents.pilot.dataset import normalize_message

    head = [normalize_message(dict(m)) for m in messages[:i]]  # as the training items were normalised
    return [*head, {**normalize_message(dict(messages[i])), "tool_calls": [cut]}]


def choose_threshold(pis: Sequence[float], deltas: Sequence[float]) -> dict[str, Any]:
    """The cut t maximising Σ ΔL over the claims kept (π ≥ t); t = 0 keeps every claim. Ties go to the lowest t, the
    one that withdraws least."""
    pairs = sorted(zip(pis, deltas, strict=True))
    best_t, best_u = 0.0, sum(deltas)
    u = best_u
    for k, (pi, d) in enumerate(pairs):  # raising t past pairs[k] withdraws it
        u -= d
        t = pairs[k + 1][0] if k + 1 < len(pairs) else 1.0 + 1e-9
        if u > best_u + 1e-12:
            best_t, best_u = t, u
    kept = sum(1 for p in pis if p >= best_t)
    return {"threshold": best_t, "utility": best_u, "utility_keep_all": sum(deltas), "claims": len(pis),
            "kept": kept}  # fmt: skip


def withdraw(report: dict[str, Any] | None, pi: float | None, t: float) -> tuple[dict[str, Any] | None, bool]:
    """The report as the reporter leaves it, and whether a claim was withdrawn."""
    if report is None or report.get("status") != "complete" or pi is None or pi >= t:
        return report, False
    return {**report, "status": WITHDRAWN_STATUS, "blocker": None, "reason": "", "completed": []}, True


def episode_key(episode_id: str) -> str:
    """The id a transcript file carries (`run_eval` writes `<episode id with / as _>.json`)."""
    return episode_id.replace("/", "_")


def load_scores(path: Path) -> dict[str, float]:
    return {r["id"]: r["pi"] for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}


def threshold_of(path: Path) -> float:
    return float(json.loads(path.read_text())["threshold"])


def withdraw_episodes(rows: Iterable[dict[str, Any]], scores: dict[str, float], t: float) -> tuple[list[dict], int]:
    """`run_eval` episode rows with each claim the reporter rejects withdrawn (the outcome's report and the stored
    reward fields that read it are recomputed by the analyses from the outcome, so only the report changes)."""
    out, n = [], 0
    for r in rows:
        o = dict(r["outcome"])
        rep, w = withdraw(o.get("report"), scores.get(episode_key(r["episode_id"])), t)
        if w:
            o["report"] = rep
            n += 1
        out.append({**r, "outcome": o})
    return out, n


def withdraw_harbor(job_dir: Path, scores: dict[str, float], t: float, bench: str) -> tuple[list[dict], int]:
    """The units of a Harbor job (TB2 or ImpossibleBench LCB) with each rejected claim withdrawn, built by the same
    units writers as the benchmark's own (`final_benchmark.tb2_units`, `impossiblebench.unit`)."""
    from honest_agents.eval.harbor_results import load_harbor_job
    from honest_agents.report import Status

    records, n = load_harbor_job(job_dir), 0
    for rec in records:
        trial = Path(str(rec.extra.get("trial_dir") or "")).name
        rep = rec.outcome.report
        pi = scores.get(trial)
        if rep is not None and rep.status == Status.COMPLETE and pi is not None and pi < t:
            rec.outcome.report = rep.model_copy(
                update={"status": Status.INCOMPLETE, "blocker": None, "reason": "", "completed": []}
            )
            n += 1
    if bench == "terminal_bench_2":
        from honest_agents.eval.final_benchmark import tb2_units

        return tb2_units(records), n
    if bench == "impossiblebench_lcb":
        from honest_agents.eval.external import impossiblebench as ib

        units = [ib.unit(ib.row_from_record(r)) for r in records]
        return [u for u in units if u is not None], n
    raise ValueError(f"withdraw-harbor knows terminal_bench_2 and impossiblebench_lcb, not {bench!r}")


def _episode_files(path: Path) -> Path:
    return next(p for p in (path / "episodes.jsonl.gz", path / "episodes.jsonl", path) if p.is_file())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def transcripts_of(source: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """(id, messages) of every transcript: a `run_eval` dir's `transcripts/*.json` (id = the episode id) or a Harbor job
    dir's trials (`<trial>/agent/honest_transcript.json`, id = the trial dir name)."""
    if (source / "transcripts").is_dir():
        for f in sorted((source / "transcripts").glob("*.json")):
            yield f.stem, json.loads(f.read_text())
        return
    for f in sorted(glob.glob(str(source / "*" / "agent" / "honest_transcript.json"))):
        yield Path(f).parents[1].name, json.loads(Path(f).read_text())


def score(cfg: Any, adapter: str, sources: Sequence[Path]) -> list[dict[str, Any]]:
    """π(claim) of the reporter and of G2 (adapter off) on every claiming transcript of `sources`."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec, pinned_revision
    from honest_agents.sft import confidence_bandit as cb
    from honest_agents.train import confidence_exact as ce

    name = cfg.model.tokenizer or cfg.model.name
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision(name))
    t = ce.StatusTokens.of(tok)
    kwargs, tools = dict(family_spec(cfg.model.family).chat_template_kwargs), tool_schemas()
    model = PeftModel.from_pretrained(ce.load_model(cfg.model.name, cfg.model.dtype), adapter)
    model.eval()
    out = []
    with torch.no_grad():
        for src in sources:
            for tid, messages in transcripts_of(src):
                cut = status_cut(messages)
                if cut is None:
                    continue
                ids = tok(cb.render_prefix(cut, tok, tools, kwargs), add_special_tokens=False).input_ids
                node, _ = ce._status_forward(model, ids, len(ids), ())
                with model.disable_adapter():
                    ref, _ = ce._status_forward(model, ids, len(ids), ())
                out.append({"id": tid, "source": str(src), "pi": float(torch.sigmoid(node[t.complete] - node[t.incomplete])),
                            "pi_g2": float(torch.sigmoid(ref[t.complete] - ref[t.incomplete]))})  # fmt: skip
    return out


def threshold_on_training_items(cfg: Any, adapter: str) -> dict[str, Any]:
    """`choose_threshold` on the training side's claims (feasible claims and claims on infeasible variants)."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from honest_agents.models import pinned_revision
    from honest_agents.train import confidence as tc
    from honest_agents.train import confidence_exact as ce

    name = cfg.model.tokenizer or cfg.model.name
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision(name))
    t = ce.StatusTokens.of(tok)
    prepared = tc.prepare(cfg, tok, target="status")
    deltas = ce.status_deltas(prepared.train, v_rows=prepared.train)
    claims = [r for r in prepared.train if r["cell"] in ("feasible/complete", "infeasible/complete")]
    model = PeftModel.from_pretrained(ce.load_model(cfg.model.name, cfg.model.dtype), adapter)
    model.eval()
    pis = []
    with torch.no_grad():
        for r in claims:
            ids = tok(r["prompt"], add_special_tokens=False).input_ids
            node, _ = ce._status_forward(model, ids, len(ids), ())
            pis.append(float(torch.sigmoid(node[t.complete] - node[t.incomplete])))
    return choose_threshold(pis, [deltas[r["item_id"]] for r in claims])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.eval.reporter", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    th = sub.add_parser("threshold", help="choose t on the training items (GPU)")
    sc = sub.add_parser("score", help="π(claim) on the claiming transcripts of run_eval / Harbor dirs (GPU)")
    for p in (th, sc):
        p.add_argument("--config", required=True)
        p.add_argument("--adapter", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("overrides", nargs="*")
    sc.add_argument("--transcripts", type=Path, action="append", required=True)
    we = sub.add_parser("withdraw-episodes", help="a run_eval episodes file with the rejected claims withdrawn (CPU)")
    we.add_argument("--episodes", type=Path, required=True)
    wh = sub.add_parser("withdraw-harbor", help="a Harbor job's units with the rejected claims withdrawn (CPU)")
    wh.add_argument("--job", type=Path, required=True)
    wh.add_argument("--bench", required=True, choices=("terminal_bench_2", "impossiblebench_lcb"))
    for p in (we, wh):
        p.add_argument("--scores", type=Path, required=True)
        p.add_argument("--threshold-file", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("threshold", "score"):
        from honest_agents.train import confidence as tc

        cfg = tc.load_config(a.config, a.overrides)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        if a.cmd == "threshold":
            res = threshold_on_training_items(cfg, a.adapter)
            a.out.write_text(json.dumps(res, indent=1) + "\n")
            print(json.dumps(res))
        else:
            rows = score(cfg, a.adapter, a.transcripts)
            a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(json.dumps({"scored": len(rows), "out": str(a.out)}))
        return 0
    if a.cmd == "withdraw-harbor":
        units, n = withdraw_harbor(a.job, load_scores(a.scores), threshold_of(a.threshold_file), a.bench)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"units": units}) + "\n")
        print(json.dumps({"units": len(units), "withdrawn": n, "out": str(a.out)}))
        return 0
    rows, n = withdraw_episodes(
        _read_jsonl(_episode_files(a.episodes)), load_scores(a.scores), threshold_of(a.threshold_file)
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"episodes": len(rows), "withdrawn": n, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
