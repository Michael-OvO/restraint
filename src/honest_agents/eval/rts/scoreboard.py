"""The RTS scoreboard (spec section 7): one row per model, every row scored on one frozen manifest and protocol.

  python -m honest_agents.eval.rts.scoreboard add --run runs/eval/rts_base --manifest data/rts/v1/manifest.jsonl \\
      --label base --order 0 [--adapter PATH]
  python -m honest_agents.eval.rts.scoreboard show
  python -m honest_agents.eval.rts.scoreboard delta --base base --trained it15-g2-s1 [--out delta.json]
  python -m honest_agents.eval.rts.scoreboard plot --out runs/rts/scoreboard.html [--wandb]

Each row keeps its units (`--units-dir`, default `runs/rts/units/`), so `delta` pairs two models without re-reading
their episodes. Episodes are scored with the default reward config, as `run_eval` does.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from honest_agents.eval.metrics import score_records
from honest_agents.eval.records import episodes_path, read_records
from honest_agents.eval.rts.manifest import load_manifest, sha256_of
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.eval.rts.score import DEFAULT_PROTOCOL, endpoint, render_endpoint, score_model
from honest_agents.eval.rts.units import build_units, coverage

DEFAULT_BOARD = Path("runs/rts/scoreboard.json")
DEFAULT_UNITS = Path("runs/rts/units")
MIN_COVERAGE = 0.9


def load_board(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"manifest_sha256": None, "protocol": None, "rows": []}
    return json.loads(path.read_text())


def save_board(board: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2) + "\n")


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "model"


def add_row(
    board: dict[str, Any],
    *,
    label: str,
    order: int,
    run_dir: str | Path,
    manifest: str | Path,
    units_dir: str | Path,
    adapter: str | None = None,
    proto: RtsProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Score one run against the manifest and put its row on the board (a row with the same label is replaced)."""
    digest = sha256_of(manifest)
    if board["manifest_sha256"] not in (None, digest):
        raise ValueError(f"board was scored on manifest {board['manifest_sha256']}, not {digest}; start a new board")
    if board["protocol"] not in (None, proto.to_json()):
        raise ValueError(f"board protocol {board['protocol']} differs from {proto.to_json()}; start a new board")
    ep = episodes_path(run_dir)
    if ep is None:
        raise ValueError(f"{run_dir}: no episodes.jsonl(.gz)")
    items = load_manifest(manifest)
    eps = score_records(read_records(ep))
    units = build_units(eps, items)
    units_path = Path(units_dir) / f"{_slug(label)}.json"
    units_path.parent.mkdir(parents=True, exist_ok=True)
    units_path.write_text(json.dumps(units))
    row = {
        "label": label,
        "order": order,
        "run": str(run_dir),
        "adapter": adapter,
        "units": str(units_path),
        "coverage": coverage(eps, items),
        "scores": score_model(units, proto),
    }
    board["manifest_sha256"], board["protocol"] = digest, proto.to_json()
    board["rows"] = sorted(
        [r for r in board["rows"] if r["label"] != label] + [row], key=lambda r: (r["order"], r["label"])
    )
    return row


def _pct(d: dict[str, Any] | None, interval: bool = False) -> str:
    if not d or d.get("value") is None:
        return "-"
    s = f"{100 * d['value']:.1f}"
    if interval and d.get("lo") is not None:
        s += f" [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return s


def render_table(board: dict[str, Any]) -> str:
    lines = [
        f"# RTS scoreboard (manifest {str(board['manifest_sha256'])[:12]})",
        "",
        "| order | model | RTS [95%] | near | mid | far | RTS-C [95%] | false refusal | tamper (twins) | coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in board["rows"]:
        h, t = r["scores"]["headline"], r["scores"]["tiers"]
        cov = f"{r['coverage']:.2f}" + (" (LOW)" if r["coverage"] < MIN_COVERAGE else "")
        lines.append(
            f"| {r['order']} | {r['label']} | {_pct(h['rts'], True)} | {_pct(t['near']['rts'])} | {_pct(t['mid']['rts'])} "
            f"| {_pct(t['far']['rts'])} | {_pct(h['rts_c'], True)} | {_pct(h['false_refusal'])} "
            f"| {_pct(h['tamper_twins'])} | {cov} |"
        )
    return "\n".join(lines) + "\n"


def _row(board: dict[str, Any], label: str) -> dict[str, Any]:
    for r in board["rows"]:
        if r["label"] == label:
            return r
    raise SystemExit(f"no row {label!r} on the board (have {[r['label'] for r in board['rows']]})")


def _log_wandb(board: dict[str, Any], page: str) -> None:
    import wandb  # train extra; only behind --wandb

    cols = ["order", "model", "rts", "rts_lo", "rts_hi", "near", "mid", "far", "rts_c", "rts_c_lo", "rts_c_hi"]
    data = []
    for r in board["rows"]:
        h, t = r["scores"]["headline"], r["scores"]["tiers"]
        data.append(
            [
                r["order"],
                r["label"],
                h["rts"]["value"],
                h["rts"]["lo"],
                h["rts"]["hi"],
                t["near"]["rts"]["value"],
                t["mid"]["rts"]["value"],
                t["far"]["rts"]["value"],
                h["rts_c"]["value"],
                h["rts_c"]["lo"],
                h["rts_c"]["hi"],
            ]
        )
    run = wandb.init(project="honest-agents", name="rts-scoreboard", job_type="eval")
    run.log({"rts/scoreboard": wandb.Table(columns=cols, data=data), "rts/plots": wandb.Html(page)})
    run.finish()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RTS scoreboard: score runs, compare two models, plot the board.")
    sub = p.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="score a run against the manifest and add its row")
    add.add_argument("--run", required=True)
    add.add_argument("--manifest", required=True)
    add.add_argument("--label", required=True)
    add.add_argument("--order", type=int, required=True, help="x position: base 0, then iterations in order")
    add.add_argument("--adapter")
    add.add_argument("--units-dir", default=str(DEFAULT_UNITS))
    add.add_argument("--n-boot", type=int, default=DEFAULT_PROTOCOL.n_boot)
    show = sub.add_parser("show", help="print the board as a markdown table")
    delta = sub.add_parser("delta", help="the pre-registered endpoint: trained minus base")
    delta.add_argument("--base", required=True)
    delta.add_argument("--trained", required=True)
    delta.add_argument("--out")
    plot = sub.add_parser("plot", help="write the HTML page (iteration curves, frontier, table)")
    plot.add_argument("--out", default="runs/rts/scoreboard.html")
    plot.add_argument("--wandb", action="store_true", help="also log the table and page to W&B (honest-agents)")
    for sp in (add, show, delta, plot):
        sp.add_argument("--board", default=str(DEFAULT_BOARD))
    args = p.parse_args(argv)
    board = load_board(args.board)
    if args.cmd == "add":
        try:
            row = add_row(
                board,
                label=args.label,
                order=args.order,
                run_dir=args.run,
                manifest=args.manifest,
                units_dir=args.units_dir,
                adapter=args.adapter,
                proto=RtsProtocol(n_boot=args.n_boot),
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        save_board(board, args.board)
        if row["coverage"] < MIN_COVERAGE:
            print(f"WARNING: coverage {row['coverage']:.2f} < {MIN_COVERAGE}: check run errors before reading scores")
        print(render_table(board))
    elif args.cmd == "show":
        print(render_table(board))
    elif args.cmd == "delta":
        arms = [json.loads(Path(_row(board, label)["units"]).read_text()) for label in (args.base, args.trained)]
        rep = endpoint(arms[0], arms[1], RtsProtocol.from_json(board["protocol"]))
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2))
        print(render_endpoint(rep))
    else:
        from honest_agents.eval.rts.plots import render_html

        page = render_html(board)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(page)
        if args.wandb:
            _log_wandb(board, page)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
