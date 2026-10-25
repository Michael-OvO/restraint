"""The RTS scoreboard as one self-contained HTML page (spec section 7).

1. Restraint over iterations: RTS with its interval band, and the near / mid / far tiers.
2. Capability over iterations: RTS-C with its interval band (its own panel: one y-axis per chart).
3. The frontier: RTS against RTS-C, one point per model, arrows in iteration order.

Inline SVG and CSS only. Hover tooltips are SVG `<title>`s; colours are CSS tokens with a selected dark mode. The
palette was validated with the dataviz validator for 4 slots in both modes; light slots 3 and 4 are below 3:1 on the
surface, so every line has a direct end label and the page ends with a table of every number.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from itertools import pairwise
from typing import Any

W, H = 680, 300
ML, MR, MT, MB = 48, 104, 16, 44

CSS = """
.viz-root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink-1: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d;
    --ink-1: #f0efec; --ink-2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; }
}
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d; --ink-1: #f0efec;
  --ink-2: #c3c2b7; --grid: #2c2c2a; --axis: #383835; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; }
body { margin: 0; background: var(--page); }
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink-1); background: var(--page);
  padding: 24px 16px; max-width: 760px; margin: 0 auto; }
.panel { background: var(--surface); border-radius: 8px; padding: 12px; margin: 0 0 20px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 4px; }
p.note { color: var(--ink-2); font-size: 13px; margin: 0 0 8px; }
svg { width: 100%; height: auto; display: block; }
svg text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg text.lab { fill: var(--ink-2); font-size: 12px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:nth-child(2), td:nth-child(2) { text-align: left; }
"""

Score = dict[str, Any] | None
Getter = Callable[[dict[str, Any]], Score]


def _hl(name: str) -> Getter:
    return lambda r: r["scores"]["headline"].get(name)


def _tier(t: str) -> Getter:
    return lambda r: r["scores"]["tiers"][t].get("rts")


def _v(s: Score, key: str = "value") -> float | None:
    return None if not s or s.get(key) is None else 100.0 * s[key]


def _fmt(s: Score) -> str:
    v = _v(s)
    if v is None:
        return "-"
    lo, hi = _v(s, "lo"), _v(s, "hi")
    return f"{v:.1f}" + ("" if lo is None else f" [{lo:.1f}, {hi:.1f}]")


class _Ax:
    def __init__(self, x0: float, x1: float, y0: float, y1: float) -> None:
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def x(self, v: float) -> float:
        return ML + (v - self.x0) / ((self.x1 - self.x0) or 1) * (W - ML - MR)

    def y(self, v: float) -> float:
        return H - MB - (v - self.y0) / ((self.y1 - self.y0) or 1) * (H - MT - MB)


def _grid(ax: _Ax, step: int) -> list[str]:
    out = []
    v = ax.y0
    while v <= ax.y1 + 1e-9:
        y = ax.y(v)
        stroke = "var(--axis)" if abs(v) < 1e-9 else "var(--grid)"
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.0f}</text>')
        v += step
    return out


def _segments(pts: list[tuple[float, float | None]]) -> list[list[tuple[float, float]]]:
    segs: list[list[tuple[float, float]]] = [[]]
    for x, y in pts:
        if y is None:
            segs.append([])
        else:
            segs[-1].append((x, y))
    return [s for s in segs if s]


def _y_floor(rows: list[dict[str, Any]], getters: list[Getter]) -> float:
    lows = [_v(g(r), "lo") if _v(g(r), "lo") is not None else _v(g(r)) for r in rows for g in getters]
    lows = [v for v in lows if v is not None]
    return min(0.0, 20.0 * (min(lows) // 20.0)) if lows else 0.0


def _line_panel(
    rows: list[dict[str, Any]],
    series: list[tuple[Getter, str, str, bool]],
    y0: float,
    y1: float,
    aria: str,
    zero_note: bool,
) -> str:
    ax = _Ax(0, max(len(rows) - 1, 1), y0, y1)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria)}">', *_grid(ax, 20)]
    if zero_note and y0 < 0 <= y1:
        p.append(f'<text x="{W - MR - 4}" y="{ax.y(0) - 6:.1f}" text-anchor="end">0 = any blind policy</text>')
    for k, r in enumerate(rows):
        p.append(f'<text x="{ax.x(k):.1f}" y="{H - MB + 18}" text-anchor="middle">{html.escape(r["label"])}</text>')
    for get, name, color, band in series:
        vals = [(ax.x(k), _v(get(r))) for k, r in enumerate(rows)]
        if band:
            for seg in _segments([(ax.x(k), _v(get(r), "lo")) for k, r in enumerate(rows)]):
                xs = {x for x, _ in seg}
                his = [(ax.x(k), _v(get(r), "hi")) for k, r in enumerate(rows) if ax.x(k) in xs]
                pts = [f"{x:.1f},{ax.y(v):.1f}" for x, v in seg] + [f"{x:.1f},{ax.y(v):.1f}" for x, v in reversed(his)]
                p.append(f'<polygon points="{" ".join(pts)}" fill="var({color})" opacity="0.15"/>')
        for seg in _segments(vals):
            d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{ax.y(v):.1f}" for i, (x, v) in enumerate(seg))
            p.append(f'<path d="{d}" fill="none" stroke="var({color})" stroke-width="2"/>')
        for k, r in enumerate(rows):
            v = _v(get(r))
            if v is None:
                continue
            tip = html.escape(f"{r['label']} · {name}: {_fmt(get(r))}")
            p.append(
                f'<circle cx="{ax.x(k):.1f}" cy="{ax.y(v):.1f}" r="4" fill="var({color})" stroke="var(--surface)" '
                f'stroke-width="2"><title>{tip}</title></circle>'
            )
        last = [(x, v) for x, v in vals if v is not None]
        if last:
            _, v = last[-1]
            y = ax.y(v)
            p.append(
                f'<line x1="{W - MR + 8}" x2="{W - MR + 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="var({color})" '
                f'stroke-width="2"/><text class="lab" x="{W - MR + 24}" y="{y + 4:.1f}">{html.escape(name)}</text>'
            )
    p.append("</svg>")
    return "".join(p)


def _frontier(rows: list[dict[str, Any]]) -> str:
    pts = [(r, _v(r["scores"]["headline"]["rts_c"]), _v(r["scores"]["headline"]["rts"])) for r in rows]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    y0 = _y_floor(rows, [_hl("rts")])
    ax = _Ax(0, 100, y0, 100)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="RTS against RTS-C, one point per model">', *_grid(ax, 20)]
    p.append(
        '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
        'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--muted)"/></marker></defs>'
    )
    for v in range(0, 101, 20):
        p.append(f'<text x="{ax.x(v):.1f}" y="{H - MB + 18}" text-anchor="middle">{v}</text>')
    p.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - 6}" text-anchor="middle">RTS-C (capability)</text>')
    if y0 < 0:
        p.append(f'<text x="{W - MR - 4}" y="{ax.y(0) - 6:.1f}" text-anchor="end">0 = any blind policy</text>')
    for (_, xa, ya), (_, xb, yb) in pairwise(pts):
        p.append(
            f'<line x1="{ax.x(xa):.1f}" y1="{ax.y(ya):.1f}" x2="{ax.x(xb):.1f}" y2="{ax.y(yb):.1f}" '
            f'stroke="var(--muted)" stroke-width="1.5" marker-end="url(#arr)"/>'
        )
    for r, x, y in pts:
        tip = html.escape(
            f"{r['label']} · RTS {_fmt(r['scores']['headline']['rts'])} · RTS-C "
            f"{_fmt(r['scores']['headline']['rts_c'])}"
        )
        p.append(
            f'<circle cx="{ax.x(x):.1f}" cy="{ax.y(y):.1f}" r="5" fill="var(--s1)" stroke="var(--surface)" '
            f'stroke-width="2"><title>{tip}</title></circle>'
            f'<text class="lab" x="{ax.x(x) + 9:.1f}" y="{ax.y(y) - 7:.1f}">{html.escape(r["label"])}</text>'
        )
    p.append("</svg>")
    return "".join(p)


def _table(rows: list[dict[str, Any]]) -> str:
    head = [
        "order",
        "model",
        "RTS [95%]",
        "near",
        "mid",
        "far",
        "RTS-C [95%]",
        "false refusal",
        "tamper (twins)",
        "coverage",
    ]
    out = ['<div class="table-wrap"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for r in rows:
        h, t = r["scores"]["headline"], r["scores"]["tiers"]
        cells = [
            str(r["order"]),
            html.escape(r["label"]),
            _fmt(h["rts"]),
            _fmt(t["near"]["rts"]),
            _fmt(t["mid"]["rts"]),
            _fmt(t["far"]["rts"]),
            _fmt(h["rts_c"]),
            _fmt(h.get("false_refusal")),
            _fmt(h.get("tamper_twins")),
            f"{r['coverage']:.2f}",
        ]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def render_html(board: dict[str, Any]) -> str:
    rows = board["rows"]
    if not rows:
        body = '<p class="note">no rows yet: add one with <code>scoreboard add</code>.</p>'
    else:
        restraint = [
            (_hl("rts"), "RTS", "--s1", True),
            (_tier("near"), "near", "--s2", False),
            (_tier("mid"), "mid", "--s3", False),
            (_tier("far"), "far", "--s4", False),
        ]
        y0 = _y_floor(rows, [g for g, *_ in restraint])
        body = "".join(
            [
                (
                    '<div class="panel"><h2>Restraint (RTS, Youden J × 100)</h2><p class="note">P(decline | impossible twin) '
                    "minus P(false refusal | solvable control); band = 95% interval over source tasks.</p>"
                ),
                _line_panel(rows, restraint, y0, 100, "RTS and its tiers by model", True),
                "</div>",
                (
                    '<div class="panel"><h2>Capability (RTS-C, verified success on controls)</h2>'
                    '<p class="note">Equal weight per source benchmark; band = 95% interval.</p>'
                ),
                _line_panel(rows, [(_hl("rts_c"), "RTS-C", "--s1", True)], 0, 100, "RTS-C by model", False),
                "</div>",
                (
                    '<div class="panel"><h2>Frontier</h2><p class="note">Up and to the right is better; arrows follow '
                    "iteration order.</p>"
                ),
                _frontier(rows),
                "</div>",
                '<div class="panel"><h2>Every number</h2>',
                _table(rows),
                "</div>",
            ]
        )
    digest = html.escape(str(board.get("manifest_sha256"))[:12])
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>RTS Scoreboard</title>'
        f'<style>{CSS}</style></head><body><div class="viz-root"><h1>Restraint Transfer Suite</h1>'
        f'<p class="note">manifest {digest} · {len(rows)} models</p>{body}</div></body></html>\n'
    )
