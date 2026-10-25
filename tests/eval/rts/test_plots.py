"""The RTS HTML page: three panels, a table, tooltips, missing values and escaping."""

from __future__ import annotations

import re

from honest_agents.eval.rts.plots import render_html


def _s(v, lo=None, hi=None):
    return {"value": v, "lo": lo, "hi": hi}


def _row(order, label, rts, cap, near=0.3, mid=0.2, far=None):
    return {
        "label": label,
        "order": order,
        "coverage": 1.0,
        "scores": {
            "headline": {
                "rts": _s(rts, rts - 0.1, rts + 0.1),
                "rts_c": _s(cap, cap - 0.05, cap + 0.05),
                "false_refusal": _s(0.05),
                "tamper_twins": _s(0.0),
            },
            "tiers": {"near": {"rts": _s(near)}, "mid": {"rts": _s(mid)}, "far": {"rts": _s(far)}},
        },
    }


BOARD = {
    "manifest_sha256": "ab" * 32,
    "protocol": {},
    "rows": [_row(0, "base", 0.05, 0.30), _row(1, "SFT-v2 <e3>", 0.35, 0.32), _row(2, "G2", 0.45, 0.31, far=0.1)],
}


def test_page_has_panels_table_and_tooltips():
    page = render_html(BOARD)
    assert page.startswith("<!doctype html>")
    assert page.count("<svg") == 3
    assert "<table" in page and page.count("<tr>") == 1 + 3
    assert "<title>G2 · RTS: 45.0 [35.0, 55.0]</title>" in page
    assert "any blind policy" in page
    assert "prefers-color-scheme: dark" in page


def test_labels_are_escaped_and_missing_values_render():
    page = render_html(BOARD)
    assert "SFT-v2 &lt;e3&gt;" in page and "SFT-v2 <e3>" not in page
    assert page.count(">far</text>") == 1  # far is None for two rows, present for G2
    no_far = {**BOARD, "rows": BOARD["rows"][:2]}
    assert ">far</text>" not in render_html(no_far)


def test_single_row_and_empty_board():
    assert render_html({**BOARD, "rows": BOARD["rows"][:1]}).count("<svg") == 3
    assert "no rows yet" in render_html({**BOARD, "rows": []})


def test_zero_label_on_all_positive_board():
    rows = [_row(0, "a", 0.5, 0.3), _row(1, "b", 0.6, 0.3)]
    assert "any blind policy" in render_html({**BOARD, "rows": rows})


def test_partial_intervals_render():
    for part in ({"value": 0.2, "lo": 0.1, "hi": None}, {"value": 0.2, "lo": None, "hi": 0.3}):
        rows = [_row(0, "a", 0.5, 0.3), _row(1, "b", 0.6, 0.3), _row(2, "c", 0.6, 0.3)]
        rows[1]["scores"]["headline"]["rts"] = part
        rows[1]["scores"]["headline"]["rts_c"] = part
        page = render_html({**BOARD, "rows": rows})
        assert "<title>b · RTS: 20.0</title>" in page


def test_charts_scroll_inside_panel():
    page = render_html(BOARD)
    assert page.count('<div class="chart"><svg') == 3
    assert ".chart svg { min-width: 560px; }" in page


def test_end_labels_do_not_collide():
    rows = [_row(0, "a", 0.5, 0.3, near=0.30, mid=0.31), _row(1, "b", 0.5, 0.3, near=0.30, mid=0.31)]
    page = render_html({**BOARD, "rows": rows})
    ys = sorted(float(y) for y in re.findall(r'<text class="lab" x="[\d.]+" y="([\d.]+)">(?:near|mid)</text>', page))
    assert len(ys) == 2 and ys[1] - ys[0] >= 14
    assert 'stroke="var(--muted)" stroke-width="1"/>' in page  # leader line for the moved label


def test_frontier_arrows_between_consecutive_models():
    assert render_html(BOARD).count('marker-end="url(#arr)"') == 2


def test_theme_tokens_on_root():
    page = render_html(BOARD)
    assert ":root { color-scheme: light;" in page and ".viz-root { color-scheme" not in page
