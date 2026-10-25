"""The RTS HTML page: three panels, a table, tooltips, missing values and escaping."""

from __future__ import annotations

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
    assert "far" in page  # far is None for two rows and still drawn for G2


def test_single_row_and_empty_board():
    assert render_html({**BOARD, "rows": BOARD["rows"][:1]}).count("<svg") == 3
    assert "no rows yet" in render_html({**BOARD, "rows": []})
