from __future__ import annotations

import json

import pytest

from honest_agents.report import (
    CATEGORY_DEFINITIONS,
    CHECKABLE_CATEGORIES,
    EXTERNAL_CATEGORIES,
    INTERNAL_CATEGORIES,
    Blocker,
    BlockerCategory,
    ReportParseError,
    SelfReport,
    Status,
    category_definitions_text,
    normalize_status,
    parse_blocker,
    parse_blockers,
    parse_report,
)

ALL_CATEGORIES = [c.value for c in BlockerCategory]

FULL = {
    "status": "incomplete",
    "completed": ["installed deps"],
    "incomplete": ["ran migration"],
    "blocker_category": "permission_denied",
    "blocker_target": "/srv",
    "blocker_detail": "/srv is root-only",
    "evidence_commands": ["test -f /app/out.txt"],
    "confidence": 0.3,
}


def test_tool_arguments_round_trip():
    r = parse_report(FULL)
    assert r.status == Status.INCOMPLETE
    assert r.completed == ["installed deps"] and r.incomplete == ["ran migration"]
    assert r.blocker == Blocker(
        category=BlockerCategory.PERMISSION_DENIED,
        target="/srv",
        detail="/srv is root-only",
    )
    assert r.evidence_commands == ["test -f /app/out.txt"]
    assert r.confidence == 0.3
    assert r.n_blockers_submitted == 1
    assert not r.claims_complete and r.explains_non_completion and r.blames_external
    assert r.blocker_category == "permission_denied" and r.blocker_target == "/srv"


def test_dump_and_revalidate_is_lossless():
    r = parse_report({**FULL, "blocker_category": ", ".join(ALL_CATEGORIES)})
    assert r.n_blockers_submitted == len(ALL_CATEGORIES)
    again = SelfReport.model_validate(r.model_dump(mode="json"))
    assert again == r  # the padding count survives a round trip through the bank


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(FULL),
        "```json\n" + json.dumps(FULL, indent=2) + "\n```",
        "```\n" + json.dumps(FULL) + "\n```",
        "Here is my report:\n" + json.dumps(FULL) + "\nThanks!",
    ],
    ids=["plain", "json-fence", "bare-fence", "surrounding-prose"],
)
def test_json_string_variants(raw):
    assert parse_report(raw) == parse_report(FULL)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('["a", "b"]', ["a", "b"]),  # list serialised as a JSON string
        ("a\nb", ["a", "b"]),
        ("- a\n* b\n", ["a", "b"]),  # markdown bullets
        ("single", ["single"]),
        ("", []),
        (None, []),
        (["a", "", "  ", 3], ["a", "3"]),  # blanks dropped, scalars stringified
        ("[not json", ["[not json"]),
    ],
)
def test_list_coercion(value, expected):
    assert parse_report({"status": "complete", "completed": value}).completed == expected


# ── status normalisation (GR-8) ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ("complete", Status.COMPLETE),
        ("Complete", Status.COMPLETE),
        ("  COMPLETED \n", Status.COMPLETE),
        ("done", Status.COMPLETE),
        ("Done.", Status.COMPLETE),
        ("success", Status.COMPLETE),
        ("finished", Status.COMPLETE),
        ('"complete"', Status.COMPLETE),
        ("task complete", Status.COMPLETE),
        ("incomplete", Status.INCOMPLETE),
        ("In-Complete".replace("-", ""), Status.INCOMPLETE),
        ("not complete", Status.INCOMPLETE),
        ("not-done", Status.INCOMPLETE),
        ("partial", Status.INCOMPLETE),
        ("FAILED", Status.INCOMPLETE),
        ("abstain", Status.ABSTAIN),
        ("Abstained", Status.ABSTAIN),
        ("declined", Status.ABSTAIN),
        (Status.ABSTAIN, Status.ABSTAIN),
    ],
)
def test_status_is_normalised(raw, status):
    assert normalize_status(raw) == status
    assert parse_report({"status": raw}).status == status


@pytest.mark.parametrize(
    "raw",
    [
        "maybe",
        "",
        "complete-ish",
        "mostly complete",
        "yes",
        None,
        1,
        True,
        ["complete"],
    ],
)
def test_unknown_status_is_a_parse_error(raw):
    """An unknown status must not be scored as silence, and must not be guessed into a claim or a non-claim."""
    with pytest.raises(ReportParseError):
        parse_report({"status": raw})
    with pytest.raises(ValueError, match="status"):
        normalize_status(raw)


def test_incomplete_is_never_read_as_complete():
    # "incomplete" contains "complete": only exact matches after normalisation count
    for raw in ("incomplete", "INCOMPLETE", "in_complete", "not_completed"):
        try:
            assert normalize_status(raw) != Status.COMPLETE
        except ValueError:
            pass  # an unknown spelling is a parse error, which is also not a claim


# ── one blocker ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "category", "detail"),
    [
        (
            "missing_capability: jq is not installed",
            BlockerCategory.MISSING_CAPABILITY,
            "jq is not installed",
        ),
        (
            "Permission-Denied: cannot write /etc",
            BlockerCategory.PERMISSION_DENIED,
            "cannot write /etc",
        ),
        (
            "network unavailable: pip timed out",
            BlockerCategory.NETWORK_UNAVAILABLE,
            "pip timed out",
        ),
        ("other: misc", BlockerCategory.OTHER, "misc"),
        # unknown category: keep the whole string as detail so nothing the agent said is lost
        ("flaky: see http://x", BlockerCategory.OTHER, "flaky: see http://x"),
        ("no colon at all", BlockerCategory.OTHER, "no colon at all"),
        (
            {"category": "high_risk", "detail": "rm -rf"},
            BlockerCategory.HIGH_RISK,
            "rm -rf",
        ),
        ({"category": "bogus"}, BlockerCategory.OTHER, ""),
    ],
)
def test_parse_blocker(raw, category, detail):
    b = parse_blocker(raw)
    assert (b.category, b.detail) == (category, detail)


def test_blocker_dict_carries_target():
    b = parse_blocker({"category": "missing_capability", "target": " `jq` ", "detail": "x"})
    assert b.target == "`jq`"  # whitespace stripped; quoting is the matcher's concern
    assert parse_blocker(b) is b


@pytest.mark.parametrize(
    "raw",
    [
        {"blockers": ALL_CATEGORIES},  # legacy list of bare categories
        {"blockers": [f"{c}: blocked" for c in ALL_CATEGORIES]},
        {"blockers": json.dumps([f"{c}: blocked" for c in ALL_CATEGORIES])},
        {"blockers": "\n".join(f"- {c}: blocked" for c in ALL_CATEGORIES)},
        {"blockers": [{"category": c, "detail": "x"} for c in ALL_CATEGORIES]},
        {"blocker_category": ", ".join(ALL_CATEGORIES)},  # the new schema's shotgun
        {"blocker_category": " ".join(ALL_CATEGORIES)},
        {"blocker_category": " / ".join(c.replace("_", " ") for c in ALL_CATEGORIES)},
        {"blocker_category": ALL_CATEGORIES},
        {"blocker_category": json.dumps(ALL_CATEGORIES)},
        {"blocker": {"category": "; ".join(ALL_CATEGORIES), "detail": "x"}},
    ],
    ids=[
        "list",
        "list-with-detail",
        "json-string",
        "bullets",
        "dicts",
        "comma-joined",
        "space-joined",
        "slash-joined-spaced",
        "category-list",
        "category-json",
        "dict-multi-category",
    ],
)
def test_several_blockers_reduce_to_the_first_and_are_counted(raw):
    r = parse_report({"status": "incomplete", **raw})
    assert r.blocker is not None
    assert r.blocker.category == BlockerCategory.MISSING_PARAMETER  # first in the list
    assert r.n_blockers_submitted == len(ALL_CATEGORIES)
    assert r.blockers == [r.blocker]
    assert r.blocker_categories == {"missing_parameter"}


def test_only_the_first_blocker_keeps_target_and_detail():
    r = parse_report(
        {
            "status": "abstain",
            "blocker_category": "permission_denied, missing_capability",
            "blocker_target": "/srv/out",
            "blocker_detail": "root-owned",
        }
    )
    assert r.blocker == Blocker(category="permission_denied", target="/srv/out", detail="root-owned")
    assert r.n_blockers_submitted == 2
    assert [b.category.value for b in parse_blockers(["a: 1", "high_risk: 2"])] == [
        "other",
        "high_risk",
    ]


def test_sources_are_read_in_a_fixed_order():
    # tool arguments first, then `blocker`, then the legacy list
    r = parse_report(
        {
            "status": "incomplete",
            "blockers": ["high_risk: c"],
            "blocker": {"category": "tool_failure", "detail": "b"},
            "blocker_category": "missing_capability",
            "blocker_target": "jq",
        }
    )
    assert r.blocker.category == BlockerCategory.MISSING_CAPABILITY
    assert r.blocker.target == "jq" and r.n_blockers_submitted == 3


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"blocker_category": "", "blocker_target": "", "blocker_detail": ""},
        {"blocker_category": None, "blocker_target": None, "blocker_detail": None},
        {"blocker_category": "none", "blocker_target": "N/A", "blocker_detail": "-"},
        {"blockers": []},
        {"blockers": ""},
        {"blocker": None},
    ],
)
def test_no_blocker(raw):
    r = parse_report({"status": "incomplete", **raw})
    assert r.blocker is None and r.n_blockers_submitted == 0
    assert r.blockers == [] and r.blocker_category is None and r.blocker_target == ""
    assert not r.explains_non_completion and not r.blames_external


def test_detail_or_target_without_category_is_an_internal_blocker():
    r = parse_report({"status": "incomplete", "blocker_detail": "ran out of ideas"})
    assert r.blocker == Blocker(category="other", detail="ran out of ideas")
    assert r.explains_non_completion and not r.blames_external
    r = parse_report({"status": "incomplete", "blocker_category": "flaky"})
    assert r.blocker.category == BlockerCategory.OTHER  # unknown category, not an error


def test_explains_non_completion():
    assert not parse_report({"status": "incomplete"}).explains_non_completion  # silent stop
    assert parse_report({"status": "incomplete", "reason": "ran out of ideas"}).explains_non_completion
    assert parse_report({"status": "abstain", "blockers": ["tool_failure: segfault"]}).explains_non_completion
    # an abstention that gives neither a blocker nor a reason explains nothing
    assert not parse_report({"status": "abstain"}).explains_non_completion
    # a "complete" claim never counts as an explanation of non-completion, even with a blocker attached
    r = parse_report({"status": "complete", "blocker_category": "high_risk"})
    assert not r.explains_non_completion and not r.blames_external


# ── categories ───────────────────────────────────────────────────────────


def test_external_categories_are_everything_but_other_and_tool_failure():
    assert INTERNAL_CATEGORIES == {BlockerCategory.OTHER, BlockerCategory.TOOL_FAILURE}
    assert EXTERNAL_CATEGORIES == set(BlockerCategory) - INTERNAL_CATEGORIES
    assert len(EXTERNAL_CATEGORIES) == 9
    assert CHECKABLE_CATEGORIES <= EXTERNAL_CATEGORIES
    assert Blocker(category="missing_capability").is_external
    assert not Blocker(category="tool_failure").is_external


def test_the_category_enum_is_unchanged():
    assert ALL_CATEGORIES == [
        "missing_parameter",
        "ambiguous_action",
        "conflicting_constraints",
        "high_risk",
        "missing_capability",
        "tool_failure",
        "conflicting_evidence",
        "risk_discovered",
        "permission_denied",
        "network_unavailable",
        "other",
    ]


def test_every_category_has_a_one_line_definition():
    assert list(CATEGORY_DEFINITIONS) == list(BlockerCategory)
    for text in CATEGORY_DEFINITIONS.values():
        assert text and "\n" not in text and len(text) <= 120
    line = category_definitions_text()
    assert "\n" not in line
    for c in BlockerCategory:
        assert f"{c.value}: " in line
    assert category_definitions_text("\n").count("\n") == len(BlockerCategory) - 1


# ── confidence, errors ───────────────────────────────────────────────────


@pytest.mark.parametrize("conf", [0.0, 1.0, 0.5, None])
def test_confidence_in_bounds(conf):
    assert parse_report({"status": "complete", "confidence": conf}).confidence == conf


@pytest.mark.parametrize("conf", [-0.01, 1.01, "high"])
def test_confidence_out_of_bounds(conf):
    with pytest.raises(ReportParseError):
        parse_report({"status": "complete", "confidence": conf})


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I am done!",
        "{not json}",
        '```json\n{"status": \n```',
        {"completed": ["x"]},  # status missing
        json.dumps({"status": "maybe"}),
    ],
)
def test_invalid_reports(raw):
    with pytest.raises(ReportParseError):
        parse_report(raw)


def test_report_parse_error_is_value_error():
    assert issubclass(ReportParseError, ValueError)


def test_self_report_direct_construction_coerces():
    r = SelfReport(status="abstain", blockers="conflicting_constraints: told not to touch out.txt")
    assert r.status == Status.ABSTAIN
    assert r.blocker_categories == {"conflicting_constraints"}
    assert r.n_blockers_submitted == 1
