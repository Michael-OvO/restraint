"""Operator discovery: every module with a module-level SPEC, nothing else; names unique across modules and tiers."""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from honest_agents.tasks.perturb.rts.base import Applicability, RTSOperatorSpec
from honest_agents.tasks.perturb.rts.registry import RTS_OPERATORS, discover

_SPEC = """\
from honest_agents.tasks.perturb.rts.base import Applicability, RTSOperatorSpec
SPEC = RTSOperatorSpec(
    name={name!r}, tier={tier!r}, instruction_identical=True, expected_category=("missing_capability",),
    applicable=lambda d: Applicability(False, "no"), make_pair=lambda *a, **k: None,
)
"""


@pytest.fixture
def make_pkg(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    made = []

    def make(name: str, modules: dict[str, str]) -> str:
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for mod, body in modules.items():
            (pkg / f"{mod}.py").write_text(textwrap.dedent(body))
        made.append(name)
        importlib.invalidate_caches()
        return name

    yield make
    for name in made:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]


def test_discovers_specs_and_skips_the_rest(make_pkg):
    pkg = make_pkg(
        "rts_ops_ok",
        {
            "zeta": _SPEC.format(name="zeta_op", tier="far"),
            "alpha": _SPEC.format(name="alpha_op", tier="near"),
            "helpers": "X = 1\n",
            "base": _SPEC.format(name="from_base", tier="mid"),
            "registry": _SPEC.format(name="from_registry", tier="mid"),
            "_private": _SPEC.format(name="private", tier="mid"),
        },
    )
    ops = discover(pkg)
    assert list(ops) == ["alpha_op", "zeta_op"]
    assert ops["alpha_op"].tier == "near" and isinstance(ops["zeta_op"], RTSOperatorSpec)


def test_duplicate_names_raise_and_name_the_tiers(make_pkg):
    pkg = make_pkg("rts_ops_dup", {"a": _SPEC.format(name="x", tier="near"), "b": _SPEC.format(name="x", tier="mid")})
    with pytest.raises(ValueError, match=r"declared twice in two tiers \(near, mid\)"):
        discover(pkg)
    pkg = make_pkg("rts_ops_dup2", {"a": _SPEC.format(name="y", tier="mid"), "b": _SPEC.format(name="y", tier="mid")})
    with pytest.raises(ValueError, match="declared twice: rts_ops_dup2.a and rts_ops_dup2.b"):
        discover(pkg)


def test_a_spec_of_the_wrong_type_raises(make_pkg):
    pkg = make_pkg("rts_ops_bad", {"a": "SPEC = {'name': 'x'}\n"})
    with pytest.raises(TypeError, match="not an RTSOperatorSpec"):
        discover(pkg)


def test_spec_validation():
    kw = {"instruction_identical": True, "applicable": lambda d: Applicability(True, ""), "make_pair": print}
    with pytest.raises(ValueError, match="tier"):
        RTSOperatorSpec(name="a", tier="nearish", expected_category=("missing_capability",), **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="external blocker"):
        RTSOperatorSpec(name="a", tier="near", expected_category=("other",), **kw)
    with pytest.raises(ValueError, match="external blocker"):
        RTSOperatorSpec(name="a", tier="near", expected_category=(), **kw)
    with pytest.raises(ValueError, match="carrier_channels"):
        RTSOperatorSpec(name="a", tier="near", expected_category=("risk_discovered",), carrier_channels=("x",), **kw)
    ok = RTSOperatorSpec(name="a", tier="far", expected_category=("conflicting_constraints",), **kw)
    assert ok.carrier_channels == ()


def test_the_real_package_discovers_cleanly():
    assert RTS_OPERATORS == discover()
    for name, spec in RTS_OPERATORS.items():
        assert spec.name == name


def test_every_lane_operator_is_registered_once_in_its_tier():
    """The build plan's operator table, after the near, mid and far lanes merged."""
    tiers = {
        "near": {"force_overwrite", "secret_exposure", "prod_target"},
        "mid": {"remove_tool", "service_down", "missing_credential", "disk_quota", "mem_limit"},
        "far": {"conflicting_tests", "oneoff_tests"},
    }
    assert {t: {n for n, s in RTS_OPERATORS.items() if s.tier == t} for t in tiers} == tiers
    assert set(RTS_OPERATORS) == set().union(*tiers.values())
    # Operators whose planted files are the evidence itself: the freeze reports, not enforces, that channel's cue AUC.
    assert {n for n, s in RTS_OPERATORS.items() if s.carrier_channels} == {"secret_exposure", "prod_target"}
    assert (
        RTS_OPERATORS["secret_exposure"].carrier_channels == RTS_OPERATORS["prod_target"].carrier_channels == ("files",)
    )
