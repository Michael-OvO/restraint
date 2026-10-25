"""analysis.power on synthetic data with known variance (DEF-10): components, MDE, the estimator, clone-pass."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from honest_agents.analysis import power as pw
from honest_agents.eval.records import record_line
from honest_agents.tasks.perturb.risk_surfaces import SURFACES_HELDOUT, SURFACES_TRAIN
from honest_agents.tasks.taskfs import write_task_toml
from tests.eval.scripted_policy import COMPLETE, make_record

SURFACES = (*SURFACES_TRAIN, *SURFACES_HELDOUT)
READ = pw.METRICS["read"]
GAP = pw.METRICS["gap"]


def design(k: int, pairs: int, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """(source, surface, pool pair) cells: every source has every surface, its pair drawn from `pairs` per surface."""
    return [(f"s{i}", surf, f"{surf}-{rng.integers(pairs)}") for i in range(k) for surf in SURFACES]


def gaussian_run(cells, samples, rng, *, src, pair, unit, noise, effects=None, shift=0.0):
    """Episodes whose `read` value is mu + a_source + b_pair + c_unit + e (all normal, the given variances), plus the
    unit-level `effects` (a dict unit -> value) and `shift`. Returns (episodes, truth per unit)."""
    sources = sorted({c[0] for c in cells})
    pairs = sorted({c[2] for c in cells})
    a = dict(zip(sources, rng.normal(0, math.sqrt(src), len(sources)), strict=True))
    b = dict(zip(pairs, rng.normal(0, math.sqrt(pair), len(pairs)), strict=True))
    eps, truth = [], {}
    for s, surf, p in cells:
        for twin in (True, False):
            task = f"{s}-{surf}-{'tw' if twin else 'ct'}"
            mu = 0.5 + a[s] + b[p] + rng.normal(0, math.sqrt(unit)) + shift + (effects or {}).get(task, 0.0)
            truth[task] = mu
            for k in range(samples):
                y = mu + rng.normal(0, math.sqrt(noise))
                eps.append(pw.Episode(task, s, p, surf, twin, f"{surf}/{twin}", k, {"read": (y, 1.0)}))
    return eps, truth


# ── the numerics ─────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("q", "df", "want"), [(0.95, 19, 1.7291), (0.80, 19, 0.8610), (0.975, 1, 12.7062),
                                               (0.95, 1000, 1.6464), (0.90, 67, 1.2943)])  # fmt: skip
def test_t_quantiles(q, df, want):
    assert pw.t_ppf(q, df) == pytest.approx(want, abs=2e-4)


def test_linearised_values_average_to_the_ratio_and_a_rate_gets_the_binomial_variance():
    num = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, np.nan, np.nan], [1.0, 1.0, 0.0, np.nan]])
    den = np.where(np.isnan(num), np.nan, 1.0)
    r, ell = pw.part_values(num, den)
    assert r == pytest.approx(5 / 9) and ell.mean() == pytest.approx(r)
    ratio_num = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ratio_den = np.array([[1.0, 1.0], [0.0, 1.0], [2.0, 1.0]])  # a ratio metric: units weigh by their denominators
    r2, ell2 = pw.part_values(ratio_num, ratio_den)
    assert r2 == pytest.approx(4 / 6) and ell2.mean() == pytest.approx(r2)
    eps = [pw.Episode("t", "s", None, "x", False, "x", k, {"read": (float(k < 1), 1.0)}) for k in range(4)]
    eps += [pw.Episode("u", "s2", None, "x", False, "x", k, {"read": (float(k < 2), 1.0)}) for k in range(4)]
    _, _, var = pw.values(pw.unit_data(eps, READ))
    assert var == pytest.approx([0.25 * 0.75 / 3, 0.5 * 0.5 / 3])  # p(1-p)/(S-1): unbiased for p(1-p)/S


def test_sign_flip_is_exact_for_few_clusters():
    assert pw.sign_flip(np.array([1.0, 2.0, 3.0]), 3) == pytest.approx(2 / 8)
    assert pw.sign_flip(np.array([1.0, -1.0]), 2) == 1.0
    many = pw.sign_flip(np.full(30, 1.0), 30, n_flip=2000)
    assert many == pytest.approx(1 / 2001)


# ── variance components ──────────────────────────────────────────────────────────────────────────────────────────

COMP = {"src": 0.04, "pair": 0.02, "unit": 0.01, "noise": 0.09}


def test_components_recover_known_source_pair_unit_and_sampling_variance():
    rng = np.random.default_rng(1)
    cells = design(300, 40, rng)
    eps, _ = gaussian_run(cells, 4, rng, **COMP)
    c = pw.run_components(pw.unit_data(eps, READ))
    assert c["source"] == pytest.approx(0.04, abs=0.008)
    assert c["pair"] == pytest.approx(0.02, abs=0.006)
    assert c["cell"] == pytest.approx(0.0, abs=0.004)
    assert c["unit"] == pytest.approx(0.01, abs=0.004)
    assert c["sampling"] == pytest.approx(0.09 / 4, abs=0.001)


def test_difference_components_are_the_interactions_and_main_effects_cancel():
    rng = np.random.default_rng(2)
    cells = design(300, 40, rng)
    ref, truth = gaussian_run(cells, 4, rng, **COMP)
    x_src = {s: rng.normal(0, 0.1) for s in {c[0] for c in cells}}  # model x source variance 0.01
    x_unit = {t: rng.normal(0, 0.1) for t in truth}  # model x unit variance 0.01
    # the candidate keeps the reference's truth (source, pair and unit main effects) and adds the interactions
    cand = [pw.Episode(e.task, e.source, e.pair, e.surface, e.twin, e.stratum, e.sample,
                       {"read": (truth[e.task] + x_src[e.source] + x_unit[e.task] + rng.normal(0, 0.3), 1.0)})
            for e in ref]  # fmt: skip
    c = pw.diff_components(pw.unit_data(ref, READ), pw.unit_data(cand, READ))
    assert c["source"] == pytest.approx(0.01, abs=0.004)
    assert c["pair"] == pytest.approx(0.0, abs=0.003)
    assert c["unit"] == pytest.approx(0.01, abs=0.004)
    assert c["sampling"] == pytest.approx(2 * 0.09 / 4, abs=0.002)


# ── MDE and the estimator ────────────────────────────────────────────────────────────────────────────────────────


def _replicate(cells, n_rep, rng, effect, x_src=0.01, x_unit=0.005, noise=0.09, samples=4):
    """n_rep paired evals on one fixed design: d_u = effect + model x source + model x unit + sampling."""
    template, _ = gaussian_run(cells, samples, rng, src=0.0, pair=0.0, unit=0.0, noise=noise)
    data = pw.unit_data(template, READ)
    src_codes, k = pw._codes(data.source)
    d = (effect + rng.normal(0, math.sqrt(x_src), (n_rep, k))[:, src_codes]
         + rng.normal(0, math.sqrt(x_unit + 2 * noise / samples), (n_rep, data.n_units)))  # fmt: skip
    comp = {"source": x_src, "pair": 0.0, "cell": 0.0, "unit": x_unit}
    return data, d, comp, 2 * noise


def test_power_at_the_mde_is_80_percent_and_a_clone_fails_the_ni10_guard_10_percent_of_the_time():
    rng = np.random.default_rng(3)
    cells = design(20, 12, rng)
    data, d0, comp, s2 = _replicate(cells, 4000, rng, 0.0)
    var = pw.design_variance(comp, s2, data, pw.Design(20, 4))
    m, ni10 = pw.mde(var, 20), pw.ni_margin(var, 20)
    cl = pw.Clusters.of(data)
    half = pw.t_ppf(0.95, cl.df) * np.sqrt(cl.variance(d0))
    assert np.mean(np.abs(d0.mean(1)) < half) == pytest.approx(0.90, abs=0.03)  # 90 % CI coverage under the null
    assert np.mean(d0.mean(1) - half > -ni10) == pytest.approx(0.90, abs=0.03)
    assert np.mean(d0.mean(1) - half > -m) == pytest.approx(0.80, abs=0.04)
    d1 = d0 + m
    assert np.mean(d1.mean(1) - half > 0) == pytest.approx(0.80, abs=0.04)  # power 0.8 at the MDE


def test_more_sources_shrink_the_mde_but_the_pair_term_is_a_floor():
    rng = np.random.default_rng(4)
    data = pw.unit_data(gaussian_run(design(20, 12, rng), 4, rng, **COMP)[0], READ)
    comp = {"source": 0.01, "pair": 0.01, "cell": 0.0, "unit": 0.0}
    v20 = pw.design_variance(comp, 0.18, data, pw.Design(20, 4))
    v68 = pw.design_variance(comp, 0.18, data, pw.Design(68, 4))
    v_huge = pw.design_variance(comp, 0.18, data, pw.Design(100000, 4))
    assert v68 < v20 and v_huge > 0.01 * pw.pair_weight_sq(data, 100000) * 0.99 > 0


def test_paired_estimate_on_a_known_shift():
    rng = np.random.default_rng(5)
    cells = design(20, 12, rng)
    ref, truth = gaussian_run(cells, 4, rng, **COMP)
    cand = [pw.Episode(e.task, e.source, e.pair, e.surface, e.twin, e.stratum, e.sample,
                       {"read": (truth[e.task] + 0.3 + rng.normal(0, 0.3), 1.0)}) for e in ref]  # fmt: skip
    est = pw.paired_estimate(pw.unit_data(ref, READ), pw.unit_data(cand, READ))
    assert abs(est["diff"] - 0.3) < 4 * est["se"] and est["lo"] > 0
    assert est["se"] == pytest.approx(math.sqrt(2 * 0.09 / 4 / 240), rel=0.35)  # sampling only: no interaction
    assert est["p_signflip_source"] < 0.01 and est["p_signflip_pair"] < 0.01 and est["significant"]
    assert est["sources"] == 20 and est["units"] == 240


# ── clone-pass and the report (DEF-10: a bar at the reference's point estimate) ─────────────────────────────────


def bernoulli_surface_run(cells, samples, rng, p_read, p_ac, p_decl):
    eps = []
    for s, surf, p in cells:
        for twin in (True, False):
            task = f"{s}-{surf}-{'tw' if twin else 'ct'}"
            for k in range(samples):
                v = {"read": (float(rng.random() < p_read[task]), 1.0)}
                if twin:
                    v["abstain_correct"] = (float(rng.random() < p_ac[task]), 1.0)
                else:
                    v["decline"] = (float(rng.random() < p_decl[task]), 1.0)
                eps.append(pw.Episode(task, s, p, surf, twin, f"{surf}/{'twin' if twin else 'control'}", k, v))
    return eps


def _probs(cells, rng):
    tasks = [f"{s}-{surf}-{b}" for s, surf, _ in cells for b in ("tw", "ct")]
    return [dict(zip(tasks, rng.uniform(0.1, 0.9, len(tasks)), strict=True)) for _ in range(3)]


def test_a_bar_at_the_reference_point_estimate_is_passed_by_half_of_its_clones_and_flagged():
    rng = np.random.default_rng(6)
    cells = design(20, 12, rng)
    eps = bernoulli_surface_run(cells, 4, rng, *_probs(cells, rng))
    data = pw.unit_data(eps, GAP)
    point = pw.values(data)[0]
    p_point = pw.clone_pass(data, data, pw.Bar("gap", "point_ge", point), samples=4, n_sim=2000)
    assert p_point == pytest.approx(0.5, abs=0.06)
    s2 = 2 * pw.per_sample_sampling(data)
    var = pw.design_variance({}, s2, data, pw.Design(20, 4))
    assert pw.clone_pass(data, data, pw.guard(GAP, pw.ni_margin(var, 20)), samples=4) == pytest.approx(0.9, abs=0.03)
    assert pw.clone_pass(data, data, pw.guard(GAP, pw.mde(var, 20)), samples=4) == pytest.approx(0.8, abs=0.04)
    result = pw.analyse({"surface": {pw.REF: eps}}, sources=[20, 68], samples=[4],
                        bars=[pw.Bar("gap", "point_ge", point)], n_sim=1000)  # fmt: skip
    clone = {c["bar"]: c for c in result["surface/gap"]["clone"]}
    assert clone[pw.Bar("gap", "point_ge", point).label()]["flag"]
    assert not any(c["flag"] for b, c in clone.items() if "lb_gt" in b and c["sampling"] > 0.87)
    assert (
        result["surface/gap"]["mde"]["K68/S4/sampling"]["mde"] < result["surface/gap"]["mde"]["K20/S4/sampling"]["mde"]
    )


def test_a_lower_is_better_metric_gets_an_upper_guard():
    assert pw.guard(pw.METRICS["control_decline"], 0.05) == pw.Bar("control_decline", "ub_lt", 0.05)
    assert pw.guard(pw.METRICS["read"], 0.05) == pw.Bar("read", "lb_gt", -0.05)
    assert pw.Bar.parse("gap_heldout:point_ge:16.2") == pw.Bar("gap_heldout", "point_ge", pytest.approx(0.162))
    with pytest.raises(ValueError):
        pw.Bar.parse("nope:point_ge:1")


# ── end to end on archived-format files ──────────────────────────────────────────────────────────────────────────


def _surface_files(tmp_path: Path, rng: np.random.Generator) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "tasks"
    cells = design(6, 3, rng)
    for s, surf, p in cells:
        for twin in (True, False):
            target = f"/data/{s}/{surf}"
            site = {"surface": "readme", "target": target, "evidence": [f"{target}/README"],
                    "expected_targets": [target], "protected": [target]}  # fmt: skip
            variant = {"operator": "high_risk", "path": target, "surface_site": site, "pool_pair": p}
            d = root / f"{s}-{surf}-{'tw' if twin else 'ct'}"
            d.mkdir(parents=True)
            write_task_toml(d, {"metadata": {"honest_variant": variant}})
    files = {}
    for label in pw.RUNS["surface"]:
        rows = []
        for s, surf, _ in cells:
            for twin in (True, False):
                task = f"{s}-{surf}-{'tw' if twin else 'ct'}"
                for k in range(4):
                    read = rng.random() < 0.4
                    decl = rng.random() < (0.6 if twin else 0.3)
                    rows.append({"family": "high_risk", "feasible": not twin, "task_id": task, "source_task": s,
                                 "sample": k, "detail": {"surface": surf},
                                 "extra": {"history": [{"command": f"cat /data/{s}/{surf}/README" if read else "ls"}]},
                                 "reward": {"claimed_complete": not decl, "has_report": True,
                                            "case": "abstain_correct" if decl else "x"}})  # fmt: skip
        files[label] = tmp_path / f"{label}.jsonl"
        files[label].write_text("\n".join(json.dumps(r) for r in rows))
    return root, files


def _write_archive(tmp_path: Path, rng: np.random.Generator) -> Path:
    archive = tmp_path / "archive"
    root, files = _surface_files(tmp_path, rng)
    for label, rel in pw.RUNS["surface"].items():
        (archive / rel).mkdir(parents=True)
        (archive / rel / "episodes.jsonl.gz").write_bytes(gzip.compress(files[label].read_bytes()))
    for ds in ("tierS", "tierF"):
        for rel in pw.RUNS[ds].values():
            recs = [make_record(f"t{i}", feasible=True, report=COMPLETE, passed=bool(rng.random() < 0.7), sample=k,
                                source_task=f"src{i // 2}") for i in range(16) for k in range(2)]  # fmt: skip
            (archive / rel).mkdir(parents=True, exist_ok=True)
            (archive / rel / "episodes.jsonl.gz").write_bytes(gzip.compress("".join(map(record_line, recs)).encode()))
    (tmp_path / "root").symlink_to(root)
    return archive


def test_report_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(7)
    archive = _write_archive(tmp_path, rng)
    out, md = tmp_path / "o.json", tmp_path / "o.md"
    argv = ["report", "--archive", str(archive), "--tasks-root", str(tmp_path / "root"), "--n-sim", "200",
            "--json", str(out), "--markdown", str(md), "--bar", "gap:point_ge:10"]  # fmt: skip
    assert pw.main(argv) == 0
    text = capsys.readouterr().out
    assert "## MDE at power 0.8" in text and "| surface/read | 6 | 4 |" in text and "| surface/read | 68 | 4 |" in text
    data = json.loads(out.read_text())
    assert set(data) >= {"surface/read", "surface/gap_heldout", "tierS/verified_success", "tierF/false_claim_share"}
    assert 0 < data["tierF/verified_success"]["value"][pw.REF] < 1
    assert "estimate_seed" in data["surface/gap"] and "estimate_seed" not in data["tierS/verified_success"]
    assert any(c["bar"] == "gap:point_ge:+10.0" for c in data["surface/gap"]["clone"])
    assert md.read_text().strip() == text.strip()


def test_estimate_cli_and_missing_task_dirs(tmp_path, capsys):
    rng = np.random.default_rng(8)
    root, files = _surface_files(tmp_path, rng)
    out = tmp_path / "e.json"
    argv = ["estimate", str(files[pw.REF]), str(files[pw.SEED]), "--metric", "gap", "--tasks-root", str(root)]
    assert pw.main([*argv, "--json", str(out)]) == 0
    assert "gap: reference" in capsys.readouterr().out
    est = json.loads(out.read_text())
    assert est["units"] == 36 and est["sources"] == 6 and est["lo"] < est["diff"] < est["hi"]
    next(root.iterdir()).joinpath("task.toml").unlink()
    assert pw.main(argv) == 2
    assert "no task dir" in capsys.readouterr().err
