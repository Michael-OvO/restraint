"""A small audited dataset for hand review (SPEC_V3 §4.9, LV-1: "audit 100 surviving variants by hand").

    python -m honest_agents.tasks.audit --out data/audit --sources endless-terminals --n-sources 24 \\
        --families remove_input revoke_permission [--validation-jsonl data/validation.jsonl] [--no-show-tests]
    python -m honest_agents.tasks.audit --tally data/audit          # after filling in the sheets

Builds the variants of 20-30 source tasks for a few operator families (plain feasible copy, infeasible variant and
near-miss control per family), prefers sources to which every selected family applies, and writes

    <out>/tasks/<variant_id>/       the Harbor task dirs
    <out>/train.jsonl               dataset rows (INTERFACES_V3 §6 columns), usable as `data.train_file` for bring-up
    <out>/sheets/<variant_id>.md    one audit sheet per variant: instruction diff, environment diff, the perturbed
                                    object, every label, the tests (file names, what each test asserts, the oracle
                                    command; `--no-show-tests` hides them), the validation result, a checklist and a
                                    verdict line to fill in
    <out>/INDEX.md                  all variants with links; <out>/audit.jsonl the same as data

Without the tests a reviewer cannot tell whether an impossible variant is truly impossible (a removed input that the
tests only compare with whatever sits at its path can be recreated; one whose content the tests hard-code cannot), so
the sheet shows them. A reviewer reads a sheet and writes `Verdict: ok | wrong | unsure` (and a note). `--tally` counts the verdicts per
family and kind: `wrong` on an infeasible variant is a residual leak (a route the label denies exists), `wrong` on a
control means the obstacle is not harmless, and `wrong` on a plain feasible task means the source task is broken.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.tasks.build import (
    DEFAULT_TEST_BP,
    DEFAULT_VAL_BP,
    BuildConfig,
    Row,
    _hash_key,
    _row_dict,
    _write_jsonl,
    collect_seeds,
    dedup_seeds,
    filter_hermetic,
    make_prompt,
    read_test_ids,
    split_bucket,
    split_for_bucket,
)
from honest_agents.tasks.perturb import OPERATORS, Variant, generate_variants
from honest_agents.tasks.sources import SeedTask
from honest_agents.tasks.taskfs import read_task_toml

log = logging.getLogger(__name__)

KIND_ORDER = {"feasible": 0, "infeasible": 1, "control": 2}
DEFAULT_FAMILIES = ("remove_input", "revoke_permission", "high_risk")  # the training families (build.py)
TEST_ASSERT_CHARS = 200  # what each test asserts, cut here
ORACLE_LINES = 30
DEFAULT_N_SOURCES = 24
MAX_CANDIDATES = 400  # source tasks examined while looking for `n_sources` that every family applies to
VERDICT_RE = re.compile(r"^Verdict:[ \t]*([A-Za-z]*)", re.MULTILINE)
VERDICTS = ("ok", "wrong", "unsure")


@dataclass
class AuditConfig:
    out: Path
    sources: list[str] = field(default_factory=lambda: ["endless-terminals"])
    families: tuple[str, ...] = DEFAULT_FAMILIES
    n_sources: int = DEFAULT_N_SOURCES
    raw_dir: Path | None = None
    seeds_dir: Path | None = None
    limit: int | None = None
    download: bool = True
    benchmark_dirs: list[Path] = field(default_factory=list)
    validation_jsonl: Path | None = None
    allow_legacy_verifier: bool = False
    task_dir_root: str | None = None
    seed: int = 0
    i_know_no_benchmark: bool = False  # tests only, as for tasks.build
    split_seeds: Path | None = None  # split_seeds.json of the dataset build: its test groups are not audited
    test_ids_file: Path | None = None  # committed test_ids.txt: those groups are not audited
    test_bp: int = DEFAULT_TEST_BP  # the dataset's hash cutoffs; a source that hashes into test is never audited
    split_seed: int = 0
    show_tests: bool = True  # tests and oracle command on every sheet (a human cannot judge "impossible" without them)

    def __post_init__(self) -> None:
        unknown = [f for f in self.families if f not in OPERATORS]
        if unknown:
            raise ValueError(f"unknown families {unknown}; known: {sorted(OPERATORS)}")
        if not 1 <= len(self.families):
            raise ValueError("need at least one family")
        if not 1 <= self.n_sources:
            raise ValueError("n_sources must be positive")


def _build_cfg(cfg: AuditConfig) -> BuildConfig:
    return BuildConfig(
        out=cfg.out,
        sources=cfg.sources,
        raw_dir=cfg.raw_dir,
        seeds_dir=cfg.seeds_dir,
        limit=cfg.limit,
        download=cfg.download,
        benchmark_dirs=cfg.benchmark_dirs,
        allow_legacy_verifier=cfg.allow_legacy_verifier,
        i_know_no_benchmark=cfg.i_know_no_benchmark,
        seed=cfg.seed,
    )


def held_out_groups(cfg: AuditConfig, groups: list[str]) -> set[str]:
    """Source groups that are (or would be) test sources of the dataset: the audit's readers tune operators on what they
    see, so no test source may be audited. Hash rule (pool-independent) plus the committed ids and split file."""
    out = {
        g for g in groups if split_for_bucket(split_bucket(g, cfg.split_seed), cfg.test_bp, DEFAULT_VAL_BP) == "test"
    }
    if cfg.test_ids_file is not None:
        out |= read_test_ids(cfg.test_ids_file)
    if cfg.split_seeds is not None:
        out |= {g for g, sp in json.loads(cfg.split_seeds.read_text()).items() if sp == "test"}
    return out


# ── selection ────────────────────────────────────────────────────────────────


def select_sources(cfg: AuditConfig, seeds: list[SeedTask], tasks_root: Path) -> list[tuple[SeedTask, list[Variant]]]:
    """Generate variants in hash order until `n_sources` sources are found to which every family applies.

    Falls back to sources with at least one family when too few qualify. Variants of sources that were examined but not
    selected are deleted again.
    """
    ordered = sorted((s for s in seeds if s.has_solution), key=lambda s: _hash_key(cfg.seed, s.pair_id))[
        :MAX_CANDIDATES
    ]
    full: list[tuple[SeedTask, list[Variant]]] = []
    partial: list[tuple[SeedTask, list[Variant]]] = []
    for s in ordered:
        if len(full) >= cfg.n_sources:
            break
        vs = generate_variants(s.task_dir, tasks_root, pair_id=s.pair_id, operators=list(cfg.families), seed=cfg.seed)
        present = {v.perturbation for v in vs if v.kind == "infeasible"}
        if present >= set(cfg.families):
            full.append((s, vs))
        elif present:
            partial.append((s, vs))
        else:
            _discard(vs)
    chosen = full + partial[: max(0, cfg.n_sources - len(full))]
    for _, vs in partial[max(0, cfg.n_sources - len(full)) :]:
        _discard(vs)
    if len(full) < cfg.n_sources:
        log.warning(
            "only %d sources take all of %s; %d partial", len(full), list(cfg.families), len(chosen) - len(full)
        )
    return chosen[: cfg.n_sources]


def _discard(variants: list[Variant]) -> None:
    for v in variants:
        shutil.rmtree(v.task_dir, ignore_errors=True)


# ── sheets ───────────────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""


def _diff(a: str, b: str, name_a: str, name_b: str) -> str:
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), name_a, name_b, lineterm="", n=2))
    return "\n".join(lines)


def _block(text: str, lang: str = "") -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def _toml_agent_diff(src: Path, dst: Path) -> str:
    a, b = read_task_toml(src).get("agent") or {}, read_task_toml(dst).get("agent") or {}
    keys = sorted(set(a) | set(b))
    return "\n".join(f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in keys if a.get(k) != b.get(k))


def validation_text(result: Any | None, task_dir: Path) -> str:
    if result is None:
        return "Not validated (no result for this variant). Run `python -m honest_agents.tasks.validate` on the Linux node."
    from honest_agents.tasks.validate import is_current

    lines = [
        f"- result: **{'ok' if result.ok else 'FAILED'}**: {result.reason}",
        f"- oracle clean passes: {getattr(result, 'oracle_passes', '?')} of {getattr(result, 'oracle_runs', '?')} runs",
        f"- null check (tests fail on the untouched environment): {getattr(result, 'null_check_failed_as_expected', None)}",
        f"- relabel to control: {getattr(result, 'relabel_to_control', False)}",
        f"- leak (tests pass cleanly on an infeasible variant): {getattr(result, 'leak', False)}",
    ]
    if not is_current(result, task_dir):
        lines.append("- WARNING: the task changed after this result was produced; it is stale")
    return "\n".join(lines)


def test_assertions(task_dir: Path, *, chars: int = TEST_ASSERT_CHARS) -> list[tuple[str, str, str]]:
    """(file, test name, what it asserts) for every test function under tests/: the asserts' source, else the body,
    collapsed to one line and cut at `chars`."""
    out = []
    for f in sorted((task_dir / "tests").glob("test_*.py")):
        src = _read(f)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            out.append((f.name, "(unparsable)", " ".join(src.split())[:chars]))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            parts = [ast.get_source_segment(src, n) or "" for n in (asserts or node.body)]
            out.append((f.name, node.name, " ".join(" ".join(parts).split())[:chars]))
    return out


def oracle_command(task_dir: Path, *, lines: int = ORACLE_LINES) -> str:
    """The oracle solution as it runs (`solution/solve.sh`, comments and blanks dropped, cut at `lines`)."""
    body = [
        ln for ln in _read(task_dir / "solution" / "solve.sh").splitlines() if ln.strip() and not ln.startswith("#")
    ]
    return "\n".join(body[:lines] + ([f"... ({len(body) - lines} more lines)"] if len(body) > lines else []))


def checklist(v: Variant) -> list[str]:
    """One tick line per question a reviewer must answer for this kind of variant (the first one needs the tests)."""
    if v.kind == "infeasible":
        items = [
            (
                "the tests above cannot pass without the removed or locked object (nothing they compare with is "
                "derived from what the agent could recreate)"
            ),
            (
                "no substitute route passes them: recreate the input from the instruction or other image files, "
                "rename or copy around the lock, another tool"
            ),
            "the label names the object a correct decline must point at",
        ]
        if v.perturbation == "remove_input":
            items.insert(
                1, "the removed file is not rebuildable: not typed in the instruction, not derivable from the image"
            )
        return items
    if v.kind == "control":
        return [
            "the obstacle is surmountable by a short, discoverable step and the tests above pass afterwards",
            "a decline would be a mistake: the oracle command shown does finish the task",
        ]
    return ["the tests above check what the instruction asks, and the oracle command passes them"]


def tests_section(task_dir: Path) -> list[str]:
    asserts = test_assertions(task_dir)
    rows = [f"- `{f}::{name}`: `{text}`" if text else f"- `{f}::{name}`" for f, name, text in asserts]
    files = ", ".join(f"`{f.name}`" for f in sorted((task_dir / "tests").glob("*")) if f.is_file())
    return [
        "## Tests (what the grader checks)",
        "",
        f"Files: {files or '(none)'}",
        "",
        *(rows or ["(no test functions found)"]),
        "",
        "Oracle command (`solution/solve.sh`):",
        "",
        _block(oracle_command(task_dir) or "(no oracle)", "bash"),
        "",
    ]


def render_sheet(
    seed: SeedTask,
    v: Variant,
    *,
    twin: Variant | None,
    result: Any | None,
    show_tests: bool = True,
) -> str:
    spec = v.spec
    src, dst = seed.task_dir, v.task_dir
    role = {
        "feasible": "the plain feasible copy: the unperturbed source task",
        "control": f"a near-miss control of `{spec.control_of}`: the cue is present but the obstacle is harmless",
        "infeasible": f"an infeasible variant (`{spec.perturbation}`): the task must be impossible, or wrong to do",
    }[v.kind]
    q = {
        "feasible": "Is the task doable exactly as written, and do the tests check what the instruction asks?",
        "control": (
            "Is the obstacle really harmless: can a competent agent finish the task by a short, discoverable step, "
            "and would a decline be a mistake?"
        ),
        "infeasible": (
            "Is there ANY legitimate route that passes the tests (a substitute tool, recreating the input, renaming "
            "around the lock, satisfying the constraint another way)? If yes the label is wrong (a leak)."
        ),
    }[v.kind]
    out = [
        f"# {v.variant_id}",
        "",
        f"**Kind:** {v.kind} · **Family:** {spec.family or 'feasible'} · **Pair:** `{spec.pair_id}` · **Source:** `{seed.source}`",
        "",
        f"This is {role}.",
        "",
        "## Verdict",
        "",
        q,
        "",
        *[f"- [ ] {c}" for c in checklist(v)],
        "",
        "Verdict: ",
        "",
        "Notes: ",
        "",
        "## Labels",
        "",
        f"- feasible: `{spec.feasible}`; is_control: `{spec.is_control}`; control_of: `{spec.control_of}`",
        f"- expected_blockers: `{spec.expected_blockers}`",
        f"- expected_targets: `{spec.expected_targets}`",
        f"- decoy_targets: `{spec.decoy_targets}`",
        f"- forbidden_paths: `{spec.forbidden_paths}`; irreversible_paths: `{spec.irreversible_paths}`",
        f"- run_tests_when_infeasible: `{spec.run_tests_when_infeasible}`; verifier: `{spec.verifier}`",
        "",
        "## Perturbed object",
        "",
        _block(json.dumps(v.detail, indent=2, default=str), "json") if v.detail else "(none: unperturbed)",
        "",
    ]
    if show_tests:
        out += tests_section(dst)
    if twin is not None:
        out += [f"Paired variant: [{twin.variant_id}]({twin.variant_id}.md) ({twin.kind})", ""]
    inst = _diff(
        _read(src / "instruction.md"), _read(dst / "instruction.md"), "source/instruction.md", "variant/instruction.md"
    )
    out += [
        "## Instruction diff",
        "",
        _block(inst, "diff") if inst else "Instruction unchanged: the obstacle is in the environment.",
        "",
    ]
    env = _diff(
        _read(src / "environment" / "Dockerfile"),
        _read(dst / "environment" / "Dockerfile"),
        "source/Dockerfile",
        "variant/Dockerfile",
    )
    out += ["## Environment diff (Dockerfile)", "", _block(env, "diff") if env else "Dockerfile unchanged.", ""]
    agent = _toml_agent_diff(src, dst)
    out += ["## Agent settings diff (task.toml `[agent]`)", "", _block(agent) if agent else "Unchanged.", ""]
    solve = _diff(
        _read(src / "solution" / "solve.sh"),
        _read(dst / "solution" / "solve.sh"),
        "source/solve.sh",
        "variant/solve.sh",
    )
    out += ["## Oracle diff (solve.sh)", "", _block(solve, "diff") if solve else "Oracle unchanged.", ""]
    tests = []
    for f in sorted((dst / "tests").glob("test_*.py")):
        d = _diff(_read(src / "tests" / f.name), _read(f), f"source/{f.name}", f"variant/{f.name}")
        if d:
            tests.append(d)
    out += ["## Tests diff", "", _block("\n".join(tests), "diff") if tests else "Tests unchanged.", ""]
    out += ["## Validation", "", validation_text(result, dst), ""]
    out += ["## Instruction (full)", "", _block(_read(dst / "instruction.md")), ""]
    return "\n".join(out)


def write_audit(cfg: AuditConfig) -> dict[str, Any]:
    cfg.out.mkdir(parents=True, exist_ok=True)
    bcfg = _build_cfg(cfg)
    seeds, _ = dedup_seeds(bcfg, collect_seeds(bcfg))
    seeds, _ = filter_hermetic(seeds, allow_legacy=cfg.allow_legacy_verifier)
    test_groups = held_out_groups(cfg, sorted({s.split_group for s in seeds}))
    if test_groups:
        log.info("audit skips %d test-split source groups", len(test_groups))
    seeds = [s for s in seeds if s.split_group not in test_groups]
    tasks_root = cfg.out / "tasks"
    chosen = select_sources(cfg, seeds, tasks_root)

    results: dict[str, Any] = {}
    if cfg.validation_jsonl is not None:
        from honest_agents.tasks.validate import read_results

        results = read_results(cfg.validation_jsonl)

    sheets = cfg.out / "sheets"
    shutil.rmtree(sheets, ignore_errors=True)
    sheets.mkdir(parents=True)
    root = cfg.task_dir_root.rstrip("/") if cfg.task_dir_root else str(tasks_root.resolve())
    prompt = make_prompt()
    rows: list[Row] = []
    index: list[dict[str, Any]] = []
    for seed, variants in chosen:
        by_key: dict[tuple, Variant] = {}
        for v in variants:
            by_key[(v.perturbation, v.kind)] = v
        for v in variants:
            twin = None
            if v.kind == "infeasible":
                twin = by_key.get((v.perturbation, "control"))
            elif v.kind == "control":
                twin = by_key.get((v.perturbation, "infeasible"))
            (sheets / f"{v.variant_id}.md").write_text(
                render_sheet(seed, v, twin=twin, result=results.get(v.variant_id), show_tests=cfg.show_tests)
            )
            res = results.get(v.variant_id)
            index.append(
                {
                    "variant_id": v.variant_id,
                    "kind": v.kind,
                    "family": v.spec.family or "feasible",
                    "pair_id": seed.pair_id,
                    "validated": None if res is None else bool(res.ok),
                    "sheet": f"sheets/{v.variant_id}.md",
                    "expected_targets": v.spec.expected_targets,
                    "decoy_targets": v.spec.decoy_targets,
                }
            )
            rows.append(
                Row(
                    prompt=prompt,
                    task_dir=f"{root}/{v.variant_id}",
                    variant_id=v.variant_id,
                    pair_id=seed.pair_id,
                    feasible=v.feasible,
                    is_control=v.is_control,
                    kind=v.kind,
                    family=v.spec.family or "feasible",
                    perturbation=v.perturbation,
                    source=seed.source,
                    split="train",
                )
            )
    _write_jsonl(cfg.out / "train.jsonl", [_row_dict(r) for r in rows])
    for name in ("val", "test"):  # consumers expect all three files; the audit set is one training pool
        _write_jsonl(cfg.out / f"{name}.jsonl", [])
    _write_jsonl(cfg.out / "audit.jsonl", index)
    (cfg.out / "INDEX.md").write_text(render_index(index, cfg))
    summary = {
        "n_sources": len(chosen),
        "n_variants": len(index),
        "kinds": dict(Counter(r["kind"] for r in index)),
        "families": dict(Counter(r["family"] for r in index)),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    (cfg.out / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def render_index(index: list[dict[str, Any]], cfg: AuditConfig) -> str:
    kinds = Counter(r["kind"] for r in index)
    lines = [
        "# Audit set",
        "",
        (
            f"{len({r['pair_id'] for r in index})} source task(s), {len(index)} variants "
            f"({kinds['feasible']} plain feasible, {kinds['control']} controls, {kinds['infeasible']} infeasible) "
            f"in families {', '.join(cfg.families)}."
        ),
        "",
        (
            "Open each sheet, read the instruction and environment diffs, decide whether the label is right, and write "
            "`Verdict: ok`, `Verdict: wrong` or `Verdict: unsure` on its verdict line. Then run "
            f"`python -m honest_agents.tasks.audit --tally {cfg.out}` for the residual leak rate."
        ),
        "",
        "| variant | kind | family | validated | targets | decoys |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(index, key=lambda r: (r["pair_id"], r["family"], KIND_ORDER[r["kind"]])):
        ok = {None: "no result", True: "ok", False: "FAILED"}[r["validated"]]
        lines.append(
            f"| [{r['variant_id']}]({r['sheet']}) | {r['kind']} | {r['family']} | {ok} | "
            f"{', '.join(f'`{t}`' for t in r['expected_targets'])} | {', '.join(f'`{t}`' for t in r['decoy_targets'])} |"
        )
    return "\n".join(lines) + "\n"


# ── tally ────────────────────────────────────────────────────────────────────


def read_verdict(sheet: Path) -> str | None:
    """`ok` / `wrong` / `unsure` as written on the sheet's verdict line; None when it is blank or not a verdict."""
    m = VERDICT_RE.search(sheet.read_text())
    word = m.group(1).lower() if m else ""
    return word if word in VERDICTS else None


def tally(audit_dir: Path) -> dict[str, Any]:
    """Verdict counts per kind and family, and the residual error rate of each kind (wrong / reviewed)."""
    index = [json.loads(line) for line in (audit_dir / "audit.jsonl").read_text().splitlines() if line.strip()]
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in index:
        verdict = read_verdict(audit_dir / r["sheet"])
        counts[(r["kind"], r["family"])][verdict or "pending"] += 1
    rows = []
    for (kind, family), c in sorted(counts.items()):
        reviewed = c["ok"] + c["wrong"] + c["unsure"]
        rows.append(
            {
                "kind": kind,
                "family": family,
                "n": sum(c.values()),
                "ok": c["ok"],
                "wrong": c["wrong"],
                "unsure": c["unsure"],
                "pending": c["pending"],
                "wrong_rate": c["wrong"] / reviewed if reviewed else None,
            }
        )
    by_kind: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for k in ("n", "ok", "wrong", "unsure", "pending"):
            by_kind[r["kind"]][k] += r[k]
    totals = {
        kind: {
            **dict(c),
            "wrong_rate": c["wrong"] / (c["ok"] + c["wrong"] + c["unsure"])
            if (c["ok"] + c["wrong"] + c["unsure"])
            else None,
        }
        for kind, c in by_kind.items()
    }
    return {
        "by_family_kind": rows,
        "by_kind": totals,
        "residual_leak_rate": totals.get("infeasible", {}).get("wrong_rate"),
    }


def parse_args(argv: list[str] | None = None) -> tuple[AuditConfig | None, Path | None]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tally", type=Path, help="count the verdicts written on the sheets of this audit dir and exit")
    ap.add_argument("--sources", nargs="+", default=["endless-terminals"])
    ap.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES), choices=list(OPERATORS))
    ap.add_argument("--n-sources", type=int, default=DEFAULT_N_SOURCES, help="20-30 source tasks")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--seeds-dir", type=Path)
    ap.add_argument("--limit", type=int, help="seed tasks sampled per source before selection")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--benchmark-dir", type=Path, action="append", default=[])
    ap.add_argument("--validation-jsonl", type=Path, help="show validation results on the sheets")
    ap.add_argument("--allow-legacy-verifier", action="store_true")
    ap.add_argument(
        "--show-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="put the test files, what each test asserts and the oracle command on every sheet (default on)",
    )
    ap.add_argument("--task-dir-root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i-know-no-benchmark", action="store_true", help="tests only: skip the mandatory decontamination")
    ap.add_argument(
        "--split-seeds", type=Path, help="split_seeds.json of the dataset build; its test groups are skipped"
    )
    ap.add_argument("--test-ids-file", type=Path, help="test_ids.txt of the dataset build; those groups are skipped")
    ap.add_argument("--test-bp", type=int, default=DEFAULT_TEST_BP, help="the dataset's hash cutoff for the test split")
    a = ap.parse_args(argv)
    if a.tally:
        return None, a.tally
    if a.out is None:
        ap.error("--out is required")
    return (
        AuditConfig(
            out=a.out,
            sources=a.sources,
            families=tuple(a.families),
            n_sources=a.n_sources,
            raw_dir=a.raw_dir,
            seeds_dir=a.seeds_dir,
            limit=a.limit,
            download=not a.no_download,
            benchmark_dirs=a.benchmark_dir,
            validation_jsonl=a.validation_jsonl,
            allow_legacy_verifier=a.allow_legacy_verifier,
            show_tests=a.show_tests,
            task_dir_root=a.task_dir_root,
            seed=a.seed,
            i_know_no_benchmark=a.i_know_no_benchmark,
            split_seeds=a.split_seeds,
            test_ids_file=a.test_ids_file,
            test_bp=a.test_bp,
        ),
        None,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, tally_dir = parse_args(argv)
    if tally_dir is not None:
        print(json.dumps(tally(tally_dir), indent=2))
        return
    assert cfg is not None
    print(json.dumps(write_audit(cfg), indent=2, default=str))


if __name__ == "__main__":
    main()
