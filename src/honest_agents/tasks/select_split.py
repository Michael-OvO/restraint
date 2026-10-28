"""The select split: 60 train-split source tasks for epoch / checkpoint selection (docs/iteration-14-design.md section 4).

    python -m honest_agents.tasks.select_split [--data data] [--n-sources 60] [--seed 0]

Writes `data/select_sources.txt` (one `pair_id` per line, sorted) and `data/select.jsonl` (every existing variant of those
sources, in the dataset-row format with `split = "select"`) and prints the counts by family and kind.

Why: checkpoints used to be chosen on the validation set, which is also the final readout, and on tier F's impossible
sources. The select split is carved out of the *train* split, so val and test stay untouched, from sources the training
data never uses:

* every source of the distill pool (all `distill_pool*.jsonl`; the iteration-13 anchors are own successes drawn from it),
  of the RL pools (`train_rl*.jsonl`, including v2 and the candidate list), of the SFT / RL mixture `train.jsonl` and of
  the teacher bank (`teacher/sources.txt`) is out (`training_sources`);
* candidates are the remaining train-split sources (`split_seeds.json`) that have a built `high_risk` twin and control
  (every select source supplies one pair per evidence surface) and a plain feasible task;
* the 60 with the lowest `sha256(f"select|{seed}|{source}")` are taken, so the choice is a function of the source id and
  the salt, not of file order.

Variants come from the task directories (`data/tasks/*/task.toml`, `[metadata.honest]`), because the dataset builds list
only the sources that reached a pool. Held-out-family variants (`corrupt_input`, `conflicting_constraint`, `no_network`:
test-only by the build's rule) are left out. New data builders (the iteration-14 surfaces) call
`excluded_from_training(data_dir)` and never use a source in it: select, val and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from honest_agents.tasks.build.config import DEFAULT_HELDOUT_FAMILIES, make_prompt

Row = dict[str, Any]

SELECT_SPLIT = "select"
SELECT_SOURCES_FILE = "select_sources.txt"
SELECT_ROWS_FILE = "select.jsonl"
SPLIT_SEEDS_FILE = "split_seeds.json"
TEACHER_SOURCES_FILE = "teacher/sources.txt"
FINAL_PROBE_FILE = "decision/probe_prompts_surfaces.jsonl"  # the decision probe of the iteration-16 final eval
DEFAULT_N_SOURCES = 60
DEFAULT_TASK_DIR_ROOT = "/root/restraint/data/tasks"  # the prefix the dataset builds write (a node's checkout)
HIGH_RISK = "high_risk"
KIND_ORDER = {"feasible": 0, "control": 1, "infeasible": 2}
# Files whose sources training uses (globs under the data directory). Order irrelevant.
TRAINING_POOL_GLOBS = ("distill_pool*.jsonl", "train_rl*.jsonl", "train.jsonl")


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_lines(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def split_sources(data_dir: Path, split: str) -> set[str]:
    """Source ids (`pair_id`) the build assigned to `split` (`split_seeds.json`)."""
    seeds = json.loads((data_dir / SPLIT_SEEDS_FILE).read_text())
    return {source for source, s in seeds.items() if s == split}


def training_sources(data_dir: Path) -> set[str]:
    """Sources some training stage uses: distill pools, RL pools, `train.jsonl` and the teacher bank."""
    out: set[str] = set()
    for pattern in TRAINING_POOL_GLOBS:
        for path in sorted(data_dir.glob(pattern)):
            out |= {str(r["pair_id"]) for r in read_jsonl(path) if r.get("pair_id")}
    teacher = data_dir / TEACHER_SOURCES_FILE
    if teacher.is_file():
        out |= read_lines(teacher)
    return out


def load_select_sources(data_dir: Path) -> set[str]:
    path = data_dir / SELECT_SOURCES_FILE
    return read_lines(path) if path.is_file() else set()


def excluded_from_training(data_dir: str | Path = "data") -> set[str]:
    """Sources no training data may use: the select sources, and every validation and test source.

    Call this from any builder of new training data (the iteration-14 evidence surfaces, splices, pools) and drop rows
    whose source is in it. The select file must exist (build it first): a missing file would silently let the select
    sources into training."""
    d = Path(data_dir)
    if not (d / SELECT_SOURCES_FILE).is_file():
        raise FileNotFoundError(
            f"{d / SELECT_SOURCES_FILE} is missing: run `python -m honest_agents.tasks.select_split`"
        )
    return load_select_sources(d) | split_sources(d, "val") | split_sources(d, "test")


def final_probe_sources(data_dir: str | Path = "data") -> set[str]:
    """Sources of the decision probe that iteration 16's final eval runs (`FINAL_PROBE_FILE`; prereg section 4: nothing
    in the final benchmark is trained on, DEF-63).

    The probe's prompts are spliced from train-split sources, so `excluded_from_training` does not hold them (and the
    builders of the probe itself need them). The file must exist: a missing one would silently let them into training."""
    path = Path(data_dir) / FINAL_PROBE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the final decision probe's sources cannot be excluded")
    return {str(r["pair_id"]) for r in read_jsonl(path) if r.get("pair_id")}


def excluded_from_commit_training(data_dir: str | Path = "data") -> set[str]:
    """The iteration-16 commit-data builds' exclusion: `excluded_from_training` and `final_probe_sources`."""
    return excluded_from_training(data_dir) | final_probe_sources(data_dir)


def _rank(seed: int, source: str) -> str:
    return hashlib.sha256(f"{SELECT_SPLIT}|{seed}|{source}".encode()).hexdigest()


def variants_by_source(
    tasks_dir: Path, sources: Iterable[str], heldout_families: Iterable[str] = DEFAULT_HELDOUT_FAMILIES
) -> dict[str, list[Row]]:
    """Every variant directory of `sources` as a row without prompt / task_dir root, from `task.toml`."""
    wanted = set(sources)
    heldout = set(heldout_families)
    out: dict[str, list[Row]] = {s: [] for s in wanted}
    for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        toml = d / "task.toml"
        if not toml.is_file():
            continue
        meta = tomllib.loads(toml.read_text()).get("metadata", {})
        honest = meta.get("honest", {})
        pair = honest.get("pair_id") or (meta.get("provenance") or {}).get("pair_id")
        if pair not in wanted or honest.get("family") in heldout:
            continue
        feasible = bool(honest.get("feasible", True))
        is_control = bool(honest.get("is_control", False))
        row: Row = {
            "variant_id": honest.get("variant_id") or d.name,
            "dir": d.name,
            "pair_id": pair,
            "feasible": feasible,
            "is_control": is_control,
            "kind": "infeasible" if not feasible else "control" if is_control else "feasible",
            "family": honest.get("family") or "feasible",
            "perturbation": honest.get("perturbation"),
            "source": (meta.get("provenance") or {}).get("source") or str(pair).split("/")[0],
        }
        if honest.get("detail"):
            row["detail"] = honest["detail"]
        out[pair].append(row)
    return out


def _eligible(variants: Sequence[Row]) -> bool:
    kinds = {(r["family"], r["kind"]) for r in variants}
    return {(HIGH_RISK, "infeasible"), (HIGH_RISK, "control"), ("feasible", "feasible")} <= kinds


def choose_sources(
    candidates: Iterable[str], variants: dict[str, list[Row]], n: int = DEFAULT_N_SOURCES, seed: int = 0
) -> list[str]:
    """The `n` eligible candidates with the lowest hash rank; raises when fewer are eligible."""
    eligible = sorted((s for s in candidates if _eligible(variants.get(s, []))), key=lambda s: (_rank(seed, s), s))
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} eligible sources (a high_risk twin, its control and a plain task), {n} wanted"
        )
    return eligible[:n]


def to_rows(
    sources: Sequence[str], variants: dict[str, list[Row]], task_dir_root: str = DEFAULT_TASK_DIR_ROOT
) -> list[Row]:
    """Dataset rows (build format, `split = "select"`), by source in the given order, plain / control / infeasible."""
    prompt = make_prompt()
    root = task_dir_root.rstrip("/")
    rows: list[Row] = []
    for s in sources:
        for v in sorted(variants[s], key=lambda r: (KIND_ORDER[r["kind"]], r["family"], r["variant_id"])):
            row = {k: x for k, x in v.items() if k != "dir"}
            rows.append({"prompt": prompt, "task_dir": f"{root}/{v['dir']}", **row, "split": SELECT_SPLIT})
    return rows


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    return {
        "sources": len({r["pair_id"] for r in rows}),
        "rows": len(rows),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "by_family_kind": {
            f"{fam} / {kind}": n for (fam, kind), n in sorted(Counter((r["family"], r["kind"]) for r in rows).items())
        },
    }


def build(
    data_dir: Path, *, n: int = DEFAULT_N_SOURCES, seed: int = 0, task_dir_root: str = DEFAULT_TASK_DIR_ROOT
) -> tuple[list[str], list[Row]]:
    """The chosen sources (sorted) and their rows."""
    candidates = split_sources(data_dir, "train") - training_sources(data_dir)
    variants = variants_by_source(data_dir / "tasks", candidates)
    chosen = sorted(choose_sources(candidates, variants, n, seed))
    return chosen, to_rows(chosen, variants, task_dir_root)


def rebuild_rows(data_dir: Path, sources: Sequence[str], *, task_dir_root: str = DEFAULT_TASK_DIR_ROOT) -> list[Row]:
    """The rows of an already frozen source list, from the task directories as they are now (new variants of a select
    source, such as the iteration-14 surface pairs, come in; the sources never change).

    Refuses a frozen source that training has used since, or that lost its high_risk pair or plain task: either would
    make the select split something other than what `excluded_from_training` promised the data builders."""
    used = sorted(set(sources) & training_sources(data_dir))
    if used:
        raise ValueError(f"{len(used)} select source(s) are now used by training data: {used[:3]}...")
    variants = variants_by_source(data_dir / "tasks", sources)
    bad = sorted(s for s in sources if not _eligible(variants.get(s, [])))
    if bad:
        raise ValueError(f"{len(bad)} select source(s) lost their high_risk pair or plain task: {bad[:3]}...")
    return to_rows(sources, variants, task_dir_root)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n-sources", type=int, default=DEFAULT_N_SOURCES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-dir-root", default=DEFAULT_TASK_DIR_ROOT, help="path prefix written into `task_dir`")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rewrite the rows of the frozen sources from the task directories as they are now (new variants of a "
        "select source come in; the sources themselves never change: delete both files to choose again)",
    )
    args = ap.parse_args(argv)
    sources_path, rows_path = args.data / SELECT_SOURCES_FILE, args.data / SELECT_ROWS_FILE
    if (sources_path.is_file() or rows_path.is_file()) and not args.force:
        raise SystemExit(
            f"{sources_path} / {rows_path} exist: the select split is frozen once built; --force to rebuild its rows"
        )
    if sources_path.is_file():
        # The frozen list wins over a fresh choice: new eligible sources (a twin built since) must not move it, because
        # the training data was built against it (`excluded_from_training`).
        chosen = sorted(load_select_sources(args.data))
        if len(chosen) != args.n_sources:
            raise SystemExit(
                f"{sources_path} holds {len(chosen)} sources, --n-sources {args.n_sources}: delete both files to choose again"
            )
        try:
            rows = rebuild_rows(args.data, chosen, task_dir_root=args.task_dir_root)
        except ValueError as e:
            raise SystemExit(str(e)) from e
    else:
        chosen, rows = build(args.data, n=args.n_sources, seed=args.seed, task_dir_root=args.task_dir_root)
    sources_path.write_text("".join(f"{s}\n" for s in chosen))
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summarize(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
