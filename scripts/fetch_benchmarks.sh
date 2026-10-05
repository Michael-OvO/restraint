#!/usr/bin/env bash
# Download the public benchmark task TEXTS the build deduplicates against (L4), at pinned commits. Nothing here is
# training data: these are the held-out evaluation benchmarks, kept so a seed that duplicates one can be rejected.
#
#   scripts/fetch_benchmarks.sh [OUT_DIR=benchmarks]      # then: BENCH_DIR=benchmarks scripts/build_dataset.sh
#
# Layout (each top-level dir is one benchmark label in dedup.jsonl and manifest.json; every task is a dir holding an
# instruction.md, which is what tasks/dedup.py reads):
#   terminal-bench-2/<task>/        laude-institute/terminal-bench-2          @ 2fd12b8   (89 tasks, instruction.md + task.toml)
#   tblite/<task>/                  open-thoughts/OpenThoughts-TBLite (GitHub) @ 5c37b41   (100 tasks)
#   agentic-abstention/<task>__<variant>/   lhannnn/agentic-abstention, terminal/data/immediate  @ ebbf1f5  (TB2 rewrites)
#   agentabstain/<pair>__<type>/    antiquality/agentabstain (HF) tasks.jsonl @ 8422284   (526 act/abstain instructions)
# _manifest.json records the exact commits and the task counts. The directory is gitignored (add `benchmarks/` to .gitignore).
set -euo pipefail

OUT="${1:-benchmarks}"
TB2_REPO=https://github.com/laude-institute/terminal-bench-2;           TB2_COMMIT=2fd12b8
TBLITE_REPO=https://github.com/open-thoughts/OpenThoughts-TBLite; TBLITE_COMMIT=5c37b41
AA_REPO=https://github.com/lhannnn/agentic-abstention;                  AA_COMMIT=ebbf1f51671f8e6d5da42a4188ff8c9254ee978e
AGENTABSTAIN_HF=https://huggingface.co/datasets/antiquality/agentabstain; AGENTABSTAIN_COMMIT=842228426c2a703347396501af61c7890972c7ee

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# checkout_texts <repo> <commit> <dest> <sparse patterns...>: blobless clone, only the listed files, exact commit.
checkout_texts() {
  local repo="$1" commit="$2" dest="$3"; shift 3
  rm -rf "$dest"
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none --no-checkout "$repo" "$dest"
  git -C "$dest" sparse-checkout set --no-cone "$@"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$dest" checkout --quiet "$commit"
  git -C "$dest" rev-parse HEAD
}

echo "terminal-bench-2 @ $TB2_COMMIT"
TB2_FULL=$(checkout_texts "$TB2_REPO" "$TB2_COMMIT" "$WORK/tb2" '/*/instruction.md' '/*/task.toml')
echo "tblite @ $TBLITE_COMMIT"
TBLITE_FULL=$(checkout_texts "$TBLITE_REPO" "$TBLITE_COMMIT" "$WORK/tblite" '/*/instruction.md' '/*/task.toml')
echo "agentic-abstention @ $AA_COMMIT"
AA_FULL=$(checkout_texts "$AA_REPO" "$AA_COMMIT" "$WORK/aa" '/terminal/data/immediate/*.jsonl')
echo "agentabstain @ $AGENTABSTAIN_COMMIT"
mkdir -p "$WORK/agentabstain"
curl -sfL --retry 3 "$AGENTABSTAIN_HF/resolve/$AGENTABSTAIN_COMMIT/tasks.jsonl" -o "$WORK/agentabstain/tasks.jsonl"

python3 - "$WORK" "$OUT" "$TB2_FULL" "$TBLITE_FULL" "$AA_FULL" "$AGENTABSTAIN_COMMIT" <<'PY'
import json, re, shutil, sys
from pathlib import Path

work, out = Path(sys.argv[1]), Path(sys.argv[2])
tb2_sha, tblite_sha, aa_sha, agentabstain_sha = sys.argv[3:7]
safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("._") or "x"
counts = {}


def copy_tasks(src: Path, dst: Path) -> int:
    shutil.rmtree(dst, ignore_errors=True)
    n = 0
    for inst in sorted(src.glob("*/instruction.md")):
        d = dst / inst.parent.name
        d.mkdir(parents=True)
        shutil.copy(inst, d / "instruction.md")
        if (inst.parent / "task.toml").exists():
            shutil.copy(inst.parent / "task.toml", d / "task.toml")
        n += 1
    return n


def jsonl_tasks(rows, dst: Path, name_of, text_of) -> int:
    shutil.rmtree(dst, ignore_errors=True)
    n = 0
    for r in rows:
        text = text_of(r)
        if not text:
            continue
        d = dst / safe(name_of(r))
        d.mkdir(parents=True, exist_ok=True)
        (d / "instruction.md").write_text(text)
        n += 1
    return n


counts["terminal-bench-2"] = copy_tasks(work / "tb2", out / "terminal-bench-2")
counts["tblite"] = copy_tasks(work / "tblite", out / "tblite")
aa_dir = work / "aa" / "terminal" / "data" / "immediate"
rows = []
for f in sorted(aa_dir.glob("*.jsonl")):  # the immediate rewrites and the source TB2 instructions they rewrite
    rows += [{"_f": f.stem, **json.loads(x)} for x in f.read_text().splitlines() if x.strip()]
counts["agentic-abstention"] = jsonl_tasks(
    rows, out / "agentic-abstention", lambda r: f"{r['task_name']}__{r.get('variant', r['_f'])}", lambda r: r.get("instruction")
)
tasks = [json.loads(x) for x in (work / "agentabstain" / "tasks.jsonl").read_text().splitlines() if x.strip()]
counts["agentabstain"] = jsonl_tasks(
    tasks, out / "agentabstain", lambda r: f"{r['pair_id']}__{r.get('task_type', 'task')}", lambda r: r.get("instruction")
)
(out / "_manifest.json").write_text(
    json.dumps(
        {
            "terminal-bench-2": {"repo": "laude-institute/terminal-bench-2", "commit": tb2_sha},
            "tblite": {"repo": "open-thoughts/OpenThoughts-TBLite", "commit": tblite_sha},
            "agentic-abstention": {"repo": "lhannnn/agentic-abstention", "commit": aa_sha},
            "agentabstain": {"repo": "antiquality/agentabstain", "commit": agentabstain_sha},
            "n_tasks": counts,
        },
        indent=2,
    )
)
print(json.dumps(counts))
assert counts["terminal-bench-2"] >= 80 and counts["tblite"] >= 90, "benchmark fetch is incomplete"
PY
echo "benchmarks: $OUT  (pass --benchmark-dir $OUT to tasks.build; BENCH_DIR=$OUT for the build scripts)"
