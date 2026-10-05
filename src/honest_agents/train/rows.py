"""Dataset rows: loading, pairing and ordering, validation, split isolation and resume-order fingerprints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from honest_agents.train.config import DATA_ORDER_FILE, SPLIT_TRAIN, VALIDATION_SPLITS, TrainRunConfig

logger = logging.getLogger("honest_agents.train")


# ── rows ─────────────────────────────────────────────────────────────────────


def pair_ordered(rows: list[dict[str, Any]], seed: int, round_size: int | None = None) -> list[dict[str, Any]]:
    """Shuffle rows as pair_id groups (rows without a pair_id are singleton groups); members stay adjacent.

    With `round_size` (prompts per generation round), groups are first-fit packed into rounds of exactly that many rows
    so no group straddles a round boundary. Mere adjacency is not enough: a single unpaired row shifts every later
    pair across a boundary. Rows that do not fill a whole round go last (TRL drops the final partial chunk).
    """
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for i, r in enumerate(rows):
        groups.setdefault(r.get("pair_id") or f"__row{i}", []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    # Feasible first within a pair: deterministic, and irrelevant to the loss.
    for k in keys:
        groups[k].sort(key=lambda r: not r.get("feasible", True))
    if not round_size:
        return [r for k in keys for r in groups[k]]

    rounds: list[list[dict[str, Any]]] = []
    for k in keys:
        g = groups[k]
        if len(g) > round_size:
            logger.warning("pair_id %r has %d rows > %d prompts per round; splitting it", k, len(g), round_size)
        for j in range(0, len(g), round_size):
            part = g[j : j + round_size]
            slot = next((b for b in rounds if len(b) + len(part) <= round_size), None)
            if slot is None:
                rounds.append(slot := [])
            slot.extend(part)
    full = [b for b in rounds if len(b) == round_size]
    partial = [r for b in rounds if len(b) < round_size for r in b]
    if partial:
        logger.info("%d row(s) do not fill a whole generation round and go last", len(partial))
    return [r for b in full for r in b] + partial


def load_rows(
    path: str | Path, max_rows: int | None = None, *, allow_custom_system_prompt: bool = False
) -> list[dict[str, Any]]:
    """Read the dataset builder's JSONL.

    Required per row: `task_dir` (absolute, or relative to the jsonl's directory). Optional: `prompt` (list of chat
    messages ending in a user turn; TRL appends `env.reset()`'s instruction to it); when absent we use
    `[system: SYSTEM_PROMPT, user: ""]`. A row's system message must be `harness.SYSTEM_PROMPT`: training and
    evaluation share one prompt. Every other column is passed through to `HonestHarborEnv.reset`.
    """
    from honest_agents.harness import SYSTEM_PROMPT

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "task_dir" not in r:
                raise ValueError(f"{path}:{ln}: row has no `task_dir`")
            td = Path(r["task_dir"])
            r["task_dir"] = str(td if td.is_absolute() else (path.parent / td).resolve())
            prompt = r.get("prompt") or [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ""},
            ]
            if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
                raise ValueError(f"{path}:{ln}: `prompt` must be a list of messages ending with a user message")
            system = [m.get("content") for m in prompt if m.get("role") == "system"]
            if not allow_custom_system_prompt and any(s != SYSTEM_PROMPT for s in system):
                raise ValueError(
                    f"{path}:{ln}: the system message differs from harness.SYSTEM_PROMPT; training rows use the one "
                    "prompt evaluation uses (data.allow_custom_system_prompt to override)"
                )
            r["prompt"] = prompt
            rows.append(r)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _constant_prompt() -> list[dict[str, str]]:
    from honest_agents.harness import SYSTEM_PROMPT

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]


def holdout_ids(cfg: TrainRunConfig) -> set[str]:
    """The held-out test ids (`data.holdout_ids`, or `test_ids.txt` beside the train file); empty if there is none."""
    from honest_agents.tasks.build import read_test_ids

    explicit = cfg.data.holdout_ids
    path = Path(explicit) if explicit else Path(cfg.data.train_file).parent / "test_ids.txt"
    if not path.is_file():
        if explicit:
            raise ValueError(f"data.holdout_ids {explicit!r} does not exist")
        return set()
    return read_test_ids(path)  # raises on a missing or wrong sha256 line


def _source_group(pair_id: str) -> str:
    """The split group of a pair id (SETA_Evolve variants go with their parent, as in `SeedTask.split_group`)."""
    return re.sub(r"__[bd]\d+$", "", pair_id) if pair_id.startswith("seta-env/") else pair_id


def check_rows(rows: list[dict[str, Any]], cfg: TrainRunConfig, *, role: str) -> dict[str, Any]:
    """Guards on one dataset file (`role` is `train` or `validation`); returns counts for the summary.

    * no row's pair_id is in the held-out `test_ids.txt` (`holdout_ids`);
    * the split column: training rows are always `train` (a missing split column is refused), probe rows always `validation`;
    * `variant_id` is unique (it is the prompt-group id the reward masks leaks by);
    * every row's task says `verifier = "pytest"` and its labels agree with the row's columns;
    * every infeasible row records `expected_blockers` and `expected_targets`.
    """
    from honest_agents.spec import VERIFIER_PYTEST, HonestTaskSpec

    where = f"{role} rows"
    held_out = holdout_ids(cfg)
    seen: set[str] = set()
    counts: dict[str, int] = {"rows": len(rows)}
    specs: dict[str, HonestTaskSpec] = {}
    for i, r in enumerate(rows):
        split = r.get("split")
        if role == "train" and split != SPLIT_TRAIN:  # a missing column is refused too: an unlabelled row may be test
            raise ValueError(f"{where}: row {i} has split={split!r}; only split='train' rows may be trained on")
        if role == "validation" and split not in VALIDATION_SPLITS:
            raise ValueError(
                f"{where}: row {i} has split={split!r}; the validation probe runs on split in {VALIDATION_SPLITS} "
                "only, never on the test split"
            )
        if not cfg.data.allow_custom_system_prompt and "prompt" in r and r["prompt"] != _constant_prompt():
            raise ValueError(  # TRL appends reset()'s instruction to it: anything else here is extra, row-specific text
                f"{where}: row {i} `prompt` is not the constant [system prompt, empty user turn]; the only "
                "row-specific text the policy may see is instruction.md"
            )
        pair = r.get("pair_id")
        if pair is not None and (pair in held_out or _source_group(pair) in held_out):
            raise ValueError(
                f"{where}: row {i} pair_id {pair!r} is a held-out test id (test_ids.txt); refusing to use it"
            )
        vid = r.get("variant_id")
        if vid is not None:
            if vid in seen:
                raise ValueError(f"{where}: variant_id {vid!r} appears twice; it identifies the prompt group")
            seen.add(vid)
        if not cfg.data.check_tasks:
            continue
        td = r["task_dir"]
        if td not in specs:
            if not (Path(td) / "task.toml").is_file():
                raise ValueError(f"{where}: row {i} task_dir {td!r} has no task.toml")
            specs[td] = HonestTaskSpec.from_task_dir(td)
        spec = specs[td]
        if spec.verifier != VERIFIER_PYTEST and not cfg.data.allow_legacy_verifier:
            raise ValueError(
                f'{where}: {td} has verifier={spec.verifier!r}; training needs [metadata.honest] verifier = "pytest" '
                "(hermetic grading, docs/design.md section 6). data.allow_legacy_verifier=true overrides."
            )
        if "feasible" in r and bool(r["feasible"]) != spec.feasible:
            raise ValueError(f"{where}: row {i} says feasible={r['feasible']} but {td}/task.toml says {spec.feasible}")
        for column, truth in (("is_control", spec.is_control), ("kind", spec.kind)):
            if r.get(column) is not None and r[column] != truth:  # a mislabelled control skews the mixture and metrics
                raise ValueError(f"{where}: row {i} says {column}={r[column]!r} but {td}/task.toml says {truth!r}")
        if vid is not None and spec.variant_id is not None and spec.variant_id != vid:
            raise ValueError(f"{where}: row {i} variant_id {vid!r} != task.toml variant_id {spec.variant_id!r}")
        if (
            not spec.feasible
            and cfg.data.require_expected_targets
            and not (spec.expected_blockers and spec.expected_targets)
        ):
            raise ValueError(
                f"{where}: infeasible task {td} needs both expected_blockers and expected_targets in task.toml; "
                "without a target a blind category guess earns the full decline reward"
            )
        counts[spec.kind] = counts.get(spec.kind, 0) + 1
    return counts


def check_split_isolation(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    """The probe split shares no variant, pair or task with training; sharing a source is only a warning."""
    for key in ("variant_id", "pair_id", "task_dir"):
        a = {r[key] for r in train_rows if r.get(key) is not None}
        b = {r[key] for r in eval_rows if r.get(key) is not None}
        if a & b:
            raise ValueError(f"train and validation rows share {len(a & b)} {key}(s), e.g. {min(a & b)!r}")
    a = {r["source"] for r in train_rows if r.get("source")}
    b = {r["source"] for r in eval_rows if r.get("source")}
    if a & b:
        logger.warning(
            "%d validation source task(s) also appear in training: the probe is not source-held-out", len(a & b)
        )


def select_probe_rows(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """A fixed validation subset: pair groups ordered by a hash of their id (not by position or run seed order), taken
    whole until `n` rows are reached, so the same ~100 variants are probed at every step of every run."""
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get("pair_id") or r.get("variant_id") or f"__row{i}"), []).append(r)
    ordered = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}|{k}".encode()).hexdigest())
    for members in groups.values():
        members.sort(key=lambda r: str(r.get("variant_id") or r.get("task_dir")))
    out: list[dict[str, Any]] = []
    for k in ordered:
        if len(out) >= n:
            break
        if len(out) + len(groups[k]) <= n or not out:
            out.extend(groups[k])
    return out[:n] if len(out) > n else out


def _normalise_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arrow needs one type per column: fill missing keys with None and JSON-encode nested non-message values."""
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = []
    for r in rows:
        nr = {}
        for k in keys:
            v = r.get(k)
            if k != "prompt" and isinstance(v, dict):
                v = json.dumps(v)
            nr[k] = v
        out.append(nr)
    return out


def order_fingerprint(rows: list[dict[str, Any]], seed: int, round_size: int | None) -> dict[str, Any]:
    """Identity of the task order: what a resumed run must reproduce for TRL to continue where the run stopped."""
    ids = [str(r.get("variant_id") or r.get("task_dir")) for r in rows]
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return {"n_rows": len(rows), "seed": seed, "round_size": round_size, "sha256": digest}


def check_resume_order(out: Path, fingerprint: dict[str, Any], *, resuming: bool) -> None:
    """Write the fingerprint on a fresh start; on resume refuse if the order differs."""
    path = out / DATA_ORDER_FILE
    if resuming and path.is_file():
        stored = json.loads(path.read_text())
        if stored != fingerprint:
            raise RuntimeError(
                f"resume refused: the task order differs from the original run ({stored} vs {fingerprint}). The data "
                "file, data.max_rows, data.pair_batches, the batch geometry or the seed changed, so TRL would skip "
                "the wrong prompts."
            )
        return
    if resuming:
        logger.warning("resuming but %s is missing; cannot verify the task order", path)
    elif path.is_file() and json.loads(path.read_text()) != fingerprint:
        raise RuntimeError(f"{path} belongs to a different task order; use a new run_name or output_root")
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fingerprint, indent=2) + "\n")


def build_dataset(cfg: TrainRunConfig, path: str | None = None, round_size: int | None = None):
    from datasets import Dataset

    rows = prepare_train_rows(cfg, path, round_size)
    return Dataset.from_list(_normalise_columns(rows))


def epochs_needed(cfg: TrainRunConfig, n_rows: int, round_size: int) -> int:
    """How many passes over the rows `max_steps` generation rounds take (1 when `max_steps` is unset)."""
    max_steps = cfg.grpo.get("max_steps")
    if not max_steps or max_steps < 1 or n_rows < 1:
        return 1
    return max(1, -(-int(max_steps) * round_size // n_rows))  # ceil


def prepare_train_rows(
    cfg: TrainRunConfig,
    path: str | None = None,
    round_size: int | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The training rows in the order TRL will draw them (its own shuffle is off with `pair_batches`).

    TRL replays a shuffle-less dataset in the same order every epoch, so every epoch would put the same pairs in the
    same rounds. When `max_steps` takes more than one pass, the dataset is therefore the concatenation of one
    independently pair-shuffled pass per epoch, each cut to whole rounds so no round straddles two passes. The order
    stays a pure function of the config and the seed, which is what resume needs.
    """
    rows = rows if rows is not None else load_rows(
        path or cfg.data.train_file, cfg.data.max_rows, allow_custom_system_prompt=cfg.data.allow_custom_system_prompt
    )  # fmt: skip
    if round_size and len(rows) < round_size:
        # RepeatSampler drops incomplete chunks: fewer rows than one round means zero training batches.
        raise ValueError(
            f"{len(rows)} training row(s) < {round_size} prompts per generation round; training would see no batches"
        )
    if not cfg.data.pair_batches:
        return list(rows)
    epochs = epochs_needed(cfg, len(rows), round_size) if round_size else 1
    out: list[dict[str, Any]] = []
    for epoch in range(epochs):
        ordered = pair_ordered(rows, cfg.seed + epoch * 1_000_003, round_size)
        if epoch < epochs - 1 and round_size:
            ordered = ordered[: len(ordered) // round_size * round_size]  # a partial tail would shift later rounds
        out.extend(ordered)
    return out
