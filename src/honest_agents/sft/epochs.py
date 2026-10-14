"""Epoch bookkeeping for an SFT run, for `scripts/merge_epochs.sh` and `scripts/select_epoch.sh` (logic lives here, the
scripts only orchestrate).

    python -m honest_agents.sft.epochs list RUN_DIR [--expect 3]
    python -m honest_agents.sft.epochs decide SELECTION_JSON OUT_DIR --fallback 3 --status N --prefix sft_v2

`list` prints `<epoch><TAB><checkpoint dir>` in epoch order. The epoch of `checkpoint-N` is read from that checkpoint's own
`trainer_state.json` (`epoch` at save time, with `save_strategy: epoch` an integer), never from the position in a sorted
listing: a run that stopped early, or an interrupted save, would shift a positional count. A checkpoint without an
adapter or without a trainer state, two checkpoints at the same epoch, an epoch that is not an integer or a count other
than `--expect` is refused.

`decide` turns the exit status of `eval.select_checkpoint` (0: an epoch passed the gate, 1: none did) and its
`selection.json` into the epoch to use: the selected one, else the pre-registered fallback (docs/iteration-14-training-plan.md
section 4: the select-split gate is underpowered at ~270 feasible episodes, so when no epoch passes it is epoch 3). Any
other status (the runs cannot be compared, a split mismatch) is an error, not a fallback. It writes
`OUT_DIR/selected_epoch.txt` (`<k> selected|fallback`) and the symlink `OUT_DIR/<prefix>_selected_merged -> <prefix>_e<k>_merged`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

EPOCH_TOLERANCE = 0.05  # `epoch` in a trainer state is step / steps-per-epoch: integral at an epoch-end save up to rounding


def checkpoint_epochs(run_dir: Path, expect: int | None = None) -> list[tuple[int, Path]]:
    """(epoch, checkpoint dir) in epoch order, from each checkpoint's trainer state."""
    found: dict[int, Path] = {}
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)]
    for ckpt in sorted(ckpts, key=lambda p: int(p.name.split("-")[1])):
        if not (ckpt / "adapter_config.json").is_file():
            raise SystemExit(f"{ckpt}: no adapter_config.json (not an adapter checkpoint)")
        state_file = ckpt / "trainer_state.json"
        if not state_file.is_file():
            raise SystemExit(f"{ckpt}: no trainer_state.json, cannot tell which epoch it is")
        state = json.loads(state_file.read_text())
        epoch = state.get("epoch")
        if not isinstance(epoch, (int, float)) or abs(epoch - round(epoch)) > EPOCH_TOLERANCE or round(epoch) < 1:
            raise SystemExit(f"{ckpt}: trainer_state epoch {epoch!r} is not an epoch-end save (save_strategy: epoch)")
        k = round(epoch)
        if k in found:
            raise SystemExit(f"{ckpt} and {found[k]} are both epoch {k}")
        found[k] = ckpt
    if expect is not None and sorted(found) != list(range(1, expect + 1)):
        raise SystemExit(f"{run_dir}: expected epoch checkpoints 1..{expect}, found epochs {sorted(found)}")
    if not found:
        raise SystemExit(f"{run_dir}: no checkpoint-* directories")
    return [(k, found[k]) for k in sorted(found)]


def decide(selection: Path, out_dir: Path, *, fallback: int, status: int, prefix: str) -> tuple[int, str]:
    """The epoch to use and whether it was `selected` or the `fallback`; writes selected_epoch.txt and the symlink."""
    if status not in (0, 1):
        raise SystemExit(f"select_checkpoint exited with status {status}: the comparison itself failed, no fallback applied")
    # select_checkpoint also exits 1 when it refuses to compare (SystemExit with a message), before writing anything: the
    # fallback applies only if it got as far as writing a selection that picked nobody.
    if not selection.is_file():
        raise SystemExit(f"{selection} missing: select_checkpoint refused the comparison (status {status}), no fallback applied")
    chosen = json.loads(selection.read_text()).get("selected")
    if (status == 0) != (chosen is not None):
        raise SystemExit(f"{selection}: status {status} contradicts selected={chosen!r}")
    if chosen is not None:
        m = re.search(r"(\d+)$", chosen)
        if not m:
            raise SystemExit(f"selected run {chosen!r} does not end in an epoch number")
        epoch, how = int(m.group(1)), "selected"
    else:
        epoch, how = fallback, "fallback"
    target = f"{prefix}_e{epoch}_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_epoch.txt").write_text(f"{epoch} {how}\n")
    link = out_dir / f"{prefix}_selected_merged"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)
    return epoch, how


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.epochs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="epoch<TAB>checkpoint dir, in epoch order")
    ls.add_argument("run_dir", type=Path)
    ls.add_argument("--expect", type=int, default=None, help="refuse unless the epochs are exactly 1..N")
    de = sub.add_parser("decide", help="selected epoch or the pre-registered fallback")
    de.add_argument("selection", type=Path, help="selection.json written by eval.select_checkpoint --out")
    de.add_argument("out_dir", type=Path)
    de.add_argument("--fallback", type=int, required=True)
    de.add_argument("--status", type=int, required=True, help="exit status of eval.select_checkpoint")
    de.add_argument("--prefix", default="sft_v2", help="merged directories are <prefix>_e<k>_merged")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for k, path in checkpoint_epochs(args.run_dir, args.expect):
            print(f"{k}\t{path}")
        return 0
    epoch, how = decide(args.selection, args.out_dir, fallback=args.fallback, status=args.status, prefix=args.prefix)
    print(f"epoch {epoch} ({how})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
