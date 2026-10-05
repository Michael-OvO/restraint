"""Report-file protocol: honesty scoring under any Harbor harness.

A third-party harness (`terminus-2`, `mini-swe-agent`, `opencode`, ...) has no `submit_report` / `abstain` tool. The
agent is instead told, by a suffix on the task instruction, to write its report as JSON to `harness.REPORT_FILE`
inside the sandbox, in the same schema as the tools. Harbor collects the file as a trial artifact and
`honest_agents.eval.harbor_results` scores it with the same `compute_reward`:

    python -m honest_agents.eval.report_file instructions --out runs/report_file_instructions.md
    harbor run -p TASKS -a terminus-2 -m hosted_vllm/policy --ak api_base=... \\
        --extra-instruction-path runs/report_file_instructions.md --artifact /logs/agent/honest_report.json
    python -m honest_agents.eval.harbor_results JOB_DIR

(`scripts/eval_harness.sh` does all three.) Both flags are harbor 0.23.0's own: `--extra-instruction-path` appends the
file to every task's instruction (`harbor.models.task.task.Task`), `--artifact` downloads an environment path into
`<trial>/artifacts/<path>` (`harbor.trial.artifact_handler`). `/logs/agent` is also the agent log directory Harbor
mounts or downloads to `<trial>/agent/`, so the file is found in either place.

What differs from our own harness, and is recorded on the episode record:

* Harbor's verifier runs the task's `tests/test.sh`, not the hermetic grader: no tamper check, no path-violation
  check, no evidence replay, and no blocker verification (`blocker_check` stays "unknown", so an external blocker on
  a feasible task counts as false blame).
* The number of commands comes from the harness's ATIF trajectory (`agent/trajectory.json`) when it writes one. The
  command that writes the report file is not work, so one is subtracted. Without a trajectory the count is unknown:
  it is recorded as 1 and `extra.n_actions_known` is false, and the lazy / zero-command / efficiency numbers of that
  run must not be quoted.

`link-split` turns a JSONL split into a directory of symlinks, which is what `harbor run -p` takes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from honest_agents import harness
from honest_agents.report import ReportParseError, SelfReport

TRAJECTORY_FILE = "trajectory.json"  # ATIF trajectory most Harbor agents write to their logs dir
_MAX_BYTES = 1 << 20


def instructions_text() -> str:
    """The instruction suffix that tells an agent to write the report file."""
    return harness.REPORT_FILE_INSTRUCTIONS


def report_file_candidates(trial_dir: str | Path) -> list[Path]:
    """Host paths where the sandbox's `harness.REPORT_FILE` can land in a Harbor trial directory, best first."""
    trial = Path(trial_dir)
    sandbox = PurePosixPath(harness.REPORT_FILE)
    out = [trial / "artifacts" / Path(*sandbox.parts[1:])]  # --artifact: mirrored under artifacts/
    if sandbox.parent == PurePosixPath("/logs/agent"):
        out.append(trial / "agent" / sandbox.name)  # the agent logs dir, mounted or downloaded by Harbor
    return out


def find_report_file(trial_dir: str | Path) -> Path | None:
    return next((p for p in report_file_candidates(trial_dir) if p.is_file()), None)


def read_trial_report(
    trial_dir: str | Path,
) -> tuple[SelfReport | None, str | None, Path | None]:
    """(report, parse error, path). All None when the agent wrote no report file."""
    path = find_report_file(trial_dir)
    if path is None:
        return None, None, None
    try:
        text = path.read_bytes()[:_MAX_BYTES].decode("utf-8", errors="replace")
        return harness.read_report_file_text(text), None, path
    except ReportParseError as e:
        return None, str(e), path
    except OSError as e:
        return None, f"report file unreadable: {e}", path


def trajectory_tool_calls(trial_dir: str | Path) -> int | None:
    """Tool calls the agent made, from its ATIF trajectory; None when the harness wrote none we can read."""
    path = Path(trial_dir) / "agent" / TRAJECTORY_FILE
    if not path.is_file():
        return None
    try:
        steps = json.loads(path.read_text()).get("steps")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(steps, list):
        return None
    return sum(len(s.get("tool_calls") or []) for s in steps if isinstance(s, dict) and s.get("source") == "agent")


def actions_before_report(n_tool_calls: int | None, wrote_report: bool) -> tuple[int, bool]:
    """(n_actions, known). The command that writes the report file is not counted as work."""
    if n_tool_calls is None:
        return 1, False
    return max(0, n_tool_calls - 1) if wrote_report else n_tool_calls, True


def link_split(split: str | Path, out_dir: str | Path) -> list[Path]:
    """Materialise a split as a directory of symlinks to its task dirs (what `harbor run -p` scans)."""
    from honest_agents.eval.rollout import load_split

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    links: list[Path] = []
    seen: dict[str, str] = {}
    for task in load_split(split):
        name = Path(task.task_dir).name
        if seen.setdefault(name, task.task_dir) != task.task_dir:
            raise ValueError(f"two tasks of the split share the directory name {name!r}; Harbor needs unique names")
        link = out / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == Path(task.task_dir).resolve():
                links.append(link)
                continue
            raise FileExistsError(f"{link} exists and is not a link to {task.task_dir}")
        link.symlink_to(Path(task.task_dir).resolve(), target_is_directory=True)
        links.append(link)
    return links


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("instructions", help="write the instruction suffix for --extra-instruction-path")
    ins.add_argument("--out", help="file to write (default: stdout)")
    ln = sub.add_parser("link-split", help="directory of symlinks to the task dirs of a JSONL split")
    ln.add_argument("split")
    ln.add_argument("out_dir")
    sub.add_parser("path", help="print the sandbox path of the report file (for --artifact)")
    args = p.parse_args(argv)
    result: Any
    if args.cmd == "instructions":
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(instructions_text())
        else:
            print(instructions_text())
    elif args.cmd == "link-split":
        result = link_split(args.split, args.out_dir)
        print(f"{len(result)} tasks linked under {args.out_dir}")
    else:
        print(harness.REPORT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
