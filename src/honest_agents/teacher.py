"""Teacher trajectories: an outside agent plays the policy in the real harness, one command at a time.

The episode runs through the evaluation loop (`eval.rollout.run_episode`) unchanged: the same prompt column, tool
schemas, `HonestHarborEnv` (budgets, output caps, network policy, users), grader and trajectory-bank rows as training.
Only the model is replaced: `MailboxClient` stands in for the chat-completions client and, for each model call, writes
what the policy would see to a mailbox directory and waits for the teacher's next action there. Token usage is counted
with the policy's own chat template, so the 16k completion budget binds exactly as it would for the policy.

The teacher never learns which variant it is playing: it gets an opaque token, and the assignment file (token ->
variant) is read only by the episode process and the audit. The grade is never shown to the teacher.

    python -m honest_agents.teacher assign --out data/teacher            # tokens for the selected train variants
    python -m honest_agents.teacher prompt                               # system prompt + tool schemas the policy sees
    python -m honest_agents.teacher start  TOKEN                         # boot; prints the task
    python -m honest_agents.teacher bash   TOKEN 'ls -la' --say '...'    # one command; prints its output
    python -m honest_agents.teacher report TOKEN '{"status": ...}' --say '...'
    python -m honest_agents.teacher abstain TOKEN '{"reason": ...}' --say '...'
    python -m honest_agents.teacher status --out data/teacher            # progress (orchestrator only)

The client commands only touch files; the episode process (`serve`, started by `start`) holds the sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "teacher"
DEFAULT_TRAIN_CONFIG = REPO / "configs" / "grpo_colocate_qwen35.yaml"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
TEACHER_MODEL = "teacher"
REPORT_TOOLS = ("submit_report", "abstain")
ACT_TIMEOUT_SEC = 1800  # a teacher that goes quiet this long ends the episode as an API error (excluded as infra)
POLL_SEC = 0.1
OVER = "The episode is over."


# ── files ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Layout:
    out: Path

    @property
    def assignments(self) -> Path:
        return self.out / "assignments.json"

    @property
    def episodes(self) -> Path:
        return self.out / "episodes.jsonl"

    @property
    def transcripts(self) -> Path:
        return self.out / "transcripts"

    @property
    def bank(self) -> Path:
        return self.out / "bank"

    def mailbox(self, token: str) -> Path:
        return self.out / "mailbox" / token


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)  # atomic: a reader never sees half a file


def _append_locked(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line if line.endswith("\n") else line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def _wait_for(paths: list[Path], timeout: float) -> Path | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for p in paths:
            if p.exists():
                return p
        time.sleep(POLL_SEC)
    return None


def local_task_dir(task_dir: str) -> str:
    """Rows were built on the node (`/root/restraint/data/tasks/...`); the same directory exists under this repo."""
    p = Path(task_dir)
    if p.is_dir():
        return str(p)
    local = REPO / "data" / "tasks" / p.name
    if not local.is_dir():
        raise FileNotFoundError(f"task directory not found: {task_dir} (nor {local})")
    return str(local)


# ── selection ────────────────────────────────────────────────────────────


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_variants(
    pool: list[dict[str, Any]], validation: list[dict[str, Any]], exclude_sources: set[str]
) -> list[dict[str, Any]]:
    """Train-pool variants the oracle validated (and that did not leak), from sources outside `exclude_sources` (the RL
    pool, so SFT and RL do not share sources)."""
    ok = {v["variant_id"] for v in validation if v.get("ok") and not v.get("leak")}
    return [r for r in pool if r["variant_id"] in ok and r["pair_id"] not in exclude_sources]


def make_token(variant_id: str, salt: str) -> str:
    return "t" + hashlib.sha256(f"{salt}:{variant_id}".encode()).hexdigest()[:10]


def assign(args: argparse.Namespace) -> int:
    lay = Layout(Path(args.out))
    data = Path(args.data)
    pool = _rows(data / "train_pool.jsonl")
    exclude = {r["pair_id"] for r in _rows(data / "train_rl.jsonl")} if not args.keep_rl_sources else set()
    rows = select_variants(pool, _rows(data / "validation.jsonl"), exclude)
    if any(r.get("split", "train") != "train" for r in rows):
        raise SystemExit("refusing: a selected row is not from the train split")
    salt = args.salt or uuid.uuid4().hex
    entries = {make_token(r["variant_id"], salt): {**r, "task_dir": local_task_dir(r["task_dir"])} for r in rows}
    if lay.assignments.exists() and not args.force:
        raise SystemExit(f"{lay.assignments} exists (use --force to replace it)")
    lay.out.mkdir(parents=True, exist_ok=True)
    _write_json(lay.assignments, {"salt": salt, "tokens": entries})
    order = list(entries)
    random.Random(0).shuffle(order)
    (lay.out / "tokens.txt").write_text("\n".join(order) + "\n")
    kinds = Counter(f"{r['kind']}/{r['family']}" for r in entries.values())
    print(json.dumps({"n": len(entries), "by_kind": dict(sorted(kinds.items())), "tokens": str(lay.out / "tokens.txt")}))
    return 0


# ── the stand-in model ───────────────────────────────────────────────────


class TokenCounter:
    """Prompt and completion token counts as the policy's chat template renders them (non-thinking mode)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = getattr(tok, "tokenizer", tok)

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], generation_prompt: bool) -> int:
        ids = self.tokenizer.apply_chat_template(
            _template_messages(messages),
            tools=tools,
            add_generation_prompt=generation_prompt,
            tokenize=True,
            enable_thinking=False,
        )
        if isinstance(ids, dict):  # BatchEncoding
            ids = ids["input_ids"]
        return len(ids)


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-completions messages with tool-call arguments as dicts, which the Qwen template expects."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


class MailboxClient:
    """`client.chat.completions.create(**kwargs)` that asks the teacher instead of a model."""

    def __init__(self, mailbox: Path, counter: TokenCounter | None, act_timeout: float = ACT_TIMEOUT_SEC):
        self.mailbox = mailbox
        self.counter = counter
        self.act_timeout = act_timeout
        self.seq = 0
        self.shown = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs: Any) -> Any:
        messages: list[dict[str, Any]] = kwargs["messages"]
        tools = kwargs.get("tools") or []
        new = messages[self.shown :]
        self.shown = len(messages)
        _write_json(
            self.mailbox / f"obs-{self.seq}.json",
            {"seq": self.seq, "messages": new, "max_tokens": kwargs.get("max_tokens")},
        )
        act_path = self.mailbox / f"act-{self.seq}.json"
        found = await asyncio.to_thread(_wait_for, [act_path], self.act_timeout)
        if found is None:
            raise TimeoutError(f"no teacher action within {self.act_timeout:.0f}s")
        act = json.loads(act_path.read_text())
        self.seq += 1
        content = act.get("content") or ""
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        calls = []
        if act.get("tool"):
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            arguments = json.dumps(act.get("arguments") or {})
            calls.append(SimpleNamespace(id=call_id, function=SimpleNamespace(name=act["tool"], arguments=arguments)))
            assistant["tool_calls"] = [
                {"id": call_id, "type": "function", "function": {"name": act["tool"], "arguments": arguments}}
            ]
        usage = None
        if self.counter is not None:
            prompt = self.counter.count(messages, tools, generation_prompt=True)
            full = self.counter.count([*messages, assistant], tools, generation_prompt=False)
            usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=max(0, full - prompt))
            max_tokens = kwargs.get("max_tokens")
            if max_tokens is not None and usage.completion_tokens > max_tokens:
                # A policy's turn would have been cut here; the teacher's turn is refused instead of truncated, so a
                # kept trajectory never contains a turn the policy could not have produced.
                raise ValueError(f"turn of {usage.completion_tokens} tokens exceeds the {max_tokens}-token cap")
        message = SimpleNamespace(content=content, tool_calls=calls or None, reasoning_content=None)
        finish = "tool_calls" if calls else "stop"
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


# ── the episode process ──────────────────────────────────────────────────


def _eval_args(out: Path, train_config: Path) -> argparse.Namespace:
    from honest_agents.eval import run_eval

    return run_eval.build_parser().parse_args(
        ["--split", str(out / "assignments.json"), "--out", str(out), "--model", TEACHER_MODEL,
         "--train-config", str(train_config), "--no-thinking"]  # fmt: skip
    )


def serve(args: argparse.Namespace) -> int:
    from honest_agents import bank
    from honest_agents.eval import run_eval
    from honest_agents.eval.records import record_line
    from honest_agents.eval.rollout import Task, run_episode

    lay = Layout(Path(args.out))
    box = lay.mailbox(args.token)
    entry = json.loads(lay.assignments.read_text())["tokens"][args.token]
    eval_args = _eval_args(lay.out, Path(args.train_config))
    rollout_cfg, reward_cfg, budget = run_eval.resolve_configs(eval_args)
    env = run_eval._env_factory(eval_args, reward_cfg, budget)()
    task = Task(task_dir=entry["task_dir"], row=entry)
    counter = None if args.no_token_count else TokenCounter(args.model_id)
    client = MailboxClient(box, counter)

    async def go():
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            return await run_episode(client, rollout_cfg, env, task, executor=ex)

    record, transcript = asyncio.run(go())
    record.extra["teacher_token"] = args.token
    _append_locked(lay.episodes, record_line(record, reward_cfg))
    lay.transcripts.mkdir(parents=True, exist_ok=True)
    (lay.transcripts / f"{args.token}.json").write_text(json.dumps(transcript, indent=1, default=str))
    if record.infra_error is None:
        row = run_eval.bank_row(record, transcript, task, reward_cfg)
        row.extra["teacher_token"] = args.token
        bank.append_rows([row], lay.bank)
    _write_json(box / "done.json", {"infra_error": record.infra_error})  # no grade: the teacher never sees it
    return 0


# ── client commands (stdlib only, no sandbox) ────────────────────────────


def _latest_obs(box: Path) -> int | None:
    seqs = [int(p.stem.split("-")[1]) for p in box.glob("obs-*.json")]
    return max(seqs) if seqs else None


def _render(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue  # the teacher was given it once; repeating it every turn only costs context
        if role == "user":
            parts.append(f"[task]\n{m.get('content', '')}")
        elif role == "tool":
            parts.append(str(m.get("content", "")))
    return "\n\n".join(parts)


def _step(lay: Layout, token: str, act: dict[str, Any], timeout: float) -> str:
    box = lay.mailbox(token)
    if (box / "done.json").exists():
        return OVER
    seq = _latest_obs(box)
    if seq is None:
        return "error: episode not started (run `start` first)"
    if (box / f"act-{seq}.json").exists():
        return "error: an action for this turn is already pending; wait for its output"
    _write_json(box / f"act-{seq}.json", act)
    found = _wait_for([box / f"obs-{seq + 1}.json", box / "done.json"], timeout)
    if found is None:
        return f"error: no response within {timeout:.0f}s (the episode process may have died; see {box / 'serve.log'})"
    if found.name == "done.json":
        done = json.loads(found.read_text())
        if done.get("infra_error"):
            return f"{OVER} (infrastructure error: {done['infra_error']})"
        return OVER
    return _render(json.loads(found.read_text())["messages"]) or "(no output)"


def start(args: argparse.Namespace) -> int:
    lay = Layout(Path(args.out))
    if args.token not in json.loads(lay.assignments.read_text())["tokens"]:
        print(f"error: unknown token {args.token}")
        return 2
    box = lay.mailbox(args.token)
    if box.exists():
        print(f"error: {args.token} was already started; every task gets exactly one attempt")
        return 2
    box.mkdir(parents=True)
    log = open(box / "serve.log", "w")
    cmd = [sys.executable, "-m", "honest_agents.teacher", "serve", args.token, "--out", str(lay.out),
           "--train-config", str(args.train_config)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    (box / "serve.pid").write_text(str(proc.pid))
    found = _wait_for([box / "obs-0.json", box / "done.json"], args.timeout)
    if found is None or found.name == "done.json":
        print(f"error: the episode did not start (see {box / 'serve.log'})")
        return 1
    print(_render(json.loads(found.read_text())["messages"]))
    return 0


def _say(args: argparse.Namespace) -> str:
    return (args.say or "").strip()


def bash_cmd(args: argparse.Namespace) -> int:
    act = {"content": _say(args), "tool": "bash", "arguments": {"command": args.command}}
    print(_step(Layout(Path(args.out)), args.token, act, args.timeout))
    return 0


def report_cmd(args: argparse.Namespace) -> int:
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as e:
        print(f"error: arguments are not JSON: {e}")
        return 2
    if not isinstance(arguments, dict):
        print("error: arguments must be a JSON object")
        return 2
    act = {"content": _say(args), "tool": args.tool, "arguments": arguments}
    print(_step(Layout(Path(args.out)), args.token, act, args.timeout))
    return 0


def end_cmd(args: argparse.Namespace) -> int:
    """Stop an episode without a report (a broken teacher session): graded as a silent stop and never audited, so
    the SFT stage drops it."""
    print(_step(Layout(Path(args.out)), args.token, {"content": "", "tool": None}, args.timeout))
    return 0


def prompt_cmd(args: argparse.Namespace) -> int:
    from honest_agents.harness import SYSTEM_PROMPT, tool_schemas

    print("SYSTEM PROMPT\n=============\n" + SYSTEM_PROMPT + "\n\nTOOLS\n=====")
    print(json.dumps(tool_schemas(), indent=1))
    return 0


def status(args: argparse.Namespace) -> int:
    from honest_agents.eval.records import iter_records
    from honest_agents.reward import RewardConfig

    lay = Layout(Path(args.out))
    tokens = json.loads(lay.assignments.read_text())["tokens"]
    started = {p.name for p in (lay.out / "mailbox").glob("t*")} if (lay.out / "mailbox").exists() else set()
    recs = list(iter_records(lay.episodes)) if lay.episodes.exists() else []
    cfg = RewardConfig()
    cases = Counter(r.infra_error and "infra_error" or r.score(cfg).case for r in recs)
    print(json.dumps({"assigned": len(tokens), "started": len(started), "finished": len(recs), "cases": dict(cases)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m honest_agents.teacher", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", default=str(DEFAULT_OUT))
        sp.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for the sandbox")

    a = sub.add_parser("assign", help="select train variants and give each an opaque token")
    a.add_argument("--out", default=str(DEFAULT_OUT))
    a.add_argument("--data", default=str(REPO / "data"))
    a.add_argument("--salt", default=None)
    a.add_argument("--keep-rl-sources", action="store_true", help="also use sources of the RL pool")
    a.add_argument("--force", action="store_true")
    a.set_defaults(fn=assign)

    sp = sub.add_parser("prompt", help="print the system prompt and tool schemas the policy sees")
    sp.set_defaults(fn=prompt_cmd)

    sp = sub.add_parser("start", help="boot a task's sandbox and print the task")
    sp.add_argument("token")
    sp.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    common(sp)
    sp.set_defaults(fn=start)

    sp = sub.add_parser("bash", help="run one command; prints its output as the policy would see it")
    sp.add_argument("token")
    sp.add_argument("command")
    sp.add_argument("--say", default="", help="the assistant text that accompanies this call (short)")
    common(sp)
    sp.set_defaults(fn=bash_cmd)

    for tool, aliases in (("submit_report", ["report"]), ("abstain", [])):
        sp = sub.add_parser(tool, aliases=aliases, help=f"end the episode with `{tool}` (JSON object of its arguments)")
        sp.add_argument("token")
        sp.add_argument("arguments")
        sp.add_argument("--say", default="")
        common(sp)
        sp.set_defaults(fn=report_cmd, tool=tool)

    sp = sub.add_parser("end", help="stop a broken episode without a report (orchestrator only)")
    sp.add_argument("token")
    common(sp)
    sp.set_defaults(fn=end_cmd)

    sp = sub.add_parser("status", help="progress counts (orchestrator only)")
    sp.add_argument("--out", default=str(DEFAULT_OUT))
    sp.set_defaults(fn=status)

    sp = sub.add_parser("serve", help=argparse.SUPPRESS)
    sp.add_argument("token")
    sp.add_argument("--out", default=str(DEFAULT_OUT))
    sp.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    sp.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    sp.add_argument("--no-token-count", action="store_true")
    sp.set_defaults(fn=serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
