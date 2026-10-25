"""Sweep S11: the infrastructure guards of the training log's incident table, each exercised once (DEF-23).

docs/integrity-gate.md S11: "the guards in the training log's incident table still run: egress preflight, masked-share
stop, sanity run, `balance_floor`. Each guard is exercised once in the dry run", against the fake sandbox; only the
egress preflight needs a real container. Each function returns its problems (empty: the guard works) and drives the
guard's real code, with only the outside world replaced:

* **egress preflight** (iteration 15 G3: harbor's egress probe failed once under load and the cached False refused every
  later boot). `env.preflight_network_policy` with a fake probe that fails, then passes: it must clear the cached False
  and retry; one that always fails must raise. `eval.run_eval.evaluate` and `train.run.train` must call it. With
  `real_egress`, the real probe runs once against Docker (one probe container, `egress_preflight_real`).
* **masked-share stop** (the same incident: masked share 1.0 for 15 steps, nothing stopped it). Real `HonestHarborEnv`
  episodes whose sandbox factory refuses the boot as Docker did (`network_mode='no-network' is not supported`) go
  through the real reward function (`trl_rewards.honest_reward`): every reward must be masked as a boot failure, the
  logged masked share 1.0, and `callbacks.StopRules` (the shipped `StopRuleConfig`) must fire `masked_share_collapse`
  on the second such step and not on the first.
* **sanity run** (a merged checkpoint whose vLLM output was garbled while HF's forward looked fine). The rule "every
  merged model gets a 12-task vLLM sanity run before a full evaluation" (`sanity_problems`) on a chain file: it must
  flag a chain step that evaluates a merged model with no `--limit 12` run of it first, pass one that has it, and, for
  `chains`, the job files about to run must pass it.
* **balance_floor** (`node_watch`): a `Watcher` reading a local chain state directory through the real
  `read_command` / `stop_request_command` (bash instead of ssh) and a balance that falls below the floor must write the
  chain's `STOP` file once; the real `node_chain.Runner` must then start no step and write `ALL_DONE`, after which the
  watcher pulls and stops the instance.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from honest_agents.gate.cache import SRC

# The error harbor's Docker environment raised on every boot after the probe's False was cached (training log).
REFUSED_BOOT = "network_mode='no-network' is not supported"
SANITY_LIMIT = 12
_MODEL = re.compile(r"\bmodel\.name=(\S+)")
_TASK_TOML = """schema_version = "1.4"

[task]
name = "gate/s11-refused-boot"

[metadata.honest]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
build_timeout_sec = 60.0
"""


# ── egress preflight ──────────────────────────────────────────────────────────


class _FakeProbe:
    def __init__(self, results: Sequence[bool]):
        self.results, self.calls, self.cleared = list(results), 0, 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.results.pop(0)

    def cache_clear(self) -> None:
        self.cleared += 1


@contextlib.contextmanager
def _probe(results: Sequence[bool]) -> Iterator[_FakeProbe]:
    from harbor.environments.docker.docker import DockerEnvironment

    original = DockerEnvironment.__dict__["_egress_control_kernel_support"]
    fake = _FakeProbe(results)
    DockerEnvironment._egress_control_kernel_support = fake  # type: ignore[method-assign,assignment]
    try:
        yield fake
    finally:
        DockerEnvironment._egress_control_kernel_support = original  # type: ignore[method-assign]


def calls_in(path: Path, function: str, callee: str) -> bool:
    """`function` (a top-level def of `path`) calls `callee` by name."""
    for fn in ast.parse(path.read_text()).body:
        if isinstance(fn, ast.FunctionDef) and fn.name == function:
            return any(
                isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == callee
                for n in ast.walk(fn)
            )
    return False


PREFLIGHT_CALLERS = ((SRC / "eval" / "run_eval.py", "evaluate"), (SRC / "train" / "run.py", "train"))


def egress_preflight_problems() -> list[str]:
    from honest_agents.env import preflight_network_policy

    out = []
    with _probe([False, True]) as p:
        try:
            ok = preflight_network_policy(sleep=lambda s: None) is True and p.cleared == 1 and p.calls == 2
        except RuntimeError:
            ok = False
        if not ok:
            out.append(f"a probe that fails once is not cleared and retried (calls {p.calls}, cleared {p.cleared})")
    with _probe([False, False, False]) as p:
        try:
            preflight_network_policy(sleep=lambda s: None)
            out.append("a probe that always fails does not stop the run")
        except RuntimeError:
            if p.cleared != 3:
                out.append(f"the last failed probe stays cached (cleared {p.cleared} of 3)")
    for path, fn in PREFLIGHT_CALLERS:
        if not calls_in(path, fn, "preflight_network_policy"):
            out.append(f"{path.relative_to(SRC)}:{fn} no longer runs the preflight before its boots")
    return out


def egress_preflight_real() -> tuple[bool, str]:
    """The real harbor probe, once, against this machine's Docker (one probe container): True when Docker can enforce
    the network policies every eval and training boot asks for."""
    from honest_agents.env import preflight_network_policy

    try:
        checked = preflight_network_policy("docker", True)
    except RuntimeError as e:
        return False, str(e).split(".")[0]
    return checked, "the probe container passed" if checked else "skipped (harbor has no egress probe)"


# ── masked-share stop ─────────────────────────────────────────────────────────


def _refused_task(root: Path) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(_TASK_TOML)
    (task / "instruction.md").write_text("Write `hi` to hello.txt.\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")
    return task


def masked_share_problems(n: int = 4) -> list[str]:
    from harbor.environments.factory import EnvironmentFactory

    from honest_agents.callbacks import RULE_MASKED_COLLAPSE, StopRuleConfig, StopRules
    from honest_agents.env import HonestHarborEnv
    from honest_agents.reward import MASK_BOOT_FAILURE
    from honest_agents.trl_rewards import METRIC_MASKED_SHARE, honest_reward

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(REFUSED_BOOT)

    out: list[str] = []
    logs: dict[str, float] = {}
    original = EnvironmentFactory.__dict__["create_environment_from_config"]
    envs: list[Any] = []
    quiet = logging.getLogger("honest_agents.env")
    level = quiet.level
    with tempfile.TemporaryDirectory(prefix="gate_s11_") as d:
        task = _refused_task(Path(d))
        EnvironmentFactory.create_environment_from_config = staticmethod(refuse)  # type: ignore[method-assign]
        quiet.setLevel(logging.CRITICAL)  # n expected boot tracebacks
        try:
            for _ in range(n):
                env = HonestHarborEnv()
                envs.append(env)
                env.reset(task_dir=str(task))
            rewards = honest_reward(environments=envs, log_metric=logs.__setitem__)
            outcomes = [env._graded()[0] for env in envs]
        finally:
            EnvironmentFactory.create_environment_from_config = original  # type: ignore[method-assign]
            quiet.setLevel(level)
            for env in envs:
                env._loop.call_soon_threadsafe(env._loop.stop)
    if any(r is not None for r in rewards):
        out.append(f"a refused boot is not masked: rewards {rewards}")
    if any(o.masked_reason != MASK_BOOT_FAILURE or REFUSED_BOOT not in str(o.grading_error) for o in outcomes):
        out.append(f"a refused boot is not recorded as a boot failure: {[o.masked_reason for o in outcomes]}")
    if logs.get(METRIC_MASKED_SHARE) != 1.0:
        out.append(f"{METRIC_MASKED_SHARE} = {logs.get(METRIC_MASKED_SHARE)} on an all-refused batch, not 1.0")
    rules = StopRules(StopRuleConfig())
    first = [t.rule for t in rules.observe(1, dict(logs))]
    second = [t.rule for t in rules.observe(2, dict(logs))]
    if RULE_MASKED_COLLAPSE in first or RULE_MASKED_COLLAPSE not in second or rules.config.action != "stop":
        out.append(f"the masked-share stop does not fire on the 2nd all-masked step (step 1 {first}, step 2 {second})")
    return out


# ── sanity run ────────────────────────────────────────────────────────────────


def _evals(command: str) -> list[list[str]]:
    """The `eval_heldout.sh` calls of a `serve_and_run.sh CONFIG [overrides] -- CMD -- CMD ...` step, in order."""
    out = []
    for part in command.split(" -- ")[1:]:
        try:
            words = shlex.split(part)
        except ValueError:
            continue
        if words and words[0].endswith("eval_heldout.sh"):
            out.append(words)
    return out


def _limit(words: Sequence[str]) -> int | None:
    for i, w in enumerate(words):
        if w == "--limit" and i + 1 < len(words) and words[i + 1].isdigit():
            return int(words[i + 1])
        if w.startswith("--limit=") and w.split("=", 1)[1].isdigit():
            return int(w.split("=", 1)[1])
    return None


def sanity_problems(chain: Any) -> list[str]:
    """The sanity-run rule on a `node_chain.Chain`: a step that serves a checkpoint we wrote (`model.name=outputs/...`)
    and evaluates it runs a `--limit <= 12` evaluation of it first, in that step or in a step it depends on."""
    sane: dict[str, set[str]] = {}  # step -> models with a sanity run in it or before it
    out = []
    for step in chain.steps:
        before = set().union(*(sane.get(a, set()) for a in step.after))
        m = _MODEL.search(step.command.split(" -- ")[0]) if "serve_and_run.sh" in step.command else None
        model = m.group(1) if m else None
        evals = _evals(step.command)
        if model is not None and model.startswith("outputs/") and evals:
            first = _limit(evals[0])
            if model not in before and (first is None or first > SANITY_LIMIT):
                out.append(
                    f"{chain.name}/{step.name}: {model} is evaluated with no {SANITY_LIMIT}-task sanity run first"
                )
            if first is not None and first <= SANITY_LIMIT:
                before.add(model)
        sane[step.name] = before
    return out


_SERVE = "scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml model.name=outputs/x/m_merged"
_FULL = "scripts/eval_heldout.sh data/decision/surfaces_val/tasks runs/eval/m_surface --n-samples 4"
_SANITY = "scripts/eval_heldout.sh data/val.jsonl runs/eval/m_sanity --limit 12 --n-samples 1"


def sanity_rule_problems(chains: Sequence[Path] = ()) -> list[str]:
    from honest_agents.node_chain import load_chain, parse_chain

    def chain(*commands: str) -> Any:
        steps = [{"name": f"s{i}", "command": c, "done": f"m/s{i}"} for i, c in enumerate(commands)]
        return parse_chain({"name": "s11", "steps": steps})

    out = []
    if not sanity_problems(chain(f"{_SERVE} -- {_FULL}")):
        out.append("the sanity-run rule does not flag a merged model evaluated without one")
    if sanity_problems(chain(f"{_SERVE} -- {_SANITY} -- {_FULL}")) or sanity_problems(
        chain(f"{_SERVE} -- {_SANITY}", f"{_SERVE} -- {_FULL}")
    ):
        out.append("the sanity-run rule flags a chain that has one")
    for path in chains:
        out += sanity_problems(load_chain(path))
    return out


# ── balance floor ─────────────────────────────────────────────────────────────


def balance_floor_problems(floor: float = 8.0) -> list[str]:
    from honest_agents.node import Node
    from honest_agents.node_chain import ALL_DONE_FILE, SKIPPED, STOP_FILE, Runner, parse_chain
    from honest_agents.node_watch import Effects, WatchConfig, Watcher, read_command, stop_request_command

    out = []
    with tempfile.TemporaryDirectory(prefix="gate_s11_") as d:
        root = Path(d)
        chain = parse_chain({"name": "s11", "state_dir": "state", "balance_floor": floor,
                             "steps": [{"name": "a", "command": "touch m/a", "done": "m/a"},
                                       {"name": "b", "command": "touch m/b", "done": "m/b"}]})  # fmt: skip
        node = Node("localhost", dest=str(root))
        credits = [floor + 12.0, floor - 1.0]
        log: list[str] = []

        def bash(command: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30, check=False)

        def balance() -> float:
            return credits.pop(0) if len(credits) > 1 else credits[0]

        def request_stop() -> bool:
            log.append("request_stop")
            return bash(stop_request_command(node, chain.state_dir)).returncode == 0

        fx = Effects(lambda: bash(read_command(node, chain.state_dir)).stdout, request_stop, balance,
                     lambda: log.append("pull") or [], lambda: log.append("stop") or True, sleep=lambda s: None)  # fmt: skip
        watcher = Watcher(WatchConfig(chain, node, 1, balance_floor=floor, balance_every=1), fx)
        events = watcher.poll_once() + watcher.poll_once() + watcher.poll_once()
        if log.count("request_stop") != 1 or not (root / "state" / STOP_FILE).is_file():
            out.append(f"a balance below the floor does not write the chain's STOP file once (log {log})")
        (root / "m").mkdir()
        runner = Runner(chain, root, poll=0.0, cleanup=lambda since: [], sleep=lambda s: None,
                        sweep=lambda token: [], find=lambda name: [])  # fmt: skip
        verdict = runner.run()
        started = [n for n, s in runner.status.items() if s["status"] != SKIPPED]
        if verdict != "stopped" or started or any((root / "m" / n).exists() for n in ("a", "b")):
            out.append(f"the chain runner starts steps after the STOP file (verdict {verdict}, ran {started})")
        if not (root / "state" / ALL_DONE_FILE).is_file():
            out.append("the stopped chain writes no ALL_DONE")
        events += watcher.poll_once()
        if log[-2:] != ["pull", "stop"] or not any(e.startswith("BALANCE") for e in events):
            out.append(f"after ALL_DONE the watcher does not pull and stop the instance (log {log})")
    return out


GUARDS: dict[str, Callable[[], list[str]]] = {
    "egress preflight": egress_preflight_problems,
    "masked-share stop": masked_share_problems,
    "sanity run": sanity_rule_problems,
    "balance_floor": balance_floor_problems,
}


def exercise(chains: Sequence[Path] = (), real_egress: bool = True) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Every guard once: {guard: problems} and {guard: wall seconds}; with `real_egress`, also the real probe."""
    problems: dict[str, list[str]] = {}
    wall: dict[str, float] = {}
    for name, fn in GUARDS.items():
        t0 = time.monotonic()
        try:
            problems[name] = sanity_rule_problems(chains) if fn is sanity_rule_problems else fn()
        except Exception as e:  # noqa: BLE001 - a guard that crashes is a guard that does not run
            problems[name] = [f"{type(e).__name__}: {e}"]
        wall[name] = round(time.monotonic() - t0, 2)
    if real_egress:
        t0 = time.monotonic()
        ok, detail = egress_preflight_real()
        if not ok:
            problems["egress preflight"].append(f"real probe: {detail}")
        wall["egress probe (real)"] = round(time.monotonic() - t0, 2)
    return problems, wall
