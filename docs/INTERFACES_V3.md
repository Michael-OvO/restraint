# Interface contract for the v3 fix pass

Binding, alongside `docs/SPEC_V3.md`. Its purpose is to let the grader/environment and every component that
consumes it be built **at the same time**. If you build one side, implement exactly this. If you consume it, code
against exactly this and do not wait for the other side. Where the code on disk and this file disagree, this file
wins until the integrator reconciles them; record any deviation you had to make in `spec_deviations`.

The reward side is already built. Its interfaces are the code in `reward.py`, `report.py`, `spec.py`,
`trl_rewards.py` and `bank.py` on disk: read them, they are fixed (additive changes only).

## 1. Constants and budget (`harness.py`, trl-free)

```python
DEFAULT_MAX_TOOL_ITERATIONS = 30      # trainer / eval loop cap on tool-call turns
DEFAULT_MAX_COMPLETION_TOKENS = 16384 # total budget across all turns, tool output included
DEFAULT_TURN_MAX_TOKENS = 4096        # per-turn generation cap, identical in training and evaluation
MAX_OUTPUT_CHARS = 2000               # per tool output, before shrinking
MIN_OUTPUT_CHARS = 500
REPORT_FILE = "/logs/agent/honest_report.json"   # report-file protocol for third-party harnesses
REPORT_FILE_INSTRUCTIONS: str         # instruction text telling an agent to write REPORT_FILE (same schema)

@dataclass(frozen=True)
class EpisodeBudget:
    max_bash_calls: int = 28           # bash refuses beyond this
    max_total_chars: int = 29000       # tool-output characters returned to the model; bash refuses beyond this
    max_output_chars: int = MAX_OUTPUT_CHARS
    min_output_chars: int = MIN_OUTPUT_CHARS
    warn_calls: int = 5                # remaining-budget line when <= this many calls remain
    warn_fraction: float = 0.25        # ... or <= this share of the character budget remains
    command_timeout_sec: int = 60      # each command runs under `timeout -k 5 <this>`

    @classmethod
    def from_limits(cls, max_tool_calling_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
                    max_completion_length: int = DEFAULT_MAX_COMPLETION_TOKENS,
                    chars_per_token: float = 2.4, **overrides) -> "EpisodeBudget": ...
        # max_bash_calls = iterations - 2; max_total_chars = int(0.75 * max_completion_length * chars_per_token)
```

Training, evaluation, the pilot and the OpenEnv wrapper all build the budget with `EpisodeBudget.from_limits(...)`
from the same two numbers. `SYSTEM_PROMPT` stays in `harness.py` and is the only system prompt used in training rows.

## 2. Tools (shared by every harness of ours)

```python
class ReportTools:                      # harness.py, trl-free mixin; needs self._accept_report(raw: dict) -> str
    def submit_report(self, status: str, completed: list[str], incomplete: list[str], blocker_category: str,
                      blocker_target: str, blocker_detail: str, evidence_commands: list[str], confidence: float): ...
    def abstain(self, reason: str, blocker_category: str, blocker_target: str, confidence: float): ...

def tool_schemas(include_bash: bool = True) -> list[dict]   # JSON schemas via transformers.utils.get_json_schema
def report_from_tool_args(tool: str, args: dict) -> SelfReport  # raises report.ReportParseError
def read_report_file_text(text: str) -> SelfReport              # same schema, for REPORT_FILE
```

- No return annotations on tool methods (keeps training and evaluation schemas identical).
- Docstrings are Google style and include the one-line category definitions from `report.category_definitions_text()`.
- `HonestHarborEnv(ReportTools, HarborEnv)` adds `def bash(self, command: str)`. The pilot's `ReplayEnv(ReportTools)`
  exposes only the two report tools. Evaluation and OpenEnv take schemas from `tool_schemas()`.

## 3. Environment (`env.py`)

```python
HonestHarborEnv(environment_type: str = "docker", reward_config: RewardConfig | None = None,
                grading_config: GradingConfig | None = None, budget: EpisodeBudget | None = None,
                enforce_network_policy: bool = True)
```

`reset(task_dir=None, **row) -> str` returns the instruction at once and boots in the background, as today.

Stable private API that other components rely on (keep these names):

| Name | Meaning |
|---|---|
| `_start_grading() -> concurrent.futures.Future` | start grading on the env's loop without blocking |
| `_graded() -> tuple[EpisodeOutcome, RewardBreakdown, dict]` | graded result, cached per episode |
| `_accept_report(raw: dict) -> str` | record the first report; later calls are refused |
| `_report`, `_report_error`, `_history`, `_n_actions_at_report`, `_stop_reason` | episode state |
| `_boot`, `_env`, `_task`, `_spec`, `_loop`, `_plan`, `_stop()` | sandbox handles (used by oracle validation and the Harbor agent) |
| `_set_external_report(report: SelfReport | None, error: str | None)` | inject a report read from `REPORT_FILE` |

The debug dict returned by `_graded()` contains at least: `history` (list of `{command, rc}`), `blocker_facts`
(`bank.BlockerFacts.model_dump()`), `stop_reason`, `test_log`, and `error` when grading failed.

## 4. Grading (`grading.py`) and verifier (`verifier/`)

```python
class GradingError(RuntimeError): ...
class ManifestError(GradingError): ...
class ToolchainError(GradingError): ...

@dataclass
class GradingConfig:
    tests_timeout_sec: int = 180
    evidence_timeout_sec: int = 30
    max_evidence_commands: int = 8
    manifest_timeout_sec: int = 120
    success_threshold: float = 1.0      # legacy verifier only
    replay_evidence: bool = False       # evaluation only
    verifier_bundle_dir: Path | None = None

async def snapshot(sb, spec, cfg=None) -> Baseline
async def self_check(sb, spec, cfg=None) -> None                      # raises ToolchainError; run at reset
async def check_blocker(sb, report, *, agent_user=None, network_policy=None) -> str   # reward.BLOCKER_*
async def collect_blocker_facts(sb, *, paths: Iterable[str], agent_user=None, network_policy=None) -> BlockerFacts
async def read_report_file(sb, *, agent_user=None) -> tuple[SelfReport | None, str | None]
async def grade(sb, spec, baseline, *, report, report_error, n_actions, agent_acted, stop_reason, tests_dir,
                verifier_env=None, verifier_user=None, agent_user=None, network_policy=None,
                cfg=None) -> tuple[EpisodeOutcome, dict]

# verifier/
def default_bundle_dir() -> Path                 # env HONEST_VERIFIER_BUNDLE, else a user cache dir
def ensure_bundle(path: Path | None = None) -> Path   # raises with the build command if missing
@dataclass
class VerifierResult:
    passed: bool; pass_fraction: float | None; n_tests: int; exit_code: int | None
    log: str; tamper: bool; tamper_details: list[str]
async def run_pytest(sb, tests_dir: Path, *, user=None, timeout_sec: int, site: bool = False,
                     bundle_dir: Path | None = None) -> VerifierResult
```

`sb` is anything with Harbor's async `exec(command, cwd, env, timeout_sec, user)` and `upload_dir(source, target)`.

## 5. Verifier contract for a task (written by the data pipeline)

A task graded hermetically has, in `task.toml`:

```toml
[metadata.honest]
verifier = "pytest"
verifier_site = false        # optional; true lets the tests import the image's site-packages (less hermetic)
```

and:
- `tests/` holds one or more `test_*.py` files that need only the standard library and pytest. The grader uploads
  `tests/` and the vendored pytest bundle fresh and runs pytest on `/tests` itself. Pass means exit code 0 with at
  least one test collected.
- `tests/test.sh` must still exist and run the same tests **offline**, writing `/logs/verifier/reward.txt`, so the
  task stays a valid Harbor task for `harbor run` with other harnesses. Our grader does not use it.
- The image provides `python3` (3.9 or newer) under a protected path.
- The tests fail on the untouched environment and pass after the oracle solution.

## 6. Data pipeline

Dataset row columns (JSONL, one row per variant), consumed by training, the pilot and evaluation:

`prompt` (messages: system = `harness.SYSTEM_PROMPT`, then an empty user message), `task_dir`, `variant_id` (unique;
the prompt-group id), `pair_id`, `feasible`, `is_control`, `kind` (`feasible` / `control` / `infeasible`), `family`,
`perturbation`, `source`, `split`.

`tasks/validate.py` keeps its public names (`ValidationResult`, `judge`, `validate_task`, `validate_many`,
`read_results`, `filter_valid`, `task_hash`, `is_current`, `OracleHarborEnv`). `ValidationResult` gains:
`oracle_runs: int`, `oracle_passes: int`, `null_check_failed_as_expected: bool | None`,
`relabel_to_control: bool` (set by the `remove_input` trivial-repair probe), `leak: bool`.
`validate_task` gains keyword arguments `oracle_runs: int = 3`, `null_check: bool = True`, `repair_probe: bool = True`.

`tasks/hermetic.py` (new) exposes `to_pytest_verifier(task_dir: Path) -> HermeticReport` (rewrites the task to the
contract in section 5 and returns what it changed, or why it could not) and `is_hermetic(task_dir: Path) -> bool`.

## 7. Training

`honest_agents.trainer.HonestGRPOTrainer` is a drop-in subclass of `trl.GRPOTrainer` with the same constructor.
`train.py` instantiates it. `honest_agents.callbacks.StopRules` is a `transformers.TrainerCallback`. Metric names are
the constants and strings logged by `trl_rewards.py`; import or copy them from there, never invent names.

Reward mode → function: `trl_rewards.reward_func_for_mode(mode)`; modes are `honest`, `outcome`, `outcome_claimed`,
`outcome_reflect`. One function per run, `reward_weights = [1.0]`.
