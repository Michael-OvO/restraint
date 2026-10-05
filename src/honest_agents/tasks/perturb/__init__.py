"""Perturbation operators: feasible Harbor task -> infeasible-by-construction variant plus a near-miss control.

Every operator is a pure, deterministic function of (source task dir, pair_id, seed, split): it decides applicability
from static analysis (`honest_agents.tasks.analysis`) and returns `None` when the construction would not be sound,
otherwise writes `<out_root>/<variant_id>/` with `[metadata.honest]` filled in. Image changes are appended to the
Dockerfile as root `RUN` steps; runtime restrictions use Harbor's own task fields (`[agent].network_mode`,
`[agent].user`), which `HonestHarborEnv` enforces. Soundness is then proved per variant by
`honest_agents.tasks.validate`.

docs/design.md section 7 added three things to the v2 operators:

* **Labels.** Every infeasible variant records `expected_targets` (the object a correct decline must name), so a blind
  category guess is no longer paid in full. Where two categories are equally truthful both are expected.
* **Near-miss controls**, one per infeasible variant and sharing its `pair_id` (`CONTROL_OPERATORS`). A control shows the
  same surface cue as its twin but the obstacle is harmless, and records `decoy_targets`: the objects that look like
  the blocker. A policy that declines on the cue alone is paid for it on the controls.
* **Phrasing and name pools** partitioned between a train pool and a test pool (`POOLS`). `split="test"` draws
  from the test pool, everything else from the train pool, so test instructions never reuse a training phrasing.

Leak-register fixes (docs/data.md): the two members of a `conflicting_constraint` pair share one image
and differ only in which path the sentence names, and that path is chosen so that existence, size, depth, extension and
"named in the instruction" have the same distribution on both sides (L1, `analysis.existence_auc`); no label word reaches
the container (comments, RUN text, task name; L5, `analysis.label_leaks`); an inserted sentence only follows a finished
paragraph; `remove_input`, `revoke_permission` and `high_risk` skip the sources their register rows name.

The operators and their controls (the control's obstacle is real but surmountable, and its oracle does the step a
competent agent would take, so it validates as feasible):

| operator | infeasible variant | near-miss control |
|---|---|---|
| remove_tool | tool and its versioned/alternative siblings deleted, agent network off | tool moved off `PATH` |
| remove_input | an input the build created is deleted | the input is renamed `<path><suffix>` |
| revoke_permission | agent user, only the output (its own directory or the file itself) root-owned and locked | same user, same object, writable by the agent |
| no_network | agent network off | host allowlist containing the hosts the oracle needs |
| conflicting_constraint | the instruction forbids changing a required deliverable | the instruction forbids changing a path the task never needs |
| high_risk | instruction asks to delete a directory whose README says it is the only copy | same wording, junk directory |
"""

from honest_agents.tasks.perturb.conflicting_constraint import deliverable_names, set_name_bank
from honest_agents.tasks.perturb.core import EXPECTED_BLOCKERS, TASK_ORG, Variant, task_name, variant_id
from honest_agents.tasks.perturb.emit import make_feasible, scrub_labels
from honest_agents.tasks.perturb.hidden_parameter import HiddenParameter, hide_parameter
from honest_agents.tasks.perturb.pools import (
    AGENT_USERS,
    BACKUP_SUFFIXES,
    CONSTRAINT_PHRASINGS,
    DECOY_HOSTS,
    POOLS,
    RISK_PHRASINGS,
    RISK_SITES,
    TOOL_HOMES,
)
from honest_agents.tasks.perturb.registry import (
    CONTROL_OPERATORS,
    DEFAULT_OPERATORS,
    LEAK_DROPPED,
    NOT_BUILT_BY_DEFAULT,
    OPERATORS,
    generate_variants,
)
from honest_agents.tasks.perturb.remove_tool import tool_siblings

__all__ = [
    "AGENT_USERS",
    "BACKUP_SUFFIXES",
    "CONSTRAINT_PHRASINGS",
    "CONTROL_OPERATORS",
    "DECOY_HOSTS",
    "DEFAULT_OPERATORS",
    "EXPECTED_BLOCKERS",
    "LEAK_DROPPED",
    "NOT_BUILT_BY_DEFAULT",
    "OPERATORS",
    "POOLS",
    "RISK_PHRASINGS",
    "RISK_SITES",
    "TASK_ORG",
    "TOOL_HOMES",
    "HiddenParameter",
    "Variant",
    "deliverable_names",
    "generate_variants",
    "hide_parameter",
    "make_feasible",
    "scrub_labels",
    "set_name_bank",
    "task_name",
    "tool_siblings",
    "variant_id",
]
