"""Static analysis of a Harbor task (oracle script, instruction, Dockerfile, tests).

Perturbation operators decide applicability from this alone, without Docker. It is deliberately conservative: a
missed candidate just means one fewer variant, while a wrong one is caught later by oracle validation
(`honest_agents.tasks.validate`). The shell parsing is heuristic (no full bash grammar) and errs towards ignoring
anything dynamic (`$VAR` paths, heredoc bodies, sourced scripts).

docs/design.md section 7 adds three stricter readings, all static:

* `image_files` — which files the image build creates and with what content, so `remove_input` only deletes regular,
  non-empty build-time files whose content the instruction does not already print;
* `mandates_tool` / `interpreter_can_substitute` — `remove_tool` applies only when the instruction requires the tool
  and no interpreter in the image can stand in for it;
* `network_hosts` — the hosts a download needs, which become the `expected_targets` of `no_network` and the allowlist
  of its near-miss control.
"""

from honest_agents.tasks.analysis.boot import boot_dirs, exists_at_boot, oracle_touches, paths_overlap
from honest_agents.tasks.analysis.cues import (
    DONOR_GROUPS,
    ENFORCED_ENV_TEXT,
    ENFORCED_FEATURES,
    INSTRUCTION_FAMILIES,
    SEMANTIC_FEATURES,
    ExistenceLeakError,
    attach_env_text,
    auc_score,
    best_rule_accuracy,
    check_existence,
    cue_classifiers,
    cue_null,
    donor_group,
    existence_auc,
    existence_check,
    existence_rows,
    existence_violations,
    name_token_frac,
    name_tokens,
    subject_features,
)
from honest_agents.tasks.analysis.envtext import (
    CARRIER_CHANNELS,
    appended_text,
    env_text_classifiers,
    environment_text,
)
from honest_agents.tasks.analysis.facts import TaskFacts, build_inputs
from honest_agents.tasks.analysis.filters import (
    code_values_of_tests,
    content_bound_tokens,
    content_shape,
    dir_is_scanned,
    example_paths,
    guessable_tokens,
    has_scope_clause,
    hidden_counts,
    is_dir_like,
    is_link_only,
    is_pass_through,
    operand_commands,
    plausible_recreation,
    safe_boundaries,
    symlink_targets,
    token_overlap,
    unreadable_inputs,
)
from honest_agents.tasks.analysis.image_files import (
    CONTENT_KINDS,
    FILE_COPIED,
    FILE_GENERATED,
    FILE_HEREDOC,
    FILE_PRINTED,
    FILE_TOUCHED,
    image_files,
    line_overlap,
)
from honest_agents.tasks.analysis.labels import label_leaks
from honest_agents.tasks.analysis.mandates import (
    INTERPRETER_PROOF_TOOLS,
    interpreter_can_substitute,
    mandates_tool,
    network_hosts,
    url_host,
)
from honest_agents.tasks.analysis.script import (
    analyse_dockerfile,
    analyse_script,
    extract_paths,
    is_ignored_path,
    likely_image_binaries,
    mentions_as_tool,
    mentions_word,
    network_targets,
    network_uses,
    strip_heredoc_bodies,
)
from honest_agents.tasks.analysis.tables import ESSENTIAL_BINARIES, SYSTEM_DIRS

__all__ = [
    "CARRIER_CHANNELS",
    "CONTENT_KINDS",
    "DONOR_GROUPS",
    "ENFORCED_ENV_TEXT",
    "ENFORCED_FEATURES",
    "ESSENTIAL_BINARIES",
    "FILE_COPIED",
    "FILE_GENERATED",
    "FILE_HEREDOC",
    "FILE_PRINTED",
    "FILE_TOUCHED",
    "INSTRUCTION_FAMILIES",
    "INTERPRETER_PROOF_TOOLS",
    "SEMANTIC_FEATURES",
    "SYSTEM_DIRS",
    "ExistenceLeakError",
    "TaskFacts",
    "analyse_dockerfile",
    "analyse_script",
    "appended_text",
    "attach_env_text",
    "auc_score",
    "best_rule_accuracy",
    "boot_dirs",
    "build_inputs",
    "check_existence",
    "code_values_of_tests",
    "content_bound_tokens",
    "content_shape",
    "cue_classifiers",
    "cue_null",
    "dir_is_scanned",
    "donor_group",
    "env_text_classifiers",
    "environment_text",
    "example_paths",
    "existence_auc",
    "existence_check",
    "existence_rows",
    "existence_violations",
    "exists_at_boot",
    "extract_paths",
    "guessable_tokens",
    "has_scope_clause",
    "hidden_counts",
    "image_files",
    "interpreter_can_substitute",
    "is_dir_like",
    "is_ignored_path",
    "is_link_only",
    "is_pass_through",
    "label_leaks",
    "likely_image_binaries",
    "line_overlap",
    "mandates_tool",
    "mentions_as_tool",
    "mentions_word",
    "name_token_frac",
    "name_tokens",
    "network_hosts",
    "network_targets",
    "network_uses",
    "operand_commands",
    "oracle_touches",
    "paths_overlap",
    "plausible_recreation",
    "safe_boundaries",
    "strip_heredoc_bodies",
    "subject_features",
    "symlink_targets",
    "token_overlap",
    "unreadable_inputs",
    "url_host",
]
