# Claim splice build

| cell | examples |
|---|---:|

pairs: 0 (0 before the source-share cap), tasks 0, sources 0, largest source share None, DPO pairs 0
examples_by_prefix: {}
examples_by_state: {}
examples_by_surface: {}
states: {"episode/fail:agreed": 226, "episode/fail:archived_replayed_disagree": 3, "episode/fail:checklist_disagrees_with_tests": 21, "episode/pass:agreed": 238, "episode/pass:checklist_disagrees_with_tests": 1, "oracle/pass:agreed": 215}
state_drops: {"episode": {"cache_ambiguous": 10, "not_replayable": 1, "twin_no_claim_point": 250}, "oracle": {}}
tasks: {"with_agreed_fail": 140, "with_agreed_pass": 215, "with_both": 140}
prefixes: {"drops": {}, "loaded": {"decision_example": 89, "own_transcript": 154}}
pair_drops: {"no_fail_state": 75, "prefix_skipped:decision_example:fail_state_contradicts_prefix": 9, "prefix_skipped:decision_example:removal_contradicts_prefix": 51, "prefix_skipped:own_transcript:fail_state_contradicts_prefix": 24, "prefix_skipped:own_transcript:removal_contradicts_prefix": 80, "source_share": 0}
checklists: {"control_tasks": 215, "invalid": {}, "valid": 215, "with_task_dir": 215}
excluded_sources_checked: {"excluded": 0, "sources": 0}
quota: {"cap_per_task": 2, "max_outcome_ratio": 1.0, "max_source_share": 0.02, "pairs_per_task": 1}
