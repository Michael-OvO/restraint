# Claim splice build

| cell | examples |
|---|---:|
| claim_done/fail | 118 |
| claim_done/pass | 118 |

pairs: 118 (138 before the source-share cap), tasks 118, sources 78, largest source share 0.0169, DPO pairs 118
examples_by_prefix: {"fail/decision_example": 2, "fail/own_transcript": 13, "fail/template": 103, "pass/decision_example": 2, "pass/own_transcript": 13, "pass/template": 103}
examples_by_state: {"fail/episode": 118, "pass/episode": 83, "pass/oracle": 35}
examples_by_surface: {"fail/config_comment": 28, "fail/file_header": 33, "fail/manifest": 30, "fail/readme": 27, "pass/config_comment": 28, "pass/file_header": 33, "pass/manifest": 30, "pass/readme": 27}
states: {"episode/fail:agreed": 226, "episode/fail:archived_replayed_disagree": 3, "episode/fail:checklist_disagrees_with_tests": 21, "episode/pass:agreed": 238, "episode/pass:checklist_disagrees_with_tests": 1, "oracle/pass:agreed": 215}
state_drops: {"episode": {"cache_ambiguous": 10, "not_replayable": 1, "twin_no_claim_point": 250}, "oracle": {}}
tasks: {"with_agreed_fail": 140, "with_agreed_pass": 215, "with_both": 140}
prefixes: {"drops": {}, "loaded": {"decision_example": 89, "own_transcript": 154}}
pair_drops: {"label_word_or_hint": 13, "no_fail_state": 75, "per_task_cap": 87, "prefix_skipped:decision_example:removal_contradicts_prefix": 51, "prefix_skipped:own_transcript:removal_contradicts_prefix": 80, "source_share": 20}
checklists: {"control_tasks": 215, "invalid": {}, "valid": 215, "with_task_dir": 215}
excluded_sources_checked: {"excluded": 0, "sources": 78}
quota: {"cap_per_task": 2, "max_outcome_ratio": 1.0, "max_source_share": 0.02, "pairs_per_task": 1}
