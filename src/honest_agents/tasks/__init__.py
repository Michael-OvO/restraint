"""Infeasible-task generation pipeline (docs/design.md section 7).

`sources` downloads feasible seed tasks and normalises them into hermetic Harbor task dirs, `perturb` derives
infeasible-by-construction variants and their near-miss controls from them, `validate` proves each variant with the
task's oracle through the same `HonestHarborEnv` code path training uses, and `build` assembles the TRL dataset
(`python -m honest_agents.tasks.build`): splits by absolute seed count, held-out families, the plain feasible / control /
infeasible mixture, the quarantine list. `difficulty` turns a Phase-0 trajectory bank into the difficulty gate and the
alpha-safety report; `audit` builds the small hand-reviewed set with one sheet per variant.
"""
