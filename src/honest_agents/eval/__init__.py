"""Evaluation harness: rollouts against a served policy, Harbor-run benchmarks under our harness or
a third-party one (report-file protocol), metrics, baselines that need no training, and external adapters.

`records`, `metrics`, `baselines`, `protocol` and `summary` only need the reward side of the core package. `rollout`,
`harbor_agent`, `report_file` and `run_eval` also need `honest_agents.harness`; openai and the TRL environment are
imported lazily.
"""
