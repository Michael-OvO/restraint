"""Adapters for external benchmarks (AgentAbstain, Agentic Abstention, AbstentionBench, ImpossibleBench, MASK). See
docs/evaluation.md.

`transfer` turns the scorers' per-episode `units` into the base-vs-trained external transfer endpoint (section 10.1).
Files named `*_standalone.py` run inside the benchmark's *own* virtualenv and must not import `honest_agents`.
"""
