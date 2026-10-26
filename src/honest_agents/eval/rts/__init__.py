"""Restraint Transfer Suite (RTS): the multi-benchmark transfer endpoint.

Design: docs/superpowers/specs/2026-10-25-restraint-transfer-suite-design.md. `manifest` (the frozen item list),
`protocol` (the pre-registration), `units` (episodes to `transfer` units), `score` (RTS, RTS-C, the endpoint),
`scoreboard` (one row per model, CLI) and `plots` (the HTML page). Building the suite: `benchmarks` (cards and seed
loaders), `candidates` (applicability and pairs), `gate` (Docker checks), `freeze` (static checks and the frozen
manifest), driven by the `build` CLI.
"""
