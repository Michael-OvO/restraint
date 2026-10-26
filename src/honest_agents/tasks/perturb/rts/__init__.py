"""RTS operators (docs/superpowers/specs/2026-10-25-restraint-transfer-suite-design.md section 3.3).

One module per operator, each defining a module-level `SPEC = RTSOperatorSpec(...)` (`base`); `registry` discovers them,
so adding an operator is adding a file.
"""
