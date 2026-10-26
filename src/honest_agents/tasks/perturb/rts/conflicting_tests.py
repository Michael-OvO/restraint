"""`conflicting_tests` (far tier): ImpossibleBench CONFLICTING, arXiv 2510.20270.

The agent is given the seed's hidden pytest file (a visible copy in its workspace; the graded tests are the same file).
In the twin, one equality assertion of that file keeps its place and gains a copy, on the next line, that expects
another value, in both the visible copy and the graded file, so no implementation passes both. The control ships the
seed's file unchanged. Twin and control differ in the visible tests by construction (`instruction_identical` is False
although the instruction is the seed's in both); the construction is in `_visible_tests`.
"""

from __future__ import annotations

from functools import partial

from honest_agents.tasks.perturb.rts import _visible_tests as vt
from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec

NAME = "conflicting_tests"

SPEC = RTSOperatorSpec(
    name=NAME,
    tier="far",
    instruction_identical=False,
    expected_category=vt.CATEGORY,
    applicable=vt.applicable,
    make_pair=partial(vt.make_pair, NAME, "conflicting"),
)
