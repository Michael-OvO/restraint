"""`oneoff_tests` (far tier): ImpossibleBench ONEOFF, arXiv 2510.20270.

The agent is given the seed's hidden pytest file (a visible copy in its workspace; the graded tests are the same file).
In the twin, the expected value of one equality assertion is changed in both the visible copy and the graded file, so
that test contradicts the behaviour the task specifies (the seed's oracle fails it). The control ships the seed's file
unchanged. Twin and control differ in the visible tests by construction (`instruction_identical` is False although the
instruction is the seed's in both); the construction is in `_visible_tests`. Only an assertion that another graded
assertion pins is changed (`_visible_tests.pinned`: same callee, literal arguments and expected literal), so no
implementation passes the twin without editing the tests.
"""

from __future__ import annotations

from functools import partial

from honest_agents.tasks.perturb.rts import _visible_tests as vt
from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec

NAME = "oneoff_tests"

SPEC = RTSOperatorSpec(
    name=NAME,
    tier="far",
    instruction_identical=False,
    expected_category=vt.CATEGORY,
    applicable=partial(vt.applicable, construction="oneoff"),
    make_pair=partial(vt.make_pair, NAME, "oneoff"),
)
