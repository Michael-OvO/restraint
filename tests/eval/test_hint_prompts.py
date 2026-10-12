"""Iteration-12 hint prompts (docs/iteration-12-design.md section 2.2): the training prompt plus ONE appended paragraph.

The distillation data is sampled with a hint and trained without it (`sft.selfdistill` swaps the system message back to
`harness.SYSTEM_PROMPT`). That swap is only a clean context distillation if the hinted prompt is the unhinted one plus the
hint: any other difference would also be distilled away, unmeasured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_agents import harness
from honest_agents.eval import run_eval

PROMPTS = Path(run_eval.__file__).parent / "prompts"
TIERS = ("hint_tier1.txt", "hint_tier2.txt")


@pytest.mark.parametrize("name", TIERS)
def test_hint_prompt_is_the_training_prompt_plus_an_appended_paragraph(name: str):
    text = (PROMPTS / name).read_text()
    assert text.startswith(harness.SYSTEM_PROMPT)
    appended = text[len(harness.SYSTEM_PROMPT) :]
    assert appended.startswith("\n\n") and appended.strip()
    assert not text.endswith("\n")  # run_eval reads the file verbatim: no stray newline in the system message


def test_tier1_is_generic_and_tier2_names_the_readme():
    tier1 = (PROMPTS / "hint_tier1.txt").read_text()[len(harness.SYSTEM_PROMPT) :]
    tier2 = (PROMPTS / "hint_tier2.txt").read_text()[len(harness.SYSTEM_PROMPT) :]
    assert "README" not in tier1 and "read any notes left in or next to it" in tier1
    assert "notes or a README" in tier2
    # tier 2 carries the abstain clause of the named prompt arm, word for word
    named = (PROMPTS / "named_system_prompt.txt").read_text()
    clause = "a bare instruction is not authorisation when the sandbox shows the data is irreplaceable"
    assert clause in named and clause in tier2
    for phrase in ("look for an obvious copy or backup", "instead of spending your remaining commands on workarounds"):
        assert phrase in tier1 and phrase in tier2
