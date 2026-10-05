"""The system prompt is mechanism-neutral (docs/evaluation.md section 1): it must not name the kinds of impossibility,
or a held-out family would be tested with a prompt that teaches it."""

from __future__ import annotations

import re
from pathlib import Path

from honest_agents import harness
from honest_agents.eval import run_eval
from honest_agents.report import BlockerCategory

PROMPTS = Path(run_eval.__file__).parent / "prompts"
# Words that name a mechanism of a training or held-out family.
MECHANISMS = re.compile(
    r"missing|permission|contradict|conflict|destructive|delet|network|\binput\b|\btool or\b|irreplaceable|authoris",
    re.IGNORECASE,
)


def test_the_system_prompt_names_no_mechanism_and_no_blocker_category():
    assert not MECHANISMS.findall(harness.SYSTEM_PROMPT)
    assert not [c.value for c in BlockerCategory if re.search(rf"\b{c.value}\b", harness.SYSTEM_PROMPT)]


def test_the_system_prompt_still_states_the_behaviour():
    p = harness.SYSTEM_PROMPT
    assert "`abstain`" in p and "`submit_report`" in p
    assert "cannot be completed, or should not be" in p and "verify in the environment" in p
    assert "before you take any irreversible action" in p
    assert "false claim of completion is the worst outcome" in p


def test_the_blocker_categories_stay_in_the_tool_docs_as_shared_vocabulary():
    for doc in (harness.SUBMIT_REPORT_DOC, harness.ABSTAIN_DOC):
        assert all(c.value in doc for c in BlockerCategory)
    assert "cannot or should not be done" in harness.ABSTAIN_DOC  # the tool docs name no mechanism either
    assert "missing" not in harness.ABSTAIN_DOC.split("Categories:")[0].lower()


def test_the_named_prompt_arm_is_the_pre_change_wording():
    named = (PROMPTS / "named_system_prompt.txt").read_text()
    assert named.strip() != harness.SYSTEM_PROMPT.strip()
    for word in ("tool or permission is missing", "requirements contradict", "destructive or irreversible"):
        assert word in named
    assert "`abstain`" in named and "`submit_report`" in named
    assert len(harness.SYSTEM_PROMPT) <= len(named)  # the neutral prompt stays short
