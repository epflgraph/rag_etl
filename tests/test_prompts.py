"""Every prompt document ships and loads: a missing file fails here, not mid-run."""

import pytest

from ragtools.documents.exercises import EXERCISE_SYSTEM_PROMPT
from ragtools.documents.harmonize import HARMONIZE_SYSTEM_PROMPT, HARMONIZE_USER_PROMPT
from ragtools.images import PAGE_SYSTEM_PROMPT, SLIDE_SYSTEM_PROMPT, TRANSCRIBE_USER_PROMPT


@pytest.mark.parametrize(
    "prompt",
    [
        EXERCISE_SYSTEM_PROMPT,
        HARMONIZE_SYSTEM_PROMPT,
        HARMONIZE_USER_PROMPT,
        PAGE_SYSTEM_PROMPT,
        SLIDE_SYSTEM_PROMPT,
        TRANSCRIBE_USER_PROMPT,
    ],
)
def test_prompt_loads(prompt):
    assert prompt.strip()


def test_harmonize_user_prompt_takes_the_pages():
    assert "{pages}" in HARMONIZE_USER_PROMPT


def test_transcription_prompts_share_the_rules_and_differ_in_their_specifics():
    for prompt in (PAGE_SYSTEM_PROMPT, SLIDE_SYSTEM_PROMPT):
        assert "{specifics}" not in prompt
        assert "**Mathematical Content**" in prompt
    assert "single PDF page" in PAGE_SYSTEM_PROMPT and "single PDF page" not in SLIDE_SYSTEM_PROMPT
    assert "handwritten" in SLIDE_SYSTEM_PROMPT and "handwritten" not in PAGE_SYSTEM_PROMPT
