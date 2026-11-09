"""Every prompt document ships and loads: a missing file fails here, not mid-run."""

import pytest

from ragtools.documents.exercises import EXERCISE_SYSTEM_PROMPT
from ragtools.documents.stitch import STITCH_SYSTEM_PROMPT, STITCH_USER_PROMPT
from ragtools.images import ALT_TEXT_PROMPT, PAGE_SYSTEM_PROMPT, SLIDE_SYSTEM_PROMPT, TRANSCRIBE_USER_PROMPT


@pytest.mark.parametrize(
    "prompt",
    [
        EXERCISE_SYSTEM_PROMPT,
        STITCH_SYSTEM_PROMPT,
        STITCH_USER_PROMPT,
        ALT_TEXT_PROMPT,
        PAGE_SYSTEM_PROMPT,
        SLIDE_SYSTEM_PROMPT,
        TRANSCRIBE_USER_PROMPT,
    ],
)
def test_prompt_loads(prompt):
    assert prompt.strip()


def test_stitch_user_prompt_takes_the_document():
    assert "{document}" in STITCH_USER_PROMPT


def test_transcription_prompts_share_the_rules_and_differ_in_their_specifics():
    for prompt in (PAGE_SYSTEM_PROMPT, SLIDE_SYSTEM_PROMPT):
        assert "{specifics}" not in prompt
        assert "**Mathematical Content**" in prompt
    assert "single PDF page" in PAGE_SYSTEM_PROMPT and "single PDF page" not in SLIDE_SYSTEM_PROMPT
    assert "handwritten" in SLIDE_SYSTEM_PROMPT and "handwritten" not in PAGE_SYSTEM_PROMPT
