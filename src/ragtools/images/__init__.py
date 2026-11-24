"""Images: one image turned into Markdown text by a vision model.

Rendered PDF pages, lecture slides and figures are all transcribed the
same way, under one set of rules; what is specific to the kind of image
is added on top. The model is the settled one for vision reads, the
endpoint comes from the config. The text comes back, and whoever asked
writes it wherever it belongs.
"""

from __future__ import annotations

import logging
import re

from pathlib import Path

from ragtools.llm import AnswerTooLong, get_model

logger = logging.getLogger(__name__)

# The prompts are documents next to this module: one set of transcription
# rules shared by every kind of image, and the rules specific to each kind
PROMPTS = Path(__file__).parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def _transcribe_system_prompt(specifics: str) -> str:
    """The shared transcription rules, completed with those specific to one kind of image."""
    return _prompt("transcribe_system").replace("{specifics}", _prompt(specifics))


PAGE_SYSTEM_PROMPT = _transcribe_system_prompt("transcribe_page")
SLIDE_SYSTEM_PROMPT = _transcribe_system_prompt("transcribe_slide")
TRANSCRIBE_USER_PROMPT = _prompt("transcribe_user")

# One page, slide or figure never reads to more than this. Asked about a
# repetitive page — a dot leader, a dense figure — the model sometimes
# loops for minutes instead of answering; the longest real transcription
# observed stays a few thousand characters below this mark. Reads are
# streamed and cut the moment they pass the mark.
MAX_ANSWER_CHARS = 20_000


def strip_code_fence(markdown: str) -> str:
    """
    Remove a fence wrapping the whole answer.

    Asked for Markdown, the model sometimes returns the document inside a
    ```markdown block. Left in place, the entire slide would render as a code
    listing, so an outer fence is unwrapped. Fenced code *within* the slide is
    untouched: only a fence on the very first line qualifies.
    """

    lines = markdown.strip().splitlines()

    if len(lines) < 2:
        return markdown.strip()

    if not lines[0].strip().startswith("```"):
        return markdown.strip()

    if lines[0].strip().lower() not in ("```", "```markdown", "```md"):
        return markdown.strip()

    if lines[-1].strip() != "```":
        return markdown.strip()

    return "\n".join(lines[1:-1]).strip()


# A Markdown image with a link target, ![alt](target): the alt may hold one
# level of brackets, the target one level of parentheses
IMAGE_WITH_TARGET = re.compile(r"!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\((?:[^()]|\([^()]*\))*\)")


def strip_image_targets(markdown: str) -> str:
    """
    Leave every image as its description alone: ![alt](target) becomes ![alt].

    A transcript describes the figures it sees in their ALT text. It is read
    off pixels, so it cannot know where a picture is stored, yet the model
    sometimes fills the link in with something plausible (a placeholder
    name, an address that leads nowhere). Nothing invented may reach a
    reader as a link.
    """
    return IMAGE_WITH_TARGET.sub(r"![\1]", markdown)


def transcribe_page(image_path: str | Path) -> str:
    """Transcribe one rendered PDF page into Markdown."""
    return _transcribe(image_path, PAGE_SYSTEM_PROMPT)


def transcribe_slide(image_path: str | Path) -> str:
    """Transcribe one lecture slide or figure into Markdown."""
    return _transcribe(image_path, SLIDE_SYSTEM_PROMPT)


def _transcribe(image_path: str | Path, system_prompt: str) -> str:
    """Read one image with a vision model and return its Markdown text.

    An empty answer stands for a blank slide — or for a read the model
    looped on; callers drop both.
    """

    try:
        markdown = get_model("ocr").generate(
            TRANSCRIBE_USER_PROMPT,
            images=[image_path],
            system=system_prompt,
            detail="high",
            name="image-to-markdown",
            max_chars=MAX_ANSWER_CHARS,
        )
    except AnswerTooLong as e:
        logger.warning(f"Dropping the read of {image_path}: {e} — beyond any transcription")
        return ""

    return strip_image_targets(strip_code_fence(markdown))
