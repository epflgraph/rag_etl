"""Heading-aware section splitting: the text cut's chunker and the annotator's input bound."""

from __future__ import annotations

import logging
import re

from collections.abc import Callable

logger = logging.getLogger(__name__)

# How many lines of a document one exercise-splitting call sees at most
MAX_SECTION_LINES = 2000


def line_count(text: str) -> int:
    """The number of lines in a text."""
    return len(text.splitlines())


def split_by_most_common_heading(text: str) -> list[str]:
    """Split the text at its most common heading level, keeping the headings."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)
    matches = heading_pattern.findall(text)

    if not matches:
        logger.warning("No headings found in Markdown text. Trying without splitting...")
        return [text]

    # Count heading levels
    counts: dict[int, int] = {}
    for hashes, _ in matches:
        level = len(hashes)
        counts[level] = counts.get(level, 0) + 1

    # Find most common level
    most_common_level = max(counts, key=lambda level: counts[level])

    # Split on that level (keep headings using lookahead)
    split_pattern = re.compile(rf"(?=^#{{{most_common_level}}}\s+)", re.MULTILINE)
    sections = split_pattern.split(text)

    return [section.strip() for section in sections if section.strip()]


def merge_sections(
    sections: list[str], max_size: int, size: Callable[[str], int] = line_count
) -> list[str]:
    """Merge adjacent sections while they fit within max_size together.

    `size` measures a text: its line count by default, `len` for characters.
    A section bigger than max_size on its own is kept as it is; the caller
    decides what to do with it.
    """
    merged: list[str] = []
    for section in sections:
        if merged and size(candidate := f"{merged[-1]}\n{section}") <= max_size:
            merged[-1] = candidate
        else:
            merged.append(section)
    return merged
