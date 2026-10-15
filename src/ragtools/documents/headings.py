"""Heading-aware section splitting: the text cut's chunker and the annotator's input bound."""

from __future__ import annotations

import logging

import re

logger = logging.getLogger(__name__)

# Sections merge until they reach this many lines; it also bounds a chunk's
# size, since a section that cannot be split further would otherwise be
# indexed as one oversized chunk
MAX_SECTION_LINES = 2000


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


def merge_up_to_max_lines(sections: list[str], max_lines: int = MAX_SECTION_LINES) -> list[str]:
    """Merge adjacent sections while they fit within max_lines together."""
    merged: list[str] = []
    merged_lines = 0
    for section in sections:
        section_lines = len(section.splitlines())
        if merged and merged_lines + section_lines <= max_lines:
            merged[-1] = f"{merged[-1]}\n{section}"
            merged_lines += section_lines
        else:
            merged.append(section)
            merged_lines = section_lines
    return merged
