"""Stitching: a document's pages joined into one text with plain code, and taken apart again.

Each page goes under a boundary marker; tables and display equations cut
by a page boundary are rejoined, and words split by hyphenation put back
together. Zero LLM calls: making the pages consistent with each other is
`harmonize.py`'s.
"""

from __future__ import annotations

import re

# Page boundary markers, one per page, in strict page order
PAGE_MARKER = "<!-- page {position} -->"
PAGE_MARKER_RE = re.compile(PAGE_MARKER.format(position=r"(\d+)"))


def stitch(pages: list[str], numbers: list[int] | None = None) -> str:
    """Join the pages in order, fixing what page-by-page processing broke.

    Tables and display equations cut by the page boundary are joined so
    they read as one again, and a word split by hyphenation is put back
    together. Each page keeps a boundary marker — except a page that
    continues a table or an equation, where a marker would break the very
    block it belongs to. The markers carry the pages' own numbers when
    given (a document missing a page), their position otherwise.
    """
    joined = ""
    for position, page in zip(numbers or range(1, len(pages) + 1), pages):
        page = page.strip()
        if not page:
            continue

        if not joined:
            joined = f"{PAGE_MARKER.format(position=position)}\n\n{page}"
            continue

        boundary = _boundary_join(joined, page)
        if boundary is not None:
            joined = boundary
        else:
            joined = f"{joined}\n\n{PAGE_MARKER.format(position=position)}\n\n{page}"

    return joined


def _boundary_join(previous: str, page: str) -> str | None:
    """The two texts joined directly when a block continues across them, None otherwise."""
    if (_ends_mid_table(previous) and _starts_mid_table(page)) or _odd_dollars(previous) and _odd_dollars(page):
        return f"{previous}\n{page}"
    if previous.endswith("-") and page[:1].islower():
        return f"{previous[:-1]}{page}"
    return None


def _ends_mid_table(text: str) -> bool:
    """Whether the last line is a table row, i.e. a table continues past it."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and lines[-1].lstrip().startswith("|")


def _starts_mid_table(text: str) -> bool:
    """Whether the first line is a table row, i.e. a table was cut before it."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and lines[0].lstrip().startswith("|")


def _odd_dollars(text: str) -> bool:
    """Whether a display equation opened without closing: an odd count of $$ delimiters.

    On the previous side this means the text ends mid-equation; on the
    page side, that it starts inside one.
    """
    return text.count("$$") % 2 == 1


def mark_page(number: int, text: str) -> str:
    """A page's text under its boundary marker."""
    return f"{PAGE_MARKER.format(position=number)}\n\n{text.strip()}"


def split_pages(text: str) -> list[tuple[int, str]]:
    """A marked document split back into its (page number, text) pages, in the order found.

    Text before the first marker belongs to the first page.
    """
    markers = list(PAGE_MARKER_RE.finditer(text))
    pages = []
    for position, marker in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(text)
        start = 0 if position == 0 else marker.start()
        pages.append((int(marker.group(1)), PAGE_MARKER_RE.sub("", text[start:end], count=1).strip()))
    return pages
