"""Stitching: pages read one by one, made one document again.

Page-by-page OCR knows nothing of the other pages: heading levels drift
from page to page, tables, lists and display equations are cut at page
boundaries, words are split by hyphenation. Two stitches mend that:

- `llm_stitch` harmonises a document's pages with a model, a batch of
  pages at a time, and hands the pages back apart (`split_pages`);
- `stitch` joins pages into one text with plain code: page markers,
  tables and equations cut by the boundary, hyphenation. Zero LLM calls.
"""

from __future__ import annotations

import re

from pathlib import Path

from ragtools.llm import get_model

# Page boundary markers, one per page, in strict page order
PAGE_MARKER = "<!-- page {position} -->"
PAGE_MARKER_RE = re.compile(PAGE_MARKER.format(position=r"(\d+)"))

# The LLM stitch's prompts, kept as documents next to this module; the user
# prompt is a template taking the fenced pages in place of {pages}
STITCH_SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "stitch_markdown_pages.md").read_text(encoding="utf-8")
STITCH_USER_PROMPT = (Path(__file__).parent / "prompts" / "stitch_user.md").read_text(encoding="utf-8").strip()

# One call sees this many pages, the last ones of a batch seen again by the
# next, so a long document never has to come back whole from a single call
# and no page is harmonised without its neighbours
BATCH_PAGES = 10
BATCH_OVERLAP = 2


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


def llm_stitch(pages: list[tuple[int, str]]) -> str:
    """Harmonise a document's (page number, text) pages with a model; the marked document comes back.

    Heading levels are made consistent across pages, and what a page
    boundary cut (paragraphs, lists, tables, display equations) is merged.
    The pages go a batch at a time, consecutive batches sharing their
    boundary pages, and are joined again at a page marker. The model is told
    to keep every marker: callers split the answer with `split_pages` and
    check that the pages they gave are the pages they get.
    """
    marked = [mark_page(number, text) for number, text in pages]
    step = BATCH_PAGES - BATCH_OVERLAP

    stitched = ""
    for start in range(0, len(marked), step):
        batch = marked[start : start + BATCH_PAGES]
        answer = _stitch_batch(batch) or "\n\n".join(batch)

        # Joined at the marker of the batch's second page: the page before
        # it was stitched with what precedes it, the pages from it on with
        # what follows them
        seam = PAGE_MARKER.format(position=pages[start + 1][0]) if start and len(batch) > 1 else None
        if seam and seam in stitched and seam in answer:
            stitched = f"{stitched[: stitched.index(seam)].rstrip()}\n\n{answer[answer.index(seam) :]}"
        else:
            stitched = f"{stitched}\n\n{answer}".strip()

        if start + BATCH_PAGES >= len(marked):
            break

    return stitched


def _stitch_batch(marked_pages: list[str]) -> str:
    """One call: the marked pages, each in its own fence, stitched; empty when the model answers with nothing."""
    fenced = "\n\n".join(f"```\n{page}\n```" for page in marked_pages)

    # Replaced rather than formatted: the document's own braces (LaTeX) must stay as they are
    user_prompt = STITCH_USER_PROMPT.replace("{pages}", fenced)

    return get_model("stitch").text(user_prompt, system=STITCH_SYSTEM_PROMPT, name="stitch-markdown-pages").strip()
