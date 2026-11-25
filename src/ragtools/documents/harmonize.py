"""Harmonizing: a document's pages, read one by one, made consistent with each other by a model.

Page-by-page OCR knows nothing of the other pages: heading levels drift
from page to page, and paragraphs, lists, tables and display equations
are cut at page boundaries. The model sees the pages together, a batch at
a time, and hands back one marked document, which `split_pages` takes
apart into the pages again.
"""

from __future__ import annotations

from pathlib import Path

from ragtools.documents.stitch import PAGE_MARKER, mark_page
from ragtools.llm import get_model

# The prompts, kept as documents next to this module; the user prompt is a
# template taking the fenced pages in place of {pages}
HARMONIZE_SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "harmonize_system.md").read_text(encoding="utf-8")
HARMONIZE_USER_PROMPT = (Path(__file__).parent / "prompts" / "harmonize_user.md").read_text(encoding="utf-8").strip()

# One call sees this many pages, the last ones of a batch seen again by the
# next, so a long document never has to come back whole from a single call
# and no page is harmonized without its neighbours
BATCH_PAGES = 10
BATCH_OVERLAP = 2


def harmonize_pages(pages: list[tuple[int, str]]) -> str:
    """Harmonize a document's (page number, text) pages with a model; the marked document comes back.

    Heading levels are made consistent across pages, and what a page
    boundary cut (paragraphs, lists, tables, display equations) is merged.
    The pages go a batch at a time, consecutive batches sharing their
    boundary pages, and are joined again at a page marker. The model is told
    to keep every marker: callers split the answer with `split_pages` and
    check that the pages they gave are the pages they get.
    """
    marked = [mark_page(number, text) for number, text in pages]
    step = BATCH_PAGES - BATCH_OVERLAP

    document = ""
    for start in range(0, len(marked), step):
        batch = marked[start : start + BATCH_PAGES]
        answer = _harmonize_batch(batch) or "\n\n".join(batch)

        # Joined at the marker of the batch's second page: the page before
        # it was harmonized with what precedes it, the pages from it on with
        # what follows them
        seam = PAGE_MARKER.format(position=pages[start + 1][0]) if start and len(batch) > 1 else None
        if seam and seam in document and seam in answer:
            document = f"{document[: document.index(seam)].rstrip()}\n\n{answer[answer.index(seam) :]}"
        else:
            document = f"{document}\n\n{answer}".strip()

        if start + BATCH_PAGES >= len(marked):
            break

    return document


def _harmonize_batch(marked_pages: list[str]) -> str:
    """One call: the marked pages, each in its own fence; empty when the model answers with nothing."""
    fenced = "\n\n".join(f"```\n{page}\n```" for page in marked_pages)

    # Replaced rather than formatted: the document's own braces (LaTeX) must stay as they are
    user_prompt = HARMONIZE_USER_PROMPT.replace("{pages}", fenced)

    answer = get_model("harmonize").generate(user_prompt, system=HARMONIZE_SYSTEM_PROMPT, name="harmonize-pages")
    return answer.strip()
