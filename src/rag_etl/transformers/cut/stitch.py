"""Stitching: ordered pages joined into one document, deterministic-first.

Page-by-page OCR leaves pages that sometimes need merging: tables or
display equations cut at page boundaries, words split by hyphenation. The
fixes here are deterministic post-processing over the ordered page texts,
zero LLM calls. The LLM stitch runs as a per-document fallback, off by
default, only where deterministic fixes leave problems.
"""

from __future__ import annotations

from pathlib import Path

from ragtools.llm import RCPClient

# Page boundary markers, one per page, in strict page order
PAGE_MARKER = "<!-- page {position} -->"

# The LLM stitch (the old prompt, ported verbatim), a per-document fallback,
# kept as a document next to this module
STITCH_SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "stitch_markdown_pages.md").read_text(encoding="utf-8")


def stitch(pages: list[str], *, llm: RCPClient | None = None, model: str | None = None) -> str:
    """Join ordered page texts into one Markdown document.

    Deterministic fixes run first: page boundary markers, tables and
    display equations split across pages joined, soft hyphenation fixed.
    Passing both `llm` and `model` enables the LLM stitch fallback, which
    rewrites the joined document in a single call.
    """
    text = deterministic_stitch(pages)

    if llm is not None and model is not None:
        text = llm_stitch(text, llm, model) or text

    return text


def deterministic_stitch(pages: list[str]) -> str:
    """Join the pages in order, fixing what page-by-page processing broke.

    Tables and display equations cut by the page boundary are joined so
    they read as one again, and a word split by hyphenation is put back
    together. Each page keeps a boundary marker — except a page that
    continues a table or an equation, where a marker would break the very
    block it belongs to.
    """
    joined = ""
    for position, page in enumerate(pages, start=1):
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
    if _ends_mid_table(previous) and _starts_mid_table(page):
        return f"{previous}\n{page}"
    if _ends_mid_equation(previous) and _starts_mid_equation(page):
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


def _ends_mid_equation(text: str) -> bool:
    """Whether a display equation opened and never closed: an odd count of $$ delimiters."""
    return text.count("$$") % 2 == 1


def _starts_mid_equation(text: str) -> bool:
    """Whether a display equation continues and never opens: an odd count of $$ delimiters."""
    return text.count("$$") % 2 == 1


def llm_stitch(text: str, llm: RCPClient, model: str) -> str | None:
    """Rewrite the joined document with one LLM call, per document, as a fallback.

    Returns None when the model answers with nothing, so the deterministic
    result stands.
    """
    md_text = "\n\n".join(["```\n" + text + "\n```"])

    user_prompt = f"""
    Stitch the following page-level Markdown snippets into one cohesive GitHub-Flavored Markdown document.

    Each snippet is enclosed in triple backticks and appears **in order**.

    Emit **only** the final Markdown (no fences).

    {md_text}
    """

    stitched = llm.text(user_prompt, model=model, system=STITCH_SYSTEM_PROMPT, name="stitch-markdown-pages", enable_thinking=False)
    return stitched.strip() or None
