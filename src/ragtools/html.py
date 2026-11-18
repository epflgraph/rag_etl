"""HTML → Markdown, and the small text clean-ups markup conversions share.

The conversion is markdownify's: every text of the page is kept, with its
structure (headings, lists, tables, code, emphasis, links). What is not
content is removed first (scripts, styles, navigation), and runs of blank
lines are collapsed after.
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import ATX, MarkdownConverter


def clean_text(text: str) -> str:
    """Unescape HTML entities, turn non-breaking spaces into spaces and collapse horizontal whitespace."""

    if not text:
        return ""

    t = unescape(text)
    t = t.replace("\xa0", " ")

    # collapse horizontal whitespace only (keep \n for structure)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def normalize_markdown(md: str) -> str:
    """Normalize newlines, trailing spaces and runs of blank lines."""

    if not md:
        return ""

    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = "\n".join(line.rstrip() for line in md.splitlines())
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def escape_markdown(text: str) -> str:
    """Escape the characters Markdown would read as emphasis."""

    if not text:
        return ""
    t = text.replace("\\", "\\\\")
    t = t.replace("*", "\\*")
    t = t.replace("_", "\\_")
    return t


def html_to_markdown(soup: BeautifulSoup) -> str:
    """Convert a parsed HTML page to Markdown. The soup is cleaned in place."""

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag in soup.select("nav#menu, nav#im"):
        tag.decompose()

    markdown = MarkdownConverter(heading_style=ATX, bullets="-").convert_soup(soup)
    return normalize_markdown(markdown)


def convert_html_to_md(html_path: str | Path, md_path: str | Path) -> None:
    """Convert an HTML file to a Markdown file."""
    html = Path(html_path).read_text(encoding="utf-8", errors="replace")
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(html_to_markdown(BeautifulSoup(html, "html.parser")), encoding="utf-8")
