"""HTML → Markdown, and the small text clean-ups markup conversions share.

The conversion keeps what carries meaning in course pages — headings,
paragraphs, lists, bold — and drops the rest (scripts, styles, navigation,
empty paragraphs).
"""

from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag


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

    _remove_noise(soup)
    _remove_empty_paragraphs(soup)

    out: list[str] = []
    root = soup.body or soup
    _walk(root, out)

    return "\n".join(out).strip()


def _remove_noise(soup: BeautifulSoup) -> None:
    """Remove scripts, styles and navigation menus."""

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag in soup.select("nav#menu, nav#im"):
        tag.decompose()


def _remove_empty_paragraphs(soup: BeautifulSoup) -> None:
    for p in soup.find_all("p"):
        text = p.get_text().replace("\xa0", " ").strip()
        if not text:
            p.decompose()


def _inline_text(node: Tag) -> str:
    """A paragraph's text, bold kept as Markdown bold."""

    parts: list[str] = []

    for child in node.children:
        if isinstance(child, NavigableString):
            t = clean_text(str(child))
            if t:
                parts.append(t)

        elif isinstance(child, Tag):
            t = clean_text(child.get_text(" ", strip=True))
            if not t:
                continue

            if child.name in {"strong", "b"}:
                parts.append(f"**{t}**")
            else:
                parts.append(t)

    return " ".join(parts).strip()


def _walk(node, out: list[str]) -> None:
    """Append the Markdown lines for a node and its descendants."""

    # Plain text is read by its block, never on its own
    if isinstance(node, NavigableString) or not isinstance(node, Tag):
        return

    name = node.name

    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = clean_text(node.get_text(" ", strip=True))
        if text:
            out.append("#" * level + " " + text)
            out.append("")
        return

    if name == "p":
        text = _inline_text(node)
        if text:
            out.append(text)
            out.append("")
        return

    if name == "br":
        out.append("")
        return

    if name == "ul":
        for li in node.find_all("li", recursive=False):
            _walk(li, out)
        out.append("")
        return

    if name == "ol":
        i = 1
        for li in node.find_all("li", recursive=False):
            text = clean_text(li.get_text(" ", strip=True))
            if text:
                out.append(f"{i}. {text}")
                i += 1
        out.append("")
        return

    if name == "li":
        text = clean_text(node.get_text(" ", strip=True))
        if text:
            out.append(f"- {text}")
        return

    # Any other tag: its children speak for it
    for child in node.children:
        _walk(child, out)
