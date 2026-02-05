import logging
from pathlib import Path
from lxml import etree
import re
from html import unescape

logger = logging.getLogger(__name__)


def load_root_elem_from_mooc_xml(xml_path: Path) -> etree._Element | None:
    """Load an XML element from a MOOC"""

    try:
        tree = etree.parse(str(xml_path))
        return tree.getroot()
    except (OSError, etree.XMLSyntaxError):
        logger.exception("Error loading xml file: %s", xml_path)
        return None


def clean_text(text: str) -> str:
    """
    Clean text
    """

    if not text:
        return ""

    # HTML unescape
    t = unescape(text)

    # To normal space
    t = t.replace("\xa0", " ")

    # collapse horizontal whitespace only (keep \n for structure)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def normalize_markdown(md: str) -> str:
    """
    Normalize markdown output including newlines, trailing spaces, and extra blank lines
    """

    if not md:
        return ""

    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = "\n".join(line.rstrip() for line in md.splitlines())
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def escape_markdown(text: str) -> str:
    """
    Escape Makdown
    """

    if not text:
        return ""
    t = text.replace("\\", "\\\\")
    t = t.replace("*", "\\*")
    t = t.replace("_", "\\_")
    return t
