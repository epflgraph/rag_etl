import logging
from pathlib import Path, PurePosixPath
from lxml import etree
import re
from html import unescape
from urllib.parse import unquote
import unicodedata

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


# For comparison only
def cmp_key(s: str) -> str:

    # Standardize text (except accents) for comparison
    # .casefold() -> like .lower() but unicode-aware
    s = unicodedata.normalize("NFKC", s).casefold()

    # Strip accents
    s = unicodedata.normalize("NFKD", s)

    filtered_chars = []
    for ch in s:
        # Is the character a combining character in Unicode?
        # "é" can be one character or two: "e" + " ́" (combining acute accent)
        if not unicodedata.combining(ch):
            filtered_chars.append(ch)

    s = "".join(filtered_chars)

    # Keep only letters+digits
    s = re.sub(r"[^0-9a-z]+", "", s)

    return s


def get_filename_via_assets(
    course_path: str, href: str, assets_map: dict[str, str]
) -> Path:
    """
    Find actual file path using previously loaded assets.json into assets_map
    """
    # unquote: %20 to space, etc.
    # URLs are POSIX-style, not tied to OS running
    url_path = PurePosixPath(unquote(href))

    try:
        rel = url_path.relative_to("/")
    except ValueError:
        return Path(course_path) / url_path

    if not rel.parts or rel.parts[0] != "static":
        return Path(course_path) / rel

    href_name = rel.name

    compare_key = cmp_key(href_name)

    real_name = assets_map.get(compare_key, href_name)

    return Path(course_path) / "static" / real_name
