import logging
from pathlib import Path, PurePosixPath
from lxml import etree
import re
from html import unescape
from urllib.parse import unquote
import unicodedata
from rag_etl.utils import sanitize_for_filename

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


def get_filename_via_assets(course_path: str, href: str, assets_map: dict[str, str]) -> Path:
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
    real_name = sanitize_for_filename(real_name)

    return Path(course_path) / "static" / real_name


def extract_number(resource_title: str) -> str | None:
    """
    Extract the numbering from a MOOC resource title.

    The number is the first run of digits and dots, wherever it sits in the
    title: some MOOCs open with it ("1.3.3. Digital Images - ...") and others
    close with it ("... - Question 3.1.1"). Trailing dots are dropped, so both
    forms yield a bare "1.3.3".

    Returns None for a title carrying no digits at all.
    """

    number = ""
    for character in resource_title:
        if character.isdigit():
            number += character
        elif character == "." and number:
            number += character
        elif number:
            break

    resource_number = number.strip(".")

    if not resource_number:
        return None

    return resource_number


def extract_week(resource_number: str | None) -> int | None:
    """
    Infer the week a resource belongs to from its numbering.

    MOOC numbering opens with the week, so "1.3.3" is material of week 1.
    Checked against the BIO695 titles that also state their week in words,
    where the first component matched in every case.

    Only a dotted number counts. A bare "1" is far more often a part number,
    as in CS-119(d)'s "Branchements conditionnels (partie 1)", where reading
    it as a week would scatter one lesson across three of them.

    Returns None when the number carries no leading week.
    """

    if not resource_number:
        return None

    components = resource_number.split(".")

    if len(components) < 2 or not components[0].isdigit():
        return None

    week = int(components[0])

    return week
