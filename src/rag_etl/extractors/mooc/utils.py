import requests
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from lxml import etree
import re
from html import unescape
from urllib.parse import unquote, urlparse
import unicodedata
from pydantic import BaseModel, Field

from rag_etl.config import CONFIG
from rag_etl.transformers.pdf_to_markdown.utils import (
    downscale_if_needed,
    render_pdf_pages,
    to_data_uri,
)
from rag_etl.utils import sanitize_for_filename
from rag_etl.utils.llms import send_llm_request

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
    title: some MOOCs open with it ("1.3.3. Digital Images - ...")

    Returns None for a title carrying no digits at all.
    """

    number = ""
    for character in resource_title:
        if character.isdigit() or character == "." and number:
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

    Returns None when the number carries no leading week.
    """

    if not resource_number:
        return None

    components = resource_number.split(".")

    if len(components) < 2 or not components[0].isdigit():
        return None

    week = int(components[0])

    return week

# One course links the same asset from several pages, so each url is only
# ever asked about once
_url_exists_cache: dict[str, bool] = {}

URL_CHECK_TIMEOUT = 20


def url_exists(url: str) -> bool:
    """
    Return whether a url can be fetched.

    An inferred url is a guess: the name of a file in the export is assumed to
    be the name it was published under. That holds for most of a course and
    fails for the rest, and the only way to tell the two apart is to ask.
    """

    if url in _url_exists_cache:
        return _url_exists_cache[url]

    try:
        response = requests.head(url, allow_redirects=True, timeout=URL_CHECK_TIMEOUT)
    except requests.RequestException as error:
        # Unreachable is not the same as absent. A dropped connection would
        # otherwise strip urls from a whole run, so the answer is kept out of
        # the cache and the url is trusted until something disproves it
        logger.warning(f"Could not check {url}, keeping it anyway: {error}")
        return True

    exists = response.status_code < 400

    # A course platform is a single page app: it answers 200 with its own page
    # for any path at all, so a status alone cannot tell a real file from a
    # wrong prefix. None of the files linked here is a web page
    content_type = response.headers.get("content-type", "")
    if exists and content_type.startswith("text/html"):
        logger.info(f"Answered with a web page rather than the file, so no url for it: {url}")
        exists = False

    if not exists and not content_type.startswith("text/html"):
        logger.info(f"Not published, so no url for it: {url} ({response.status_code})")

    _url_exists_cache[url] = exists

    return exists


@dataclass
class UntaggedDocuments:
    """
    How to treat the documents a MOOC page links without tagging them.

    A course that tags nothing still publishes readings and slide decks, and
    they are worth indexing. Which of the two tags a PDF gets is decided by
    looking at it, since only the file itself says whether it is a deck.
    """

    include: bool = False
    theory_tag: str | None = None
    theory_slides_tag: str | None = None
    extensions: tuple[str, ...] = (".pdf", ".txt", ".zip")


SLIDES_SYSTEM_PROMPT = "You classify the first page of a PDF."

SLIDES_USER_PROMPT = (
    "Is this page from a slide deck used to present a lecture, or from an ordinary "
    "document such as a report, an article, lecture notes, or an exercise sheet?"
)


class SlidesClassification(BaseModel):
    is_slides: bool = Field(
        ...,
        description=(
            "True when the page comes from a slide deck used to present a lecture, "
            "false for an ordinary document such as a report, an article, lecture "
            "notes, or an exercise sheet."
        ),
    )


def pdf_is_slides(pdf_path: Path) -> bool:
    """
    Return whether a PDF is a slide deck.

    Only the first page is looked at: a deck is recognisable from its title
    slide, and one call per document keeps this affordable for a whole course.
    """

    pages = render_pdf_pages(str(pdf_path))
    first_page = downscale_if_needed(pages[0])

    messages = [
        {"role": "system", "content": SLIDES_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SLIDES_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": to_data_uri(first_page)}},
            ],
        },
    ]

    classification = send_llm_request(
        CONFIG["RCP_VISION_MODEL"],
        messages,
        response_format=SlidesClassification,
        name="classify-pdf-slides",
        enable_thinking=False,
    )

    return classification.is_slides


# The key naming a published course, as in "course-v1:EPFL+mems+2023"
COURSE_KEY = re.compile(r"course-v1:([^/?#]+)")


def asset_base_url_from_course_url(course_url: str) -> str | None:
    """
    Return the prefix a course's files are published under, read from the
    address of the course itself.

    A course is known by the page a student opens, as in
    "https://courseware.epfl.ch/learning/course/course-v1:EPFL+mems+2023/home".
    Its files live elsewhere on the same host, under the same key but an
    asset address, which is what this builds.
    """

    parsed = urlparse(course_url)
    match = COURSE_KEY.search(course_url)

    if not (parsed.scheme and parsed.netloc and match):
        logger.warning(f"No course key in {course_url}, so linked files will carry no url")
        return None

    return f"{parsed.scheme}://{parsed.netloc}/asset-v1:{match.group(1)}+type@asset+"
