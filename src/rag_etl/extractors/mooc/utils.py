import requests
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from lxml import etree
import re
from urllib.parse import unquote, urlparse
import unicodedata

from ragtools.encoding import sanitize_for_filename

logger = logging.getLogger(__name__)


def load_root_elem_from_mooc_xml(xml_path: Path) -> etree._Element | None:
    """Load an XML element from a MOOC"""

    try:
        tree = etree.parse(str(xml_path))
        return tree.getroot()
    except (OSError, etree.XMLSyntaxError):
        logger.exception("Error loading xml file: %s", xml_path)
        return None


@dataclass(frozen=True)
class Export:
    """What every step of the walk through an edX export needs, read once.

    An export keeps each block of the course in its own file, named
    `<kind>/<url_name>.xml`; the course, chapters, sequentials, verticals
    and the items they hold (html, problem, video) are all found that way.
    """

    dir: Path
    assets_map: dict[str, str]         # assets.json: comparison key of a file name -> the file in static/
    asset_base_url: str | None = None  # the prefix the course's files are published under
    language: str | None = None        # which subtitle track a video keeps

    def xml_path(self, kind: str, url_name: str) -> Path:
        """Where the block of this kind and name is described in the export."""
        return self.dir / kind / f"{url_name}.xml"

    def load(self, kind: str, url_name: str) -> etree._Element | None:
        """The block's root element, or None (logged) when it is missing or unreadable."""
        return load_root_elem_from_mooc_xml(self.xml_path(kind, url_name))


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


def get_filename_via_assets(export_dir: Path, href: str, assets_map: dict[str, str]) -> Path:
    """
    Find actual file path using previously loaded assets.json into assets_map
    """
    # unquote: %20 to space, etc.
    # URLs are POSIX-style, not tied to OS running
    url_path = PurePosixPath(unquote(href))

    try:
        rel = url_path.relative_to("/")
    except ValueError:
        return export_dir / url_path

    if not rel.parts or rel.parts[0] != "static":
        return export_dir / rel

    href_name = rel.name

    compare_key = cmp_key(href_name)

    real_name = assets_map.get(compare_key, href_name)

    # assets.json names the file as the export wrote it, punctuation and all,
    # so it is tried untouched first. Sanitising is only a fallback, for a name
    # the filesystem could not have taken as it stands
    resource_path = export_dir / "static" / real_name
    if resource_path.exists():
        return resource_path

    return export_dir / "static" / sanitize_for_filename(real_name)


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
