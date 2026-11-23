"""An edX export as the parsers read it: its blocks, its assets and where they are published."""

import json
import logging
import re
import unicodedata

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import requests
from lxml import etree

from ragtools.encoding import sanitize_for_filename

logger = logging.getLogger(__name__)

# The part of an edX asset link that is the same for every asset of a course,
# as in "https://courses.edx.org/asset-v1:EPFLx+init-prog-cpp+1T2025+type@asset+"
ASSET_BASE_URL = re.compile(r"https://[^\"\']*?type@asset\+")

# The key naming a published course, as in "course-v1:EPFL+mems+2023"
COURSE_KEY = re.compile(r"course-v1:([^/?#]+)")

URL_CHECK_TIMEOUT = 20


def load_xml(xml_path: Path) -> etree._Element | None:
    """The root element of one of the export's XML files, or None (logged) when missing or unreadable."""
    try:
        return etree.parse(str(xml_path)).getroot()
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

    @property
    def markdown_dir(self) -> Path:
        """The one folder the extractor writes into the export (the quizzes as Markdown), wiped at every extraction."""
        return self.dir / "markdown"

    def xml_path(self, kind: str, url_name: str) -> Path:
        """Where the block of this kind and name is described in the export."""
        return self.dir / kind / f"{url_name}.xml"

    def load(self, kind: str, url_name: str) -> etree._Element | None:
        """The block's root element, or None (logged) when it is missing or unreadable."""
        return load_xml(self.xml_path(kind, url_name))


def cmp_key(s: str) -> str:
    """A file name as names are compared: case, accents and punctuation set aside."""
    # .casefold() is like .lower() but unicode-aware
    s = unicodedata.normalize("NFKC", s).casefold()

    # "é" can be one character or two ("e" + a combining accent): decomposed,
    # the combining characters are dropped
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    return re.sub(r"[^0-9a-z]+", "", s)


def load_assets_map(export_dir: Path) -> dict[str, str]:
    """Read policies/assets.json: the comparison key of each asset's name -> its file."""

    data = json.loads((export_dir / "policies" / "assets.json").read_text(encoding="utf-8"))

    assets: dict[str, str] = {}
    for asset_key, value in data.items():
        assets[cmp_key(asset_key)] = value.get("import_path") or value.get("displayname") or asset_key

    return assets


def get_filename_via_assets(export_dir: Path, href: str, assets_map: dict[str, str]) -> Path:
    """Find the file a link names, through the names assets.json lists."""
    # unquote: %20 to space, etc.; URLs are POSIX-style, not tied to the OS running
    url_path = PurePosixPath(unquote(href))

    try:
        rel = url_path.relative_to("/")
    except ValueError:
        return export_dir / url_path

    if not rel.parts or rel.parts[0] != "static":
        return export_dir / rel

    real_name = assets_map.get(cmp_key(rel.name), rel.name)

    # assets.json names the file as the export wrote it, punctuation and all,
    # so it is tried untouched first. Sanitising is only a fallback, for a name
    # the filesystem could not have taken as it stands
    resource_path = export_dir / "static" / real_name
    if resource_path.exists():
        return resource_path

    return export_dir / "static" / sanitize_for_filename(real_name)


def load_asset_base_url(export_dir: Path) -> str | None:
    """
    Return the prefix an asset of this course is published under, or None.

    The export's own course key is not usable: course.xml and assets.json
    are rewritten on re-export and name a course that does not exist
    publicly. The absolute links the authors wrote into the pages do name
    the published run, so the prefix is read back from those.
    """

    found: dict[str, int] = {}
    for html_path in (export_dir / "html").glob("*.html"):
        for match in ASSET_BASE_URL.findall(html_path.read_text(encoding="utf-8")):
            found[match] = found.get(match, 0) + 1

    if not found:
        logger.warning(f"No absolute asset link in {export_dir}, so linked files will carry no url")
        return None

    # One course publishes its assets under one prefix, so the most common
    # is the right one even if a stale link survives somewhere
    base_url = max(found, key=lambda name: found[name])
    logger.info(f"Asset base url: {base_url}")

    return base_url


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


# One course links the same asset from several pages, so each url is only
# ever asked about once
_url_exists_cache: dict[str, bool] = {}


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
