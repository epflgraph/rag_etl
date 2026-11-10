"""The structure of an edX export: course → chapters → sequentials → verticals → items.

Each level is a block described in its own file and listing the blocks it
holds; the walk below follows that nesting down to the items (html pages,
problems, videos), which their own parsers turn into leaves. A container
left with nothing in it is dropped.
"""

import json
import logging
import re
from pathlib import Path

from lxml.etree import _Element

from rag_etl.core import Resource
from rag_etl.extractors.mooc.html_parser import HtmlParser
from rag_etl.extractors.mooc.quiz_parser import QuizParser
from rag_etl.extractors.mooc.utils import Export, cmp_key, load_root_elem_from_mooc_xml
from rag_etl.extractors.mooc.video_parser import VideoParser

logger = logging.getLogger(__name__)

# The part of an edX asset link that is the same for every asset of a course,
# as in "https://courses.edx.org/asset-v1:EPFLx+init-prog-cpp+1T2025+type@asset+"
ASSET_BASE_URL = re.compile(r"https://[^\"\']*?type@asset\+")


def parse_course(
    export_dir: Path,
    course_url: str | None = None,
    language: str | None = None,
    asset_base_url: str | None = None,
    first_week_chapter: int | None = None,
    week_count: int | None = None,
) -> Resource:
    """Parse a MOOC export into a resource tree: one container per chapter."""

    course = _course_element(export_dir)

    # A stated prefix wins over one read back from the pages
    export = Export(
        dir=export_dir,
        assets_map=load_assets_map(export_dir),
        asset_base_url=asset_base_url or load_asset_base_url(export_dir),
        language=language,
    )

    root = Resource(title=(course.get("display_name") if course is not None else None) or "", url=course_url)

    # Chapters are taken in course order rather than in the order the
    # filesystem lists them, because a chapter's place in the course is
    # what says which week it is
    chapters = chapter_weeks(course, first_week_chapter, week_count)
    for url_name, week in chapters:
        logger.debug(f"chapter {url_name} is week {week}")

    weeks = [week for _, week in chapters if week is not None]
    logger.info(f"{len(chapters)} chapters, {len(weeks)} of them weeks {min(weeks, default=0)}-{max(weeks, default=0)}")

    for url_name, week in chapters:
        if (chapter := _chapter(export, url_name, week)) is not None:
            root.add_child(chapter)

    return root


def _chapter(export: Export, url_name: str, week: int | None) -> Resource | None:
    """A chapter: a container of its sequentials, all of one week."""

    element = export.load("chapter", url_name)
    if element is None:
        return None

    chapter = Resource(title=element.get("display_name", " "), week=week)
    for child in element.iterchildren():
        if (sequential := _sequential(export, child.get("url_name"), week)) is not None:
            chapter.add_child(sequential)

    return chapter if chapter.children else None


def _sequential(export: Export, url_name: str, week: int | None) -> Resource | None:
    """A sequential (a subsection): a container of its verticals."""

    element = export.load("sequential", url_name)
    if element is None:
        return None

    sequential = Resource(title=element.get("display_name", " "))
    for child in element.iterchildren():
        if (vertical := _vertical(export, child.get("url_name"), week)) is not None:
            sequential.add_child(vertical)

    return sequential if sequential.children else None


def _vertical(export: Export, url_name: str, week: int | None) -> Resource | None:
    """A vertical (a unit, one page a student sees): a container of its items' leaves."""

    element = export.load("vertical", url_name)
    if element is None:
        return None

    title = element.get("display_name", " ")

    # A page that holds a video already has its slides indexed frame by
    # frame, each linking to its own moment, so any document it also
    # links is that same deck a second time
    has_video = any(child.tag == "video" for child in element.iterchildren())

    vertical = Resource(title=title)
    for child in element.iterchildren():
        for leaf in _item(export, child, title, has_video, week):
            vertical.add_child(leaf)

    return vertical if vertical.children else None


def _item(export: Export, element: _Element, vertical_title: str, has_video: bool, week: int | None) -> list[Resource]:
    """The leaves one item of a vertical yields; an item of any other kind yields none."""

    if element.tag == "html":
        return HtmlParser().parse(export, element, vertical_title, has_video, week)
    if element.tag == "problem":
        return QuizParser().parse(export, element, vertical_title, week)
    if element.tag == "video":
        video = VideoParser().parse(export, element, vertical_title, week)
        return [video] if video is not None else []
    return []


def load_assets_map(export_dir: Path) -> dict[str, str]:
    """Read policies/assets.json: the comparison key of each asset's name -> its file."""

    data = json.loads((export_dir / "policies" / "assets.json").read_text(encoding="utf-8"))

    assets: dict[str, str] = {}
    for asset_key, value in data.items():
        assets[cmp_key(asset_key)] = value.get("import_path") or value.get("displayname") or asset_key

    return assets


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


def _course_element(export_dir: Path) -> _Element | None:
    """The course's own block, course/<run>.xml, or None (logged)."""

    course_xml_path = next(iter((export_dir / "course").glob("*.xml")), None)
    if course_xml_path is None:
        logger.warning(f"No course file under {export_dir}, so it has no title and no resource will carry a week")
        return None

    return load_root_elem_from_mooc_xml(course_xml_path)


def chapter_weeks(
    course: _Element | None, first_week_chapter: int | None, week_count: int | None
) -> list[tuple[str, int | None]]:
    """
    Return the course's chapters in the order a student meets them, each
    with the week it belongs to.

    A MOOC is one chapter per week, but not every chapter is a week: a
    preamble opens the course and a feedback survey or a closing note ends
    it, and none of those is material of any week. The course says which
    chapter opens week one and how many weeks follow, since the chapters
    themselves are titled by subject and carry no number.
    """

    if course is None:
        return []

    chapters = []
    position = 0
    for child in course.iterchildren():
        # A course holds more than its chapters, a wiki among them
        if child.tag != "chapter":
            continue

        position += 1
        week = None

        if first_week_chapter is not None and position >= first_week_chapter:
            candidate = position - first_week_chapter + 1
            if week_count is None or candidate <= week_count:
                week = candidate

        chapters.append((child.get("url_name", ""), week))

    return chapters
