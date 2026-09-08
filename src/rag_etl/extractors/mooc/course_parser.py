import logging
from pathlib import Path
from rag_etl.extractors.mooc.chapter_parser import ChapterParser
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import UntaggedDocuments, cmp_key, load_root_elem_from_mooc_xml
import json
import re
import unicodedata


logger = logging.getLogger(__name__)

# The part of an edX asset link that is the same for every asset of a course,
# as in "https://courses.edx.org/asset-v1:EPFLx+init-prog-cpp+1T2025+type@asset+"
ASSET_BASE_URL = re.compile(r"https://[^\"\']*?type@asset\+")


class CourseParser:
    """
    MOOC Parser.
    """

    def load_assets_map(self, course_path: str) -> dict[str, str]:

        # assets.json is always in the same path
        assets_path = Path(course_path) / "policies" / "assets.json"
        data = json.loads(assets_path.read_text(encoding="utf-8"))

        m: dict[str, str] = {}
        for asset_key, value in data.items():
            # Get import_path if present
            file_path = value.get("import_path") or value.get("displayname") or asset_key

            # Normalize key and store file_path in dictionary
            m[cmp_key(asset_key)] = file_path

        return m

    def load_asset_base_url(self, course_path: str) -> str | None:
        """
        Return the prefix an asset of this course is published under, or None.

        The export's own course key is not usable: course.xml and assets.json
        are rewritten on re-export and name a course that does not exist
        publicly. The absolute links the authors wrote into the pages do name
        the published run, so the prefix is read back from those.
        """

        found: dict[str, int] = {}
        for html_path in (Path(course_path) / "html").glob("*.html"):
            for match in ASSET_BASE_URL.findall(html_path.read_text(encoding="utf-8")):
                found[match] = found.get(match, 0) + 1

        if not found:
            logger.warning(f"No absolute asset link in {course_path}, so linked files will carry no url")
            return None

        # One course publishes its assets under one prefix, so the most common
        # is the right one even if a stale link survives somewhere
        base_url = max(found, key=found.get)
        logger.info(f"Asset base url: {base_url}")

        return base_url

    def load_chapter_weeks(
        self, course_path: str, first_week_chapter: int | None, week_count: int | None
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

        course_xml_path = next(iter((Path(course_path) / "course").glob("*.xml")), None)
        if course_xml_path is None:
            logger.warning(f"No course file under {course_path}, so no resource will carry a week")
            return []

        root_course = load_root_elem_from_mooc_xml(course_xml_path)
        if root_course is None:
            return []

        chapters = []
        position = 0
        for child in root_course.iterchildren():
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

    def parse(
        self,
        course_path: str,
        tag_metadata: dict | None = None,
        language: str | None = None,
        untagged_documents: UntaggedDocuments | None = None,
        asset_base_url: str | None = None,
        first_week_chapter: int | None = None,
        week_count: int | None = None,
    ) -> list[MOOCResource]:
        """Parse a MOOC course"""

        # Load policies/assets.json with url_name to path mapping
        assets_map: dict[str, str] = self.load_assets_map(course_path)
        # A stated prefix wins over one read back from the pages
        if not asset_base_url:
            asset_base_url = self.load_asset_base_url(course_path)

        items: list[MOOCResource] = []

        chapter_parser = ChapterParser()

        # Chapters are taken in course order rather than in the order the
        # filesystem lists them, because a chapter's place in the course is
        # what says which week it is
        chapters = self.load_chapter_weeks(course_path, first_week_chapter, week_count)
        for chapter_url_name, week in chapters:
            logger.debug(f"chapter {chapter_url_name} is week {week}")

        weeks = [week for _, week in chapters if week is not None]
        logger.info(f"{len(chapters)} chapters, {len(weeks)} of them weeks {min(weeks, default=0)}-{max(weeks, default=0)}")

        # For each one of the chapters
        for chapter_url_name, week in chapters:
            items.extend(
                chapter_parser.parse(
                    course_path=course_path,
                    chapter_filename=f"{chapter_url_name}.xml",
                    week=week,
                    assets_map=assets_map,
                    asset_base_url=asset_base_url,
                    untagged_documents=untagged_documents,
                    tag_metadata=tag_metadata,
                    language=language,
                )
            )

        for item in items:
            logger.debug(f"item={item}")

        return items
