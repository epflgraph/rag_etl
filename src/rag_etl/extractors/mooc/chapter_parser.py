import logging
from pathlib import Path

from rag_etl.extractors.mooc.sequential_parser import SequentialParser
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

logger = logging.getLogger(__name__)


class ChapterParser:
    """
    Chapter Parser for MOOCs.
    """

    def parse(
        self,
        course_path: str,
        chapter_filename: str,
        assets_map: dict[str, str],
        tag_metadata: dict | None = None,
        language: str | None = None,
    ) -> list[MOOCResource]:
        """Parse a MOOC chapter"""

        chapter_xml_path = Path(course_path) / "chapter" / chapter_filename

        root_chapter = load_root_elem_from_mooc_xml(chapter_xml_path)
        if root_chapter is None:
            return []

        chapter_display_name = root_chapter.get("display_name", " ")
        logger.debug(f"  chapter {chapter_display_name}")

        items: list[MOOCResource] = []
        sequential_parser = SequentialParser()

        # Parse sequentials
        for elem_sequential in root_chapter.iterchildren():
            items.extend(
                sequential_parser.parse(
                    elem_sequential=elem_sequential,
                    course_path=course_path,
                    tag_metadata=tag_metadata,
                    language=language,
                    assets_map=assets_map,
                )
            )

        return items
