import logging
from pathlib import Path
from rag_etl.extractors.mooc.chapter_parser import ChapterParser
from rag_etl.resources.mooc_resource import MOOCResource
import os

logger = logging.getLogger(__name__)


class CourseParser:
    """
    MOOC Parser.
    """

    def parse(
        self,
        course_path: str,
    ) -> list[MOOCResource]:
        """Parse a MOOC course"""

        items: list[MOOCResource] = []

        chapter_path = Path(course_path) / "chapter"
        chapter_parser = ChapterParser()

        # For each one of the chapters
        for chapter_filename in os.listdir(chapter_path):
            items.extend(
                chapter_parser.parse(
                    course_path=course_path,
                    chapter_filename=chapter_filename,
                )
            )

        for item in items:
            logger.debug(f"item={item}")

        return items
