import logging
from pathlib import Path
from rag_etl.extractors.mooc.chapter_parser import ChapterParser
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import cmp_key
import os
import json
import unicodedata


logger = logging.getLogger(__name__)


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
            file_path = (
                value.get("import_path") or value.get("displayname") or asset_key
            )

            # Normalize key and store file_path in dictionary
            m[cmp_key(asset_key)] = file_path

        return m

    def parse(
        self,
        course_path: str,
        tag_metadata: dict | None = None,
    ) -> list[MOOCResource]:
        """Parse a MOOC course"""

        # Load policies/assets.json with url_name to path mapping
        assets_map: dict[str, str] = self.load_assets_map(course_path)

        items: list[MOOCResource] = []

        chapter_path = Path(course_path) / "chapter"
        chapter_parser = ChapterParser()

        # For each one of the chapters
        for chapter_filename in os.listdir(chapter_path):
            items.extend(
                chapter_parser.parse(
                    course_path=course_path,
                    chapter_filename=chapter_filename,
                    assets_map=assets_map,
                    tag_metadata=tag_metadata,
                )
            )

        for item in items:
            logger.debug(f"item={item}")

        return items
