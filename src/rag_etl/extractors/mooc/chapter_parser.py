import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_etl.extractors.mooc.sequential_parser import SequentialParser
from rag_etl.extractors.mooc.utils import UntaggedDocuments
from rag_etl.core import Resource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

logger = logging.getLogger(__name__)


class ChapterParser:
    """
    Chapter Parser for MOOCs.
    """

    def parse(
        self,
        course_path: Path,
        chapter_filename: str,
        assets_map: dict[str, str],
        tag_metadata: Mapping[str, Mapping[str, Any]],
        asset_base_url: str | None = None,
        untagged_documents: UntaggedDocuments | None = None,
        week: int | None = None,
        language: str | None = None,
    ) -> Resource | None:
        """Parse a MOOC chapter into a container of its sequentials"""

        chapter_xml_path = Path(course_path) / "chapter" / chapter_filename

        root_chapter = load_root_elem_from_mooc_xml(chapter_xml_path)
        if root_chapter is None:
            return None

        chapter_display_name = root_chapter.get("display_name", " ")
        logger.debug(f"  chapter {chapter_display_name}")

        chapter = Resource(title=chapter_display_name, week=week)
        sequential_parser = SequentialParser()

        # Parse sequentials
        for elem_sequential in root_chapter.iterchildren():
            sequential = sequential_parser.parse(
                elem_sequential=elem_sequential,
                course_path=course_path,
                tag_metadata=tag_metadata,
                language=language,
                assets_map=assets_map,
                asset_base_url=asset_base_url,
                untagged_documents=untagged_documents,
                week=week,
            )
            if sequential is not None:
                chapter.add_child(sequential)

        return chapter if chapter.children else None
