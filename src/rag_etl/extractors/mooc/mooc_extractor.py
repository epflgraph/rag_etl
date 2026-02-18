from pathlib import Path

from rag_etl.resources import MOOCResource
from rag_etl.extractors import BaseExtractor

import rag_etl.utils.mime_types as mt
from rag_etl.extractors.mooc.course_parser import CourseParser


class MOOCExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from an exported MOOC.
    """

    def __init__(
        self,
        mooc_base_path: str,
        tag_metadata: dict | None = None,
        mime_types: list[str] | None = None,
    ) -> None:
        self.mooc_base_path = Path(mooc_base_path)
        self.tag_metadata = tag_metadata
        if mime_types is None:
            self.mime_types = mt.DEFAULT_MIME_TYPES
        else:
            self.mime_types = mime_types

    def extract(self) -> list[MOOCResource]:
        """
        Extract resources for this MOOC.
        """

        course_parser = CourseParser()
        return course_parser.parse(
            course_path=self.mooc_base_path,
            tag_metadata=self.tag_metadata,
        )
