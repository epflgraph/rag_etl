from pathlib import Path

from rag_etl.resources import MOOCResource
from rag_etl.extractors import BaseExtractor

import rag_etl.utils.mime_types as mt
from rag_etl.extractors.mooc.course_parser import CourseParser
from rag_etl.extractors.mooc.utils import UntaggedDocuments, asset_base_url_from_course_url


class MOOCExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from an exported MOOC.
    """

    def __init__(
        self,
        mooc_base_path: str,
        tag_metadata: dict | None = None,
        mime_types: list[str] | None = None,
        language: str | None = None,
        include_untagged_documents: bool = False,
        theory_tag: str | None = None,
        theory_slides_tag: str | None = None,
        untagged_extensions: tuple[str, ...] = (".pdf", ".txt", ".zip"),
        course_url: str | None = None,
    ) -> None:
        # Both tags are needed to file an untagged document, so a course that
        # asks for them without saying where they go is refused here rather
        # than silently indexing everything as one kind
        if include_untagged_documents and not (theory_tag and theory_slides_tag):
            raise ValueError("include_untagged_documents needs both theory_tag and theory_slides_tag")

        self.untagged_documents = UntaggedDocuments(
            include=include_untagged_documents,
            theory_tag=theory_tag,
            theory_slides_tag=theory_slides_tag,
            extensions=untagged_extensions,
        )

        # An export does not always reveal where its files are published. The
        # address of the course does, since its files live on the same host
        # under the same key, so that is what a course states
        if course_url:
            self.asset_base_url = asset_base_url_from_course_url(course_url)
        else:
            self.asset_base_url = None

        self.mooc_base_path = mooc_base_path
        self.tag_metadata = tag_metadata
        # Selects which of the subtitle tracks shipped with the export is kept
        # for each video. Accepts a name ("French") or a code ("fr").
        self.language = language
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
            language=self.language,
            untagged_documents=self.untagged_documents,
            asset_base_url=self.asset_base_url,
        )
