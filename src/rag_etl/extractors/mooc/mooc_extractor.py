
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_etl.core import Resource
from rag_etl.extractors.base_extractor import Extractor

import rag_etl.utils.mime_types as mt
from rag_etl.extractors.mooc.course_parser import CourseParser
from rag_etl.extractors.mooc.utils import UntaggedDocuments, asset_base_url_from_course_url


@dataclass(frozen=True)
class MOOCExtractor(Extractor):
    """
    Extractor for retrieving course materials from an exported MOOC.

    One container per chapter, nested down to the item leaves a student
    meets. Leaves carry facts and whatever labels the export itself infers.
    """

    tag_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    mime_types: Sequence[str] = tuple(mt.DEFAULT_MIME_TYPES)
    # Selects which of the subtitle tracks shipped with the export is kept
    # for each video. Accepts a name ("French") or a code ("fr").
    language: str | None = None
    include_untagged_documents: bool = False
    theory_tag: str | None = None
    theory_slides_tag: str | None = None
    untagged_extensions: tuple[str, ...] = (".pdf", ".txt", ".zip")
    course_url: str | None = None
    # Which chapter opens week one, and how many weeks follow it. The
    # chapters before and after those carry no week
    first_week_chapter: int | None = None
    week_count: int | None = None

    def extract(self) -> Resource:
        """
        Extract the MOOC's material into a resource tree.
        """

        # Both tags are needed to file an untagged document, so a course that
        # asks for them without saying where they go is refused here rather
        # than silently indexing everything as one kind
        untagged_documents: UntaggedDocuments | None = None
        if self.include_untagged_documents:
            if not (self.theory_tag and self.theory_slides_tag):
                raise ValueError("include_untagged_documents needs both theory_tag and theory_slides_tag")

            untagged_documents = UntaggedDocuments(
                theory_tag=self.theory_tag,
                theory_slides_tag=self.theory_slides_tag,
                extensions=self.untagged_extensions,
            )

        # An export does not always reveal where its files are published. The
        # address of the course does, since its files live on the same host
        # under the same key, so that is what a project states
        asset_base_url = asset_base_url_from_course_url(self.course_url) if self.course_url else None

        return CourseParser().parse(
            course_path=self.dir,
            course_url=self.course_url,
            tag_metadata=self.tag_metadata,
            language=self.language,
            untagged_documents=untagged_documents,
            asset_base_url=asset_base_url,
            first_week_chapter=self.first_week_chapter,
            week_count=self.week_count,
        )
