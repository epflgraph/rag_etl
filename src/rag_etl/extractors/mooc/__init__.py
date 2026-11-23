"""The MOOC extractor: an edX export, read into a resource tree."""

from dataclasses import dataclass

from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor
from rag_etl.extractors.mooc.course import parse_course
from rag_etl.extractors.mooc.export import asset_base_url_from_course_url

__all__ = ["MOOCExtractor"]


@dataclass(frozen=True)
class MOOCExtractor(Extractor):
    """
    Extractor for retrieving course materials from an exported MOOC.

    One container per chapter, nested down to the item leaves a student
    meets: every page, every document a page links (except beside a video),
    every quiz and every video, bar what is marked NO_BOT. Leaves carry the
    facts the export states; what the material is, the judge decides.
    """

    # The MOOC's language, as an ISO code ("fr", "en"): which of the subtitle
    # tracks shipped with the export is kept for each video
    language: str | None = None
    course_url: str | None = None
    # Which chapter opens week one, and how many weeks follow it. The
    # chapters before and after those carry no week
    first_week_chapter: int | None = None
    week_count: int | None = None

    def extract(self) -> Resource:
        """Extract the MOOC's material into a resource tree."""

        # An export does not always reveal where its files are published. The
        # address of the course does, since its files live on the same host
        # under the same key, so that is what a project states
        asset_base_url = asset_base_url_from_course_url(self.course_url) if self.course_url else None

        return parse_course(
            export_dir=self.dir,
            course_url=self.course_url,
            language=self.language,
            asset_base_url=asset_base_url,
            first_week_chapter=self.first_week_chapter,
            week_count=self.week_count,
        )
