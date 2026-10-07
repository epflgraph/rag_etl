import logging
from collections.abc import Mapping
from pathlib import Path
from lxml.etree import _Element
from typing import Any

from rag_etl.extractors.mooc.video_parser import VideoParser
from rag_etl.extractors.mooc.html_parser import HtmlParser
from rag_etl.extractors.mooc.quiz_parser import QuizParser
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

from rag_etl.extractors.mooc.utils import UntaggedDocuments
from rag_etl.core import Resource

logger = logging.getLogger(__name__)


class VerticalParser:
    """
    Vertical Parser for MOOCs.
    """

    def parse(
        self,
        course_path: Path,
        elem_vertical: _Element,
        assets_map: dict[str, str],
        tag_metadata: Mapping[str, Mapping[str, Any]],
        asset_base_url: str | None = None,
        untagged_documents: UntaggedDocuments | None = None,
        week: int | None = None,
        language: str | None = None,
    ) -> Resource | None:
        """Parse a MOOC vertical into a container of its item leaves"""

        vertical_url_name = elem_vertical.get("url_name")
        vertical_filename = vertical_url_name + ".xml"

        vertical_xml_path = Path(course_path) / "vertical" / vertical_filename

        root_vertical = load_root_elem_from_mooc_xml(vertical_xml_path)
        if root_vertical is None:
            return None

        vertical_display_name = root_vertical.get("display_name", " ")

        # A page that holds a video already has its slides indexed frame by
        # frame, each linking to its own moment, so any document it also
        # links is that same deck a second time
        vertical_has_video = any(child.tag == "video" for child in root_vertical.iterchildren())

        vertical = Resource(title=vertical_display_name)
        html_parser = HtmlParser()
        quiz_parser = QuizParser()
        video_parser = VideoParser()

        for child in root_vertical.iterchildren():
            # Parse HTML files
            if child.tag == "html":
                html_extracted_resources = html_parser.parse(
                    course_path=course_path,
                    elem_vertical=child,
                    vertical_display_name=vertical_display_name,
                    assets_map=assets_map,
                    asset_base_url=asset_base_url,
                    untagged_documents=untagged_documents,
                    vertical_has_video=vertical_has_video,
                    tag_metadata=tag_metadata,
                    week=week,
                )
                # Add the returned resources as leaves
                if html_extracted_resources is not None:
                    for resource in html_extracted_resources:
                        vertical.add_child(resource)

            # Parse quizzes
            elif child.tag == "problem":
                quiz_extracted_resources = quiz_parser.parse(
                    course_path=course_path,
                    elem_vertical=child,
                    vertical_display_name=vertical_display_name,
                    tag_metadata=tag_metadata,
                    week=week,
                )
                # Add the returned resources as leaves
                if quiz_extracted_resources is not None:
                    for resource in quiz_extracted_resources:
                        vertical.add_child(resource)

            # Parse videos
            elif child.tag == "video":
                video_resource = video_parser.parse(
                    course_path=course_path,
                    elem_vertical=child,
                    vertical_display_name=vertical_display_name,
                    tag_metadata=tag_metadata,
                    language=language,
                    week=week,
                )
                # Add the returned resource as a leaf
                if video_resource is not None:
                    vertical.add_child(video_resource)

        return vertical if vertical.children else None
