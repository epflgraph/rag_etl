import logging
from pathlib import Path
from lxml.etree import _Element

from rag_etl.extractors.mooc.video_parser import VideoParser
from rag_etl.extractors.mooc.html_parser import HtmlParser
from rag_etl.extractors.mooc.quiz_parser import QuizParser
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

from rag_etl.resources.mooc_resource import MOOCResource

logger = logging.getLogger(__name__)


class VerticalParser:
    """
    Vertical Parser for MOOCs.
    """

    def parse(
        self,
        course_path: str,
        elem_vertical: _Element,
        tag_metadata: dict | None = None,
    ) -> list[MOOCResource]:
        """Parse a MOOC vertical"""

        vertical_url_name = elem_vertical.get("url_name")
        vertical_filename = vertical_url_name + ".xml"

        vertical_xml_path = Path(course_path) / "vertical" / vertical_filename

        root_vertical = load_root_elem_from_mooc_xml(vertical_xml_path)
        if root_vertical is None:
            return []

        vertical_display_name = root_vertical.get("display_name", " ")

        items: list[MOOCResource] = []
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
                    tag_metadata=tag_metadata,
                )
                # Extend the returned list of resources
                if html_extracted_resources is not None:
                    items.extend(html_extracted_resources)

            # Parse quizzes
            elif child.tag == "problem":
                quiz_extracted_resources = quiz_parser.parse(
                    course_path=course_path,
                    elem_vertical=child,
                    vertical_display_name=vertical_display_name,
                    tag_metadata=tag_metadata,
                )
                # Extend the returned list of resources
                if quiz_extracted_resources is not None:
                    items.extend(quiz_extracted_resources)

            # Parse videos
            elif child.tag == "video":
                video_resource = video_parser.parse(
                    course_path=course_path,
                    elem_vertical=child,
                    vertical_display_name=vertical_display_name,
                    tag_metadata=tag_metadata,
                )
                # Append the returned resource
                if video_resource is not None:
                    items.append(video_resource)

        return items
