import logging
from pathlib import Path
from lxml.etree import _Element
from rag_etl.extractors.mooc.vertical_parser import VerticalParser
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

logger = logging.getLogger(__name__)


class SequentialParser:
    """
    Sequential Parser for MOOCs.
    """

    def parse(
        self,
        course_path: str,
        elem_sequential: _Element,
        assets_map: dict[str, str],
        tag_metadata: dict | None = None,
    ) -> list[MOOCResource]:
        """Parse a MOOC sequential"""

        sequential_url_name = elem_sequential.get("url_name")
        sequential_filename = sequential_url_name + ".xml"

        sequential_xml_path = Path(course_path) / "sequential" / sequential_filename

        root_sequential = load_root_elem_from_mooc_xml(sequential_xml_path)
        if root_sequential is None:
            return []

        sequential_display_name = root_sequential.get("display_name")
        logger.debug(f"    sequential {sequential_display_name}")

        items: list[MOOCResource] = []
        vertical_parser = VerticalParser()

        # Parse verticals
        for elem_vertical in root_sequential.iterchildren():
            items.extend(
                vertical_parser.parse(
                    course_path=course_path,
                    elem_vertical=elem_vertical,
                    assets_map=assets_map,
                    tag_metadata=tag_metadata,
                )
            )

        return items
