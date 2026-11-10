import logging
from pathlib import Path
from lxml.etree import _Element

from rag_etl.extractors.mooc.vertical_parser import VerticalParser
from rag_etl.core import Resource
from rag_etl.extractors.mooc.utils import load_root_elem_from_mooc_xml

logger = logging.getLogger(__name__)


class SequentialParser:
    """
    Sequential Parser for MOOCs.
    """

    def parse(
        self,
        course_path: Path,
        elem_sequential: _Element,
        assets_map: dict[str, str],
        asset_base_url: str | None = None,
        week: int | None = None,
        language: str | None = None,
    ) -> Resource | None:
        """Parse a MOOC sequential into a container of its verticals"""

        sequential_url_name = elem_sequential.get("url_name")
        sequential_filename = sequential_url_name + ".xml"

        sequential_xml_path = Path(course_path) / "sequential" / sequential_filename

        root_sequential = load_root_elem_from_mooc_xml(sequential_xml_path)
        if root_sequential is None:
            return None

        sequential_display_name = root_sequential.get("display_name")
        logger.debug(f"    sequential {sequential_display_name}")

        sequential = Resource(title=sequential_display_name)
        vertical_parser = VerticalParser()

        # Parse verticals
        for elem_vertical in root_sequential.iterchildren():
            vertical = vertical_parser.parse(
                course_path=course_path,
                elem_vertical=elem_vertical,
                assets_map=assets_map,
                asset_base_url=asset_base_url,
                week=week,
                language=language,
            )
            if vertical is not None:
                sequential.add_child(vertical)

        return sequential if sequential.children else None
