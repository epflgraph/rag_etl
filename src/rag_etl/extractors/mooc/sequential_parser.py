import logging
from collections.abc import Mapping
from pathlib import Path
from lxml.etree import _Element
from typing import Any

from rag_etl.extractors.mooc.vertical_parser import VerticalParser
from rag_etl.extractors.mooc.utils import UntaggedDocuments
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
        tag_metadata: Mapping[str, Mapping[str, Any]],
        asset_base_url: str | None = None,
        untagged_documents: UntaggedDocuments | None = None,
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
                untagged_documents=untagged_documents,
                week=week,
                tag_metadata=tag_metadata,
                language=language,
            )
            if vertical is not None:
                sequential.add_child(vertical)

        return sequential if sequential.children else None
