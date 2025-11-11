from __future__ import annotations

from typing import List, Tuple
from pathlib import Path

import logging

from rag_etl.resources import MoodleResource
from rag_etl.extractors import BaseExtractor

from rag_etl.extractors.moodle.moodle_parser import parse_index, resolve_resource


class MoodleExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from Moodle.
    """

    def __init__(
        self,
        moodle_dump_path: str,
        allowed_href_prefixes: Tuple[str] = ("./File",),
    ) -> None:
        self.moodle_dump_path = Path(moodle_dump_path)
        self.allowed_href_prefixes = allowed_href_prefixes

    def extract(self) -> List[MoodleResource]:
        """
        Extract resources for this course from Moodle.

        Returns:
            List[MoodleResource]: List of raw Resources.
        """

        logging.debug(f"Extracting Moodle resources from {self.moodle_dump_path}")

        # Parse index file
        resources = parse_index(self.moodle_dump_path, self.allowed_href_prefixes)

        # Replace local references with actual Moodle urls
        resources = [resolve_resource(resource, self.moodle_dump_path) for resource in resources]

        # Convert dicts to MoodleResource instances
        resources = [
            MoodleResource(
                section_title=resource['section_title'],
                section_text=resource['section_text'],
                title=resource['title'],
                url=resource['url'],
                path=resource['path'],
                source='moodle',
                mime_type=resource['mime_type'],
            )
            for resource in resources
        ]

        logging.debug(f"Extracted {len(resources)} Moodle resources from {self.moodle_dump_path}")

        return resources
