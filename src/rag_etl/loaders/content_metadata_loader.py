from __future__ import annotations

import shutil
from pathlib import Path

import json

from typing import Sequence

import logging

from rag_etl.loaders import BaseLoader
from rag_etl.resources import BaseResource


class ContentMetadataLoader(BaseLoader):
    """
    Loader that persists the resources as content and metadata folders,
    containing the actual content files and the metadata files respectively.

    The result is ready to be processed by the chatbotpipelines scripts.
    """

    def __init__(self, output_path: str, course_info: dict):
        self.output_path = output_path
        self.course_info = course_info

    def load(self, resources: Sequence[BaseResource]) -> None:
        logging.debug(f"Populating content and metadata folders at {self.output_path}")

        # Create paths and folders if needed
        output_path = Path(self.output_path)
        content_path = output_path / "content"
        metadata_path = output_path / "metadata"

        content_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)

        # Iterate over resources, create content files and keep track of metadata files
        metadata = {}
        for resource in resources:
            # Initialise metadata for the given source
            if resource.source not in metadata:
                metadata[resource.source] = []

            # Build actual location of the content file
            resource_output_path = content_path / Path(resource.path).relative_to(output_path)
            resource_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy actual file
            shutil.copy(resource.path, resource_output_path.parent)

            # Make path relative to base path
            resource.path = str(content_path.relative_to(output_path) / Path(resource.path).relative_to(output_path))

            # Store metadata
            metadata[resource.source].append(resource.metadata_dict())

        for source, source_metadata in metadata.items():
            # Build path for the metadata file for this source
            source_metadata_path = (metadata_path / source).with_suffix('.json')

            # Build full metadata in final format
            full_source_metadata = {
                "course_info": self.course_info,
                "documents": source_metadata
            }

            # Write full metadata into metadata file
            source_metadata_path.write_text(json.dumps(full_source_metadata, ensure_ascii=False, indent=2))
