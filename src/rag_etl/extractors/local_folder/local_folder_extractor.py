from __future__ import annotations

import os
from pathlib import Path

from typing import List, Optional

import logging

from rag_etl.resources import LocalResource
from rag_etl.extractors import BaseExtractor

import rag_etl.utils.mime_types as mt
from rag_etl.utils.tags import split_tag_number_text

from rag_etl.config import CONFIG


class LocalFolderExtractor(BaseExtractor):
    """
    Extractor for retrieving course materials from a local folder.
    """

    METADATA_FILES = ['from', 'until', 'url']

    def __init__(
        self,
        folder_base_path: str,
        tag_metadata: Optional[dict] = None,
        mime_types: Optional[list[str]] = None,
    ) -> None:
        self.folder_base_path = Path(folder_base_path)

        if tag_metadata:
            self.tag_metadata = tag_metadata
        else:
            self.tag_metadata = {}

        if mime_types is None:
            self.mime_types = mt.DEFAULT_MIME_TYPES
        else:
            self.mime_types = mime_types

    def extract_closest_tag_and_number(self, path):
        # Extract tag and number from current path
        tag, number, _ = split_tag_number_text(path.name)

        # If found, return them
        if tag:
            return (tag, number)

        # If base is a proper subpath of path, we recurse
        if path.is_relative_to(self.folder_base_path) and not self.folder_base_path.is_relative_to(path):
            return self.extract_closest_tag_and_number(path.parent)

        # Otherwise we stop
        return (None, None)

    def extract_closest_metadata(self, path, metadata_file):
        # Try to get metadata from current path
        metadata_file_path = path / metadata_file
        if metadata_file_path.exists():
            return metadata_file_path.read_text(encoding="utf-8").strip()

        # If base is a proper subpath of path, we recurse
        if path.is_relative_to(self.folder_base_path) and not self.folder_base_path.is_relative_to(path):
            return self.extract_closest_metadata(path.parent, metadata_file)

        # Otherwise we stop
        return None

    def extract(self) -> List[LocalResource]:
        """
        Extract resources for this course from a local folder.

        Returns:
            List[BaseResource]: List of raw Resources.
        """

        # Check that folder exists
        if not self.folder_base_path.exists():
            raise ValueError(f"Directory {self.folder_base_path} does not exist.")

        # Iterate over all files in folder and subfolders
        resources = []
        for dir_path, dir_names, file_names in os.walk(self.folder_base_path, topdown=True):
            # Drop hidden directories to prevent descending into them
            dir_names[:] = [d for d in dir_names if not d.startswith(".")]

            for file_name in file_names:
                file_path = Path(dir_path) / file_name

                # Skip if hidden file (e.g. .DS_STORE, .git, .idea, etc.)
                if file_path.name.startswith("."):
                    logging.info(f"Skipping file {str(file_path)} because we consider it to be metadata.")
                    continue

                # Skip if metadata file
                if file_path.name in self.METADATA_FILES:
                    logging.info(f"Skipping file {str(file_path)} because we consider it to be metadata.")
                    continue

                # Skip if unexpected mime type
                mime_type = mt.guess_mime_type(str(file_path))
                if mime_type not in self.mime_types:
                    logging.info(f"Skipping file {str(file_path)} because its mime type ({mime_type}) is not expected ({self.mime_types}).")
                    continue

                # Extract tag from filename
                tag, number = self.extract_closest_tag_and_number(file_path)

                # Skip if no tag
                if not tag:
                    logging.info(f"Skipping file {str(file_path)} because it has no tag.")
                    continue

                # Skip if unrecognised tag
                if tag not in self.tag_metadata:
                    logging.info(f"Skipping file {str(file_path)} because its tag ({tag}) is unexpected ({self.tag_metadata.keys()})")
                    continue

                # Extract tag metadata
                type_ = self.tag_metadata.get(tag, {}).get('type')
                subtype = self.tag_metadata.get(tag, {}).get('subtype')
                is_solution = self.tag_metadata.get(tag, {}).get('is_solution', False)
                one_chunk_per_page = self.tag_metadata.get(tag, {}).get('one_chunk_per_page', False)
                one_chunk_per_doc = self.tag_metadata.get(tag, {}).get('one_chunk_per_doc', False)

                # Extract metadata from files
                from_ = self.extract_closest_metadata(file_path.parent, 'from')
                until = self.extract_closest_metadata(file_path.parent, 'until')
                url = self.extract_closest_metadata(file_path.parent, 'url')

                # Processing method and model
                if mime_type == mt.PDF:
                    processing_method = 'rcp'
                    model = CONFIG['RCP_VISION_MODEL']
                else:
                    processing_method = None
                    model = None

                # Append resource
                resources.append(LocalResource(
                    tag=tag,
                    title=str(file_path.relative_to(self.folder_base_path)),
                    url=url,
                    path=str(file_path),
                    source='local',
                    mime_type=mime_type,
                    type=type_,
                    subtype=subtype,
                    number=number,
                    is_solution=is_solution,
                    processing_method=processing_method,
                    model=model,
                    one_chunk_per_page=one_chunk_per_page,
                    one_chunk_per_doc=one_chunk_per_doc,
                    from_=from_,
                    until=until,
                ))

        return resources
