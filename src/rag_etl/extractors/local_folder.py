from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ragtools.files import iter_files
from ragtools.mime_types import DEFAULT_MIME_TYPES, guess_mime_type
from rag_etl.core import Resource
from rag_etl.extractors.base import Extractor, SourceUnavailable
from rag_etl.extractors.markers import NO_BOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalFolderExtractor(Extractor):
    """
    Extractor for retrieving project material from a local folder.

    One leaf per file, with the metadata files (from, until, url) beside it.
    Everything not marked NO_BOT is extracted; the judge labels it later.
    """

    METADATA_FILES = ["from", "until", "url"]

    mime_types: Sequence[str] = tuple(DEFAULT_MIME_TYPES)

    def extract_closest_metadata(self, path, metadata_file):
        # Try to get metadata from current path
        metadata_file_path = path / metadata_file
        if metadata_file_path.exists():
            return metadata_file_path.read_text(encoding="utf-8").strip()

        # If base is a proper subpath of path, we recurse
        if path.is_relative_to(self.dir) and not self.dir.is_relative_to(path):
            return self.extract_closest_metadata(path.parent, metadata_file)

        # Otherwise we stop
        return None

    def extract(self) -> Resource:
        """
        Extract the folder's material as leaves under one container.
        """

        # Check that folder exists, failing loud when it does not
        if not self.dir.exists():
            raise SourceUnavailable(f"Directory {self.dir} does not exist.")

        base = Resource(title=f"Local folder {self.dir.name}")

        # Every file in the folder and its subfolders, hidden ones aside
        for file_path in iter_files(self.dir):
            relative = file_path.relative_to(self.dir)

            # A file under a NO_BOT-marked folder, or marked itself, is not for the bots
            if any(NO_BOT.search(part) for part in relative.parts):
                logger.info(f"Skipping file {str(file_path)} because it is marked NO_BOT.")
                continue

            # Skip if metadata file
            if file_path.name in self.METADATA_FILES:
                logger.info(f"Skipping file {str(file_path)} because we consider it to be metadata.")
                continue

            # Skip if unexpected mime type
            mime_type = guess_mime_type(str(file_path))
            if mime_type not in self.mime_types:
                logger.info(
                    f"Skipping file {str(file_path)} because its mime type ({mime_type}) is not expected ({self.mime_types})."
                )
                continue

            # Extract metadata from files
            from_ = self.extract_closest_metadata(file_path.parent, "from")
            until = self.extract_closest_metadata(file_path.parent, "until")
            url = self.extract_closest_metadata(file_path.parent, "url")

            base.add_child(
                title=str(relative),
                url=url,
                path=file_path,
                mime_type=mime_type,
                from_=from_,
                until=until,
            )

        return base
