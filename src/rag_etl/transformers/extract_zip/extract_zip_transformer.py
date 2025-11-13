from __future__ import annotations

from typing import List, Sequence
from pathlib import Path

import logging

import zipfile

from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource

import rag_etl.utils.mime_types as mt


def unzip_file(zip_path):
    """
    Unzips a ZIP archive into a folder named after the zip file (without .zip).

    Args:
        zip_path (str or Path): Path to the zip file.

    Returns:
        Path: Path to the folder where files were extracted.
    """

    zip_path = Path(zip_path)
    extract_dir = zip_path.parent

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def iter_files(root):
    """
    Recursively yields all files under `root`, skipping macOS metadata.
    """
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file():
            # Skip macOS metadata
            parts = set(path.parts)
            if "__MACOSX" in parts or '.ipynb_checkpoints' in parts or path.name == ".DS_Store":
                continue
            yield path


class ExtractZipTransformer(BaseTransformer):
    """
    Transformer that extracts zip resources replacing them with their contents.
    """

    def __init__(self, mime_types=None):
        self.mime_types = mime_types

    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """
        Extract zip resources and replace them with their contents.
        Non-zip resources are returned unchanged.
        """

        transformed_resources: List[BaseResource] = []

        for resource in resources:
            # Skip if resource is not a zip file
            if resource.mime_type != mt.ZIP:
                transformed_resources.append(resource)
                continue

            logging.debug(f"Unzipping {resource.path}")

            # If resource is a zip file, unzip it
            resource_folder = unzip_file(resource.path)

            # Iterate over extracted files and add new resources for each of them
            for extracted_file in iter_files(resource_folder):
                mime_type = mt.guess_mime_type(str(extracted_file))

                # Skip if mime type not in list
                if mime_type not in self.mime_types:
                    continue

                new_resource = resource.copy_with(
                    title=f"{resource.title} > {extracted_file.name}",
                    path=str(extracted_file),
                    mime_type=mime_type,
                )

                logging.debug(f"Appending {new_resource.path}")

                transformed_resources.append(new_resource)

        return transformed_resources
