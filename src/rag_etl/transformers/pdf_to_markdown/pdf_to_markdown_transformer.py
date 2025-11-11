from __future__ import annotations

from typing import List, Sequence
from pathlib import Path

import logging

from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource

from rag_etl.transformers.pdf_to_markdown.utils import convert_pdf_to_md


class PDFToMarkdownTransformer(BaseTransformer):
    """
    Transformer that converts PDF resources into Markdown resources.

    Non-PDF resources as well as resources not matching the specified type_subtypes are left unchanged.
    """

    def __init__(self, type_subtypes=None) -> None:
        self.type_subtypes = type_subtypes

    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """
        Convert PDF resources into Markdown text.

        Non-PDF resources as well as resources not matching the specified type_subtypes are left unchanged.
        """

        transformed_resources: List[BaseResource] = []

        for resource in resources:
            # Skip if resource is not in the specified list of types and subtypes
            if self.type_subtypes and (resource.type, resource.subtype) not in self.type_subtypes:
                transformed_resources.append(resource)
                continue

            # Skip if resource is not a PDF
            if resource.mime_type != "application/pdf":
                transformed_resources.append(resource)
                continue

            # Build paths of PDF file and md file
            pdf_path = Path(resource.path)
            md_path = pdf_path.with_suffix(".md")

            # Only convert if not cached
            if not md_path.exists():
                logging.debug(f"Converting {resource.path} → {md_path.name}")
                convert_pdf_to_md(pdf_path, md_path)

            # Build transformed resource and append it
            new_resource = resource.copy_with(
                path=str(md_path),
                mime_type="text/markdown",
                processing_method=None,
            )
            transformed_resources.append(new_resource)

        logging.debug(f"Transformed {len(transformed_resources)} resources in total")

        return transformed_resources
