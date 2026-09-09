from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import logging
import re

from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource

import rag_etl.utils.mime_types as mt


PAGE_MARKER_PATTERN = re.compile(r"<!-- page (\d+) -->\s*")


def split_md_into_pages(md_text: str) -> list[tuple[int, str]]:
    """
    Split a Markdown document by `<!-- page N -->` markers.

    Returns a list of (page_number, page_content) tuples. Any content before the
    first marker is prepended to page 1.
    """
    parts = PAGE_MARKER_PATTERN.split(md_text)

    pages: list[tuple[int, str]] = []
    prefix = parts[0].strip()

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if i == 1 and prefix:
            content = f"{prefix}\n\n{content}".strip()
        pages.append((page_number, content))

    return pages


class SplitPagesTransformer(BaseTransformer):
    """
    Transformer that splits Markdown resources into one resource per page.

    Only Markdown resources are considered. Any PDF should first be converted to Markdown.
    The Markdown is expected to contain `<!-- page N -->` markers.
    """

    def __init__(self, type_subtypes=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.type_subtypes = type_subtypes

    def transform(self, resources: Sequence[BaseResource]) -> list[BaseResource]:
        """
        Splits Markdown resources into one resource per page.
        Non-Markdown resources as well as resources not matching the specified type_subtypes are left unchanged.
        """

        transformed_resources: list[BaseResource] = []

        for resource in resources:
            # Skip if resource is not in the specified list of types and subtypes
            if self.type_subtypes is not None and (resource.type, resource.subtype) not in self.type_subtypes:
                transformed_resources.append(resource)
                continue

            # Skip if resource is not Markdown
            if resource.mime_type != mt.MARKDOWN:
                transformed_resources.append(resource)
                continue

            # Build paths of md file and pages folder
            md_path = Path(resource.path)
            pages_dir = md_path.with_suffix("").with_name(f"{md_path.stem}_pages")

            # Only split if not cached
            cached = self.get_from_cache(md_path, pages_dir)
            if not cached:
                logging.debug(f"Splitting {resource.path} into pages")
                md_text = md_path.read_text(encoding="utf-8")
                pages = split_md_into_pages(md_text)
                pages_dir.mkdir(parents=True, exist_ok=True)
                for page_number, content in pages:
                    page_path = pages_dir / f"page_{page_number}.md"
                    page_path.write_text(content, encoding="utf-8")
                self.set_to_cache(md_path, pages_dir)

            # Build resource for each page file
            page_paths = sorted(pages_dir.glob("page_*.md"))
            if not page_paths:
                # No pages could be extracted; keep the original resource unchanged
                transformed_resources.append(resource)
                continue

            for page_md_path in page_paths:
                page_number = int(page_md_path.stem.split("_")[1])
                page_url = f"{resource.url}#page={page_number}" if resource.url else None

                new_resource = resource.copy_with(
                    title=f"{resource.title} (page {page_number})",
                    path=str(page_md_path),
                    url=page_url,
                    processing_method=None,
                    model=None,
                )
                transformed_resources.append(new_resource)

        return transformed_resources
