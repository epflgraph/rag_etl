import logging
from collections.abc import Sequence
from pathlib import Path

from rag_etl.resources import BaseResource
from rag_etl.transformers.base_transformer import BaseTransformer
from rag_etl.transformers.image_to_md.utils import convert_image_to_md
import rag_etl.utils.mime_types as mt

logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES = [mt.JPEG, mt.PNG]


class ImageToMarkdownTransformer(BaseTransformer):
    """
    Transformer that converts image resources into Markdown resources.

    Non-image resources as well as resources not matching the specified
    type_subtypes are left unchanged.
    """

    def __init__(self, type_subtypes=None, mime_types: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.type_subtypes = type_subtypes

        if mime_types is None:
            self.mime_types = IMAGE_MIME_TYPES
        else:
            self.mime_types = mime_types

    # Sequence: no mutation
    def transform(self, resources: Sequence[BaseResource]) -> list[BaseResource]:
        """Convert image resources into Markdown text."""

        transformed_resources: list[BaseResource] = []

        for resource in resources:
            # Skip if resource is not in the specified list of types and subtypes
            if self.type_subtypes is not None and (resource.type, resource.subtype) not in self.type_subtypes:
                transformed_resources.append(resource)
                continue

            # Skip if resource is not an image
            if resource.mime_type not in self.mime_types:
                transformed_resources.append(resource)
                continue

            image_path = Path(resource.path)
            md_path = image_path.with_suffix(".md")

            # Convert if not cached
            cached = self.get_from_cache(image_path, md_path)
            if not cached:
                logger.debug(f"Converting {resource.path} into {md_path.name}")
                convert_image_to_md(image_path, md_path)
                self.set_to_cache(image_path, md_path)

            # Drop images with no content
            if not md_path.read_text(encoding="utf-8").strip():
                logger.info(f"Dropping {resource.title}: no readable content on the slide")
                continue

            transformed_resources.append(
                resource.copy_with(
                    path=str(md_path),
                    mime_type=mt.MARKDOWN,
                )
            )

        return transformed_resources
