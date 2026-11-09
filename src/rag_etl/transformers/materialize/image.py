"""Image materialization: an image is read in place — a natural Merkle leaf."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from ragtools.images import SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_bytes
from rag_etl.transformers.materialize.ocr import ocr_cached

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageMaterializer:
    """Turns one image into its own transcript, in place."""

    cache: Cache
    work_root: Path

    def materialize(self, node: Resource) -> None:
        image_path = node.path
        if image_path is None:
            return

        work_dir = self.work_root / hash_bytes(image_path.read_bytes())
        md_path = ocr_cached(
            self.cache, image_path, SLIDE_SYSTEM_PROMPT, SLIDE_USER_PROMPT, work_dir / f"{image_path.stem}.md"
        )

        if not md_path.read_text(encoding="utf-8").strip():
            logger.info(f"Dropping {node.title}: no readable content on it")
            node.detach()
            return

        node.path = md_path
        node.mime_type = mt.MARKDOWN
