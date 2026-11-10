"""Markdown materialization: the local images a markdown file shows are read in place.

A reference `![alt](figure.png)` (or `![alt](<path with spaces.png>)`) to
an image on disk, relative to the markdown file or absolute, is replaced
by the image's transcript, where the image was. Images share the image
materializer's cache scope: an image is read once, wherever it is shown.
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import ragtools.mime_types as mt
from ragtools.images import transcribe_slide

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_bytes
from rag_etl.transformers.materialize.image import CACHE_VERSION
from rag_etl.transformers.materialize.ocr import ocr_cached

logger = logging.getLogger(__name__)

# ![alt](target) or ![alt](<target>), an optional "title" after the target
IMAGE_REFERENCE = re.compile(r'!\[[^\]]*\]\((?:<([^>]+)>|([^)\s]+))(?:\s+"[^"]*")?\)')

# Images that are read, as for image files of their own
READABLE_IMAGES = (mt.JPEG, mt.PNG)


@dataclass(frozen=True)
class MarkdownMaterializer:
    """Replaces the local images a markdown file references by their transcripts."""

    cache: Cache
    work_root: Path

    def materialize(self, node: Resource) -> None:
        md_path = node.path
        if md_path is None:
            return

        text = md_path.read_text(encoding="utf-8")
        work_dir = self.work_root / hash_bytes(text.encode("utf-8"))
        read_any = False

        def transcript(match: re.Match[str]) -> str:
            nonlocal read_any
            target = match.group(1) or match.group(2)

            # Web images and embedded data are not files the material ships
            if re.match(r"[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                return match.group(0)

            image_path = (md_path.parent / unquote(target)).resolve()
            if mt.guess_mime_type(str(image_path)) not in READABLE_IMAGES:
                return match.group(0)

            # A folder may ship a markdown file without the images it shows:
            # the text is still worth indexing, the reference stays as written
            if not image_path.is_file():
                logger.warning(f"{node.title}: image {target} not found, left as a reference")
                return match.group(0)

            read_any = True
            image_md = ocr_cached(
                self.cache.scope(CACHE_VERSION), image_path, transcribe_slide, work_dir / "images" / f"{image_path.stem}.md"
            )
            return image_md.read_text(encoding="utf-8").strip()

        materialized = IMAGE_REFERENCE.sub(transcript, text)
        if not read_any:
            return

        # The source file stays as it is: the materialized text is scratch
        out_path = work_dir / md_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(materialized, encoding="utf-8")
        node.path = out_path
