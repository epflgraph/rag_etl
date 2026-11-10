"""Materialize: files that hold text in pixels are turned into text.

Every PDF page is rendered deterministically and read by a vision model,
one child per page; images are read the same way; videos are turned into
slides — slide changes are detected, one frame per slide segment is
extracted straight from the video's url, every frame is read, and the
words spoken over the segment are appended to it.

Everything read by a model is cached as a Merkle tree: the transcripts
are leaves stored once under their own content hash, and each pdf and
video adds a small node record of its ordered leaves — a node hit adopts
the whole subtree without rendering or detecting anything, while a miss
falls back to the per-page or per-frame leaves, so only what changed is
read again.

This module is the facade the pipeline holds: a thin dispatch over the
three materializers, which own their own caching conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from ragtools.video import SlideDetector

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.materialize.image import ImageMaterializer
from rag_etl.transformers.materialize.markdown import MarkdownMaterializer
from rag_etl.transformers.materialize.pdf import PdfMaterializer
from rag_etl.transformers.materialize.video import VideoMaterializer
from ragtools.kaltura import KalturaVideoResolver


@dataclass(frozen=True)
class MaterializeTransformer(Transformer):
    """Turns files that hold text in pixels into text files:

    - every PDF page is rendered and read, the pdf gaining one child per page,
    - images are read the same way, in place,
    - the local images a markdown file shows are replaced by their transcripts,
    - videos gain one slide child per slide segment, captions appended.
    """

    name = "materialize"

    cache: Cache
    # The scratch work area for rendered pages, extracted frames and read
    # transcripts: outside every source tree (generated files must never
    # sit next to the sources they came from), wiped by the pipeline at
    # every run start — everything in it is re-derivable
    work_root: Path
    detector: SlideDetector
    mediaspace_url: str | None = None
    min_slide_seconds: int = 40
    language: str | None = None

    def transform(self, roots: list[Resource]) -> list[Resource]:
        pdf = PdfMaterializer(cache=self.cache, work_root=self.work_root)
        image = ImageMaterializer(cache=self.cache, work_root=self.work_root)
        markdown = MarkdownMaterializer(cache=self.cache, work_root=self.work_root)
        video = VideoMaterializer(
            cache=self.cache,
            work_root=self.work_root,
            detector=self.detector,
            mediaspace_url=self.mediaspace_url,
            min_slide_seconds=self.min_slide_seconds,
            language=self.language,
        )
        resolver = KalturaVideoResolver()

        for root in roots:
            for node in list(root.walk()):
                # A pdf is materialized once, when it has no pages yet; a
                # video once, when it has no slides yet; an image by mime,
                # since it is read in place. The walk is listed before anything
                # is materialized, so only markdown the sources gave is read for
                # images, never the transcripts made here
                if node.mime_type == mt.PDF and node.path is not None and node.is_leaf:
                    pdf.materialize(node)
                elif node.mime_type in (mt.JPEG, mt.PNG) and node.path is not None:
                    image.materialize(node)
                elif node.mime_type == mt.MARKDOWN and node.path is not None and node.is_leaf:
                    markdown.materialize(node)
                elif node.mime_type == mt.MP4 and not any(child.mime_type == mt.MARKDOWN for child in node.children):
                    video.materialize(node, resolver)

        return roots
