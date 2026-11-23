"""The one OCR step the images, pdf and video transformers share, cached by content."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_bytes


def file_to_read(node: Resource) -> Path:
    """The node's file; a file-backed node without one is an extractor's bug, raised."""
    if node.path is None:
        raise ValueError(f"{node.title!r} ({node.mime_type}) has no file to read")
    return node.path

# Page and frame reads run bounded: twenty concurrent calls is what kept
# the old pipeline well behaved against RCP
MAX_CONCURRENT_OCR = 20


def ocr_cached(cache: Cache, image_path: Path, transcribe: Callable[[Path], str], md_path: Path) -> Path:
    """Transcribe one image into md_path, cached by the image's content hash.

    `transcribe` is the ragtools transcription for the kind of image it is:
    `transcribe_page` for a rendered PDF page, `transcribe_slide` for a
    slide or figure.
    """
    content_hash = hash_bytes(image_path.read_bytes())

    if not cache.get(content_hash, md_path):
        text = transcribe(image_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        cache.set(content_hash, md_path)

    return md_path
