"""The one OCR step every materializer shares, cached by content."""

from __future__ import annotations

from pathlib import Path

from ragtools.images import ocr_image

from rag_etl.cache import Cache
from rag_etl.core.resource import hash_bytes

# Page and frame reads run bounded: twenty concurrent calls is what kept
# the old pipeline well behaved against RCP
MAX_CONCURRENT_OCR = 20


def ocr_cached(cache: Cache, image_path: Path, system: str, user: str, md_path: Path) -> Path:
    """Read one image into md_path, cached by the image's content hash."""
    content_hash = hash_bytes(image_path.read_bytes())

    if not cache.get(content_hash, md_path):
        text = ocr_image(image_path, system=system, user=user)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        cache.set(content_hash, md_path)

    return md_path
