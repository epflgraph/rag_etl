"""The one OCR step the images, pdf and video transformers share, cached by content."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_etl.cache import Cache
from rag_etl.core import Resource, hash_file

# Page and frame reads run bounded: twenty concurrent calls is what kept
# the old pipeline well behaved against RCP
MAX_CONCURRENT_OCR = 20


def file_to_read(node: Resource) -> Path:
    """The node's file; a file-backed node without one is an extractor's bug, raised."""
    if node.path is None:
        raise ValueError(f"{node.title!r} ({node.mime_type}) has no file to read")
    return node.path


def cached_transcript(cache: Cache, image_path: Path, transcribe: Callable[[Path], str], md_path: Path) -> Path:
    """Write one image's transcript to md_path, transcribing it unless cached under the image's hash.

    `transcribe` is the ragtools transcription for the kind of image it is:
    `transcribe_page` for a rendered PDF page, `transcribe_slide` for a
    slide or figure.
    """
    key = hash_file(image_path)

    text = cache.read(key, md_path.name)
    if text is None:
        text = transcribe(image_path)
        cache.write(key, md_path.name, text)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    return md_path


def write_transcripts(parts: list[tuple[str, str]], work_dir: Path) -> list[Path]:
    """Write a document's (name, text) transcripts, as its cached index gave them, into its work folder."""
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in parts:
        path = work_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
