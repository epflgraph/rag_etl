"""PDF: one child per page, every page rendered and read, cached as a Merkle tree.

The parts are the per-page transcripts, each stored exactly once under
its own rendered-page hash; the pdf's index, keyed by the pdf's hash,
lists its ordered pages. A pdf found by its index costs nothing: nothing
is rendered, no page is read again. One whose index misses is rendered and
falls back to its pages, so one edited page re-OCRs one page. Either way
the index is written back, so the run after a miss finds it again.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from ragtools.images import transcribe_page
from ragtools.pdf import save_pdf_pages

from rag_etl.cache import Cache
from rag_etl.core import Resource, hash_file
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.ocr import MAX_CONCURRENT_OCR, cached_transcript, file_to_read, write_transcripts

logger = logging.getLogger(__name__)


def page_number(md_path: Path) -> int:
    """The page a transcript belongs to, read back from its name (page_3.md → 3)."""
    return int(md_path.stem.removeprefix("page_"))


@dataclass(frozen=True)
class PdfTransformer(Transformer):
    """Turns every PDF into one child per page."""

    name = "pdf"
    cache_version = "pdf-ocr-1"  # the page reads (rendering settings, prompts, model)

    def transform(self, roots: list[Resource]) -> list[Resource]:
        cache = self.cache
        for root in roots:
            for node in list(root.walk()):
                if node.mime_type == mt.PDF:
                    self._read(cache, node)
        return roots

    def _read(self, cache: Cache, node: Resource) -> None:
        pdf_path = file_to_read(node)
        pdf_hash = node.content_hash
        work_dir = self.work_root / pdf_hash

        pages = cache.read_index(pdf_hash)
        if pages is None:
            md_paths = self._render(cache, pdf_path, pdf_hash, work_dir)
        else:
            md_paths = write_transcripts(pages, work_dir)

        for md_path in md_paths:
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping page {page_number(md_path)} of {node.title}: no readable content")
                continue

            node.add_child(title=f"page {page_number(md_path)}", path=md_path, mime_type=mt.MARKDOWN)

        logger.info(f"{node.title}: {len(node.children)} pages")

    def _render(self, cache: Cache, pdf_path: Path, pdf_hash: str, work_dir: Path) -> list[Path]:
        """Render every page and read it, refilling the pages and the index."""
        page_paths = save_pdf_pages(pdf_path, work_dir)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(
                executor.map(
                    lambda page: cached_transcript(cache, page, transcribe_page, page.with_suffix(".md")),
                    page_paths,
                )
            )

        cache.write_index(
            pdf_hash, [(md_path.name, hash_file(page_path)) for page_path, md_path in zip(page_paths, md_paths)]
        )

        return md_paths
