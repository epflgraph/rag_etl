"""PDF materialization: one child per page, cached as a Merkle tree.

The leaves are the per-page transcripts, each stored exactly once under
its own rendered-page hash; the pdf's node record, keyed by the pdf's
hash, lists its ordered page leaves. A node hit adopts the whole subtree —
nothing is rendered, no page is read again; a node miss renders and falls
back to the leaves, so one edited page re-OCRs one page. Either way the
node is written back, so the run after a miss hits at the top again.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ragtools.mime_types as mt
from ragtools.pdf import PDF_SYSTEM_PROMPT, PDF_USER_PROMPT, save_pdf_pages

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_bytes
from rag_etl.transformers.materialize.ocr import MAX_CONCURRENT_OCR, ocr_cached

logger = logging.getLogger(__name__)

# The cache scope the transcripts live in. Bumped whenever rendering or
# reading changes what the same page pixels mean (dpi, prompts, model): the
# new scope starts empty, so every page is read again
CACHE_VERSION = "pdf-ocr-1"


def _page_number(md_path: Path) -> int:
    """The page a transcript belongs to, read back from its name (page_3.md → 3)."""
    return int(md_path.stem.removeprefix("page_"))


@dataclass(frozen=True)
class PdfMaterializer:
    """Turns one PDF into one child per page, under a Merkle node."""

    cache: Cache
    work_root: Path

    def materialize(self, node: Resource) -> None:
        pdf_path = node.path
        if pdf_path is None:
            return

        cache = self.cache.scope(CACHE_VERSION)
        pdf_hash = node.content_hash
        work_dir = self.work_root / pdf_hash

        md_paths = cache.adopt(pdf_hash, work_dir)
        if md_paths is None:
            md_paths = self._render(cache, pdf_path, pdf_hash, work_dir)

        for md_path in md_paths:
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping page {_page_number(md_path)} of {node.title}: no readable content")
                continue

            node.add_child(title=f"page {_page_number(md_path)}", path=md_path, mime_type=mt.MARKDOWN)

        logger.info(f"{node.title}: {len(node.children)} pages")

    def _render(self, cache: Cache, pdf_path: Path, pdf_hash: str, work_dir: Path) -> list[Path]:
        """Render every page and read it, refilling the leaves and the node."""
        page_paths = save_pdf_pages(pdf_path, work_dir)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(
                executor.map(
                    lambda page: ocr_cached(cache, page, PDF_SYSTEM_PROMPT, PDF_USER_PROMPT, page.with_suffix(".md")),
                    page_paths,
                )
            )

        cache.write_node(
            pdf_hash,
            [(md_path.name, hash_bytes(page_path.read_bytes())) for page_path, md_path in zip(page_paths, md_paths)],
        )

        return md_paths
