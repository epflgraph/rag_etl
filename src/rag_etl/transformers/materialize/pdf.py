"""PDF materialization: one child per page, cached as a Merkle tree.

The leaves are the per-page transcripts, each stored exactly once under
its own rendered-page hash; the pdf's node (`children.json`, keyed by the
pdf's hash) is a small record of its ordered page leaves. A node hit
adopts the whole subtree — nothing is rendered, no page is read again; a
node miss renders and falls back to the leaves, so one edited page re-OCRs
one page. Either way the node is written back, so the run after a miss
hits at the top again.
"""

from __future__ import annotations

import json
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

NODE_NAME = "children.json"

# Bumped whenever rendering or reading changes what the same page pixels
# mean (dpi, prompts, model): nodes written under an older recipe are
# stale and never adopted, and their pages are read again wholesale
FINGERPRINT = "pdf-ocr-1"


@dataclass(frozen=True)
class PdfMaterializer:
    """Turns one PDF into one child per page, under a Merkle node."""

    cache: Cache
    work_root: Path

    def materialize(self, node: Resource) -> None:
        pdf_path = node.path
        if pdf_path is None:
            return

        pdf_hash = node.content_hash
        bench = self.work_root / pdf_hash

        record = self._node(pdf_hash)
        pages = self._adopt(record, bench)
        if pages is None:
            # A stale recipe means the leaves are not trusted either: they
            # are keyed by pixels alone, so every page is read again
            stale = record is not None and record.get("fingerprint") != FINGERPRINT
            pages = self._render(pdf_path, pdf_hash, bench, stale)

        for page_number, md_path in pages:
            if not md_path.read_text(encoding="utf-8").strip():
                logger.debug(f"Dropping page {page_number} of {node.title}: no readable content")
                continue

            node.add_child(title=f"page {page_number}", path=md_path, mime_type=mt.MARKDOWN)

        logger.info(f"{node.title}: {len(node.children)} pages")

    def _node(self, pdf_hash: str) -> dict | None:
        """The pdf's node record, or None when there is none or it is malformed."""
        text = self.cache.read(pdf_hash, NODE_NAME)
        if text is None:
            return None

        try:
            record = json.loads(text)
            if not isinstance(record, dict) or not isinstance(record.get("children"), list):
                return None
        except json.JSONDecodeError:
            return None

        return record

    def _adopt(self, record: dict | None, bench: Path) -> list[tuple[int, Path]] | None:
        """The bench pages adopted from the node, or None to render instead.

        Adoption needs the current fingerprint and every leaf it names
        still in the cache — anything less falls back to rendering, which
        re-fills what is missing. Copies already made before the gap are
        harmless: the render path re-reads them from the same leaves.
        """
        if record is None or record.get("fingerprint") != FINGERPRINT:
            return None

        bench.mkdir(parents=True, exist_ok=True)
        pages: list[tuple[int, Path]] = []
        try:
            for name, leaf_hash in record["children"]:
                md_path = bench / name
                if not self.cache.get(leaf_hash, md_path):
                    return None
                pages.append((int(Path(name).stem.removeprefix("page_")), md_path))
        except (TypeError, ValueError):
            return None

        return pages

    def _render(self, pdf_path: Path, pdf_hash: str, bench: Path, stale: bool) -> list[tuple[int, Path]]:
        """Render every page and read it, refilling the leaves and the node."""
        page_paths = save_pdf_pages(pdf_path, bench)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
            md_paths = list(
                executor.map(
                    lambda page: ocr_cached(
                        self.cache, page, PDF_SYSTEM_PROMPT, PDF_USER_PROMPT, page.with_suffix(".md"), bypass=stale
                    ),
                    page_paths,
                )
            )

        children = [
            [md_path.name, hash_bytes(page_path.read_bytes())] for page_path, md_path in zip(page_paths, md_paths)
        ]
        self.cache.write(pdf_hash, NODE_NAME, json.dumps({"fingerprint": FINGERPRINT, "children": children}))

        return [(int(page_path.stem.removeprefix("page_")), md_path) for page_path, md_path in zip(page_paths, md_paths)]
