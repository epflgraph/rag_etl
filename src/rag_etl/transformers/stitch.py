"""Stitch: a PDF's pages, read one by one, harmonised as one document.

Each page was transcribed knowing nothing of the others, so heading
levels drift and tables, lists and equations are cut at page boundaries.
A model stitches the pages of every PDF (`ragtools.documents.llm_stitch`),
and the result is split back at the page markers: the pages stay child
resources, their text now consistent with their neighbours'. Whatever the
cut that follows (per page, whole, per exercise), it works on these pages.

Cached by the hash of the pages as read: an unchanged PDF is stitched once.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ragtools.mime_types as mt
from ragtools.documents import llm_stitch, split_pages

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_text
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.pdf import page_number

logger = logging.getLogger(__name__)

# The stitched document of one PDF, cached by the hash of its pages as read
STITCHED_FILE = "stitched.md"

# Documents stitched at once; each is several calls of many pages
MAX_CONCURRENT_STITCHES = 10


def _pages(node: Resource) -> list[Resource]:
    """The page children the pdf step gave a PDF."""
    return [child for child in node.children if child.mime_type == mt.MARKDOWN and child.path is not None]


@dataclass(frozen=True)
class StitchTransformer(Transformer):
    """Harmonises the pages of every PDF with a model, the pages staying child resources."""

    name = "stitch"
    cache_version = "stitch-1"  # the stitched documents (prompts, model, batch size)

    def transform(self, roots: list[Resource]) -> list[Resource]:
        cache = self.cache

        # A single page has no neighbour to be consistent with
        documents = [
            node for root in roots for node in root.walk() if node.mime_type == mt.PDF and len(_pages(node)) > 1
        ]
        logger.info(f"{len(documents)} documents to stitch")

        # The model calls run side by side; the trees are only touched after
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STITCHES) as executor:
            stitched = list(executor.map(lambda node: self._stitched(cache, node), documents))

        for node, (key, text) in zip(documents, stitched):
            self._replace_pages(node, key, text)

        return roots

    def _stitched(self, cache: Cache, node: Resource) -> tuple[str, str]:
        """The document's key and its stitched text, from the cache or the model."""
        pages = [(page_number(page.path), page.path.read_text(encoding="utf-8")) for page in _pages(node) if page.path]
        key = hash_text("\n".join(f"{number}\n{text}" for number, text in pages))

        text = cache.read(key, STITCHED_FILE)
        if text is None:
            logger.info(f"Stitching {node.title} ({len(pages)} pages)")
            text = llm_stitch(pages)
            cache.write(key, STITCHED_FILE, text)

        return key, text

    def _replace_pages(self, node: Resource, key: str, text: str) -> None:
        """Split the stitched document back into the page children, or keep them as read."""
        pages = _pages(node)
        stitched = split_pages(text)

        # The pages given must be the pages got: a marker the model lost,
        # moved or repeated would file text under the wrong page
        if [number for number, _ in stitched] != [page_number(page.path) for page in pages if page.path]:
            logger.warning(f"Keeping the pages of {node.title} as read: the stitch did not keep its page markers")
            return

        for page, (number, page_text) in zip(pages, stitched):
            # A page whose content was merged into its neighbour (the end of a table) holds nothing anymore
            if not page_text:
                page.detach()
                continue

            path = self.work_root / key / f"page_{number}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page_text, encoding="utf-8")
            page.path = path
