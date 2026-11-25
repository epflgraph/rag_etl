"""Harmonize: a PDF's pages, read one by one, made consistent as one document.

Each page was transcribed knowing nothing of the others, so heading
levels drift and tables, lists and equations are cut at page boundaries.
A model harmonizes the pages of every PDF (`ragtools.documents.harmonize_pages`),
and the result is split back at the page markers: the pages stay child
resources, their text now consistent with their neighbours'. Nothing is
joined here: whatever the cut that follows (per page, whole, per
exercise), it works on these pages.

Cached by the hash of the pages as read: an unchanged PDF is harmonized once.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ragtools.mime_types as mt
from ragtools.documents import harmonize_pages, split_pages

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_text
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.pdf import page_number

logger = logging.getLogger(__name__)

# The harmonized document of one PDF, cached by the hash of its pages as read
HARMONIZED_FILE = "harmonized.md"

# Documents harmonized at once; each is several calls of many pages
MAX_CONCURRENT_DOCUMENTS = 10


def _pages(node: Resource) -> list[Resource]:
    """The page children the pdf step gave a PDF."""
    return [child for child in node.children if child.mime_type == mt.MARKDOWN and child.path is not None]


@dataclass(frozen=True)
class HarmonizeTransformer(Transformer):
    """Harmonizes the pages of every PDF with a model, the pages staying child resources."""

    name = "harmonize"
    cache_version = "harmonize-1"  # the harmonized documents (prompts, model, batch size)

    def transform(self, roots: list[Resource]) -> list[Resource]:
        cache = self.cache

        # A single page has no neighbour to be consistent with
        nodes = [node for root in roots for node in root.walk() if node.mime_type == mt.PDF and len(_pages(node)) > 1]

        # A course hands the same PDF over more than once (a file and the
        # archive it is also in): each document is harmonized once, for all
        # its copies, rather than once per copy that misses the cache together
        documents: dict[str, list[tuple[int, str]]] = {}
        copies: dict[str, list[Resource]] = {}
        for node in nodes:
            pages = [(page_number(page.path), page.path.read_text(encoding="utf-8")) for page in _pages(node) if page.path]
            key = hash_text("\n".join(f"{number}\n{text}" for number, text in pages))
            documents[key] = pages
            copies.setdefault(key, []).append(node)
        logger.info(f"{len(documents)} documents to harmonize, in {len(nodes)} PDFs")

        # The model calls run side by side; the trees are only touched after
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOCUMENTS) as executor:
            texts = list(
                executor.map(lambda key: self._harmonized(cache, key, documents[key], copies[key][0].title), documents)
            )

        for key, text in zip(documents, texts):
            for node in copies[key]:
                self._replace_pages(node, key, text)

        return roots

    def _harmonized(self, cache: Cache, key: str, pages: list[tuple[int, str]], title: str) -> str:
        """The document's harmonized text, from the cache or the model."""
        text = cache.read(key, HARMONIZED_FILE)
        if text is None:
            logger.info(f"Harmonizing {title} ({len(pages)} pages)")
            text = harmonize_pages(pages)
            cache.write(key, HARMONIZED_FILE, text)

        return text

    def _replace_pages(self, node: Resource, key: str, text: str) -> None:
        """Split the harmonized document back into the page children, or keep them as read."""
        pages = _pages(node)
        harmonized = split_pages(text)

        # The pages given must be the pages got: a marker the model lost,
        # moved or repeated would file text under the wrong page
        if [number for number, _ in harmonized] != [page_number(page.path) for page in pages if page.path]:
            logger.warning(f"Keeping the pages of {node.title} as read: the model did not keep its page markers")
            return

        for page, (number, page_text) in zip(pages, harmonized):
            # A page whose content was merged into its neighbour (the end of a table) holds nothing anymore
            if not page_text:
                page.detach()
                continue

            path = self.work_root / key / f"page_{number}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page_text, encoding="utf-8")
            page.path = path
