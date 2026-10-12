"""Normalize: archives into files, notebooks into Markdown, oversized files dropped."""

from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass, field

import ragtools.mime_types as mt
from ragtools.archives import iter_files, untar_file, unzip_file
from ragtools.encoding import ensure_utf8
from ragtools.llm import RCPClient
from ragtools.notebooks import convert_ipynb_to_md

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.base import Transformer

logger = logging.getLogger(__name__)

# A file bigger than its family's limit is dropped: unreadable at that size
# by whoever consumes it next, and it would only inflate the index
PDF_SIZE_LIMIT = 50 * 1024**2
IMAGE_SIZE_LIMIT = 10 * 1024**2
TEXT_SIZE_LIMIT = 5 * 1024**2


def size_limit(mime_type: str | None) -> int | None:
    """Return the size limit of a mime type, or None when it has none."""
    if mime_type == mt.PDF:
        return PDF_SIZE_LIMIT
    if mime_type in (mt.JPEG, mt.PNG):
        return IMAGE_SIZE_LIMIT
    if mime_type is not None and (mime_type.startswith("text/") or mime_type in (mt.IPYNB, mt.SRT)):
        return TEXT_SIZE_LIMIT
    return None


def is_archive(node: Resource) -> bool:
    """True when a node stands for a zip or tar archive not expanded yet."""

    if not node.is_leaf or node.path is None:
        return False

    if node.mime_type in (mt.ZIP, mt.TAR, mt.GZIP):
        return True

    return node.path.name.endswith((".tar.gz", ".tgz"))


@dataclass(frozen=True)
class NormalizeTransformer(Transformer):
    """
    Normalizes the material of every tree:

    - zip and tar.gz archives are extracted, replaced by a container holding
      one child per extracted file,
    - Jupyter notebooks are converted to Markdown,
    - files bigger than their mime family's size limit are dropped.
    """

    cache: Cache
    llm: RCPClient
    vision_model: str
    mime_types: Sequence[str] = field(default_factory=lambda: list(mt.DEFAULT_MIME_TYPES))

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            self._expand_archives(root)
            self._convert_notebooks(root)
            self._drop_oversized(root)
        return roots

    def _expand_archives(self, root: Resource) -> None:
        for node in list(root.walk()):
            if not is_archive(node) or node.path is None:
                continue

            logging.info(f"Extracting {node.title}")

            if node.mime_type == mt.ZIP:
                extract_dir = unzip_file(node.path)
            else:
                extract_dir = untar_file(node.path)

            # The archive node keeps its identity and gains one child per
            # extracted file, becoming the container of its contents
            for extracted_file in iter_files(extract_dir):
                mime_type = mt.guess_mime_type(str(extracted_file))

                if mime_type not in self.mime_types:
                    continue

                # An archive can hold text saved before UTF-8 was the norm, and
                # a reader that assumes UTF-8 gets nothing at all out of it
                if mime_type.startswith("text/"):
                    ensure_utf8(extracted_file)

                node.add_child(
                    Resource(
                        title=f"{node.title} > {extracted_file.name}",
                        path=extracted_file,
                        mime_type=mime_type,
                    )
                )

            if not node.children:
                logger.warning(f"Dropping {node.title}: archive holds no supported file")
                if node.parent is not None:
                    node.parent.children.remove(node)

    def _convert_notebooks(self, root: Resource) -> None:
        for node in list(root.walk()):
            if node.mime_type != mt.IPYNB or node.path is None:
                continue

            md_path = node.path.with_suffix(".md")
            content_hash = node.content_hash

            if not self.cache.get(content_hash, md_path):
                logger.info(f"Converting notebook {node.title}")
                convert_ipynb_to_md(node.path, md_path, self.llm, self.vision_model)
                self.cache.set(content_hash, md_path)

            node.path = md_path
            node.mime_type = mt.MARKDOWN

    def _drop_oversized(self, root: Resource) -> None:
        for node in list(root.walk()):
            if not node.is_leaf or node.path is None:
                continue

            limit = size_limit(node.mime_type)
            if limit is not None and node.path.stat().st_size > limit:
                logger.warning(f"Dropping {node.title}: over the {node.mime_type} size limit")
                if node.parent is not None:
                    node.parent.children.remove(node)
