"""Cut: the trees the judge labelled become chunk leaves.

A cut consumes the node it applies to: the node is replaced by chunk
leaves (a chunk is a leaf with text), which the embed step then turns
into (chunk, embedding) pairs for the loader. The cuts live in one
registry; new kinds of cuts join via PR, never course-local code.
"""

from __future__ import annotations

import json
import logging
import re

from collections.abc import Callable
from dataclasses import asdict, dataclass

from rag_etl.cache import Cache
from rag_etl.core import Resource, child_inheriting, is_text_node
from rag_etl.core.resource import hash_text
from rag_etl.transformers.base import Transformer
from ragtools.documents import (
    MAX_SECTION_LINES,
    PAGE_MARKER,
    ExerciseSnippet,
    llm_stitch,
    merge_up_to_max_lines,
    split_by_most_common_heading,
    split_exercises,
    stitch,
)

logger = logging.getLogger(__name__)

# The exercises of one stitched document, cached by the document's content hash
EXERCISES_FILE = "exercises.json"

# Deep links into a pdf page, read back from the markers the stitch left
PAGE_MARKER_RE = re.compile(PAGE_MARKER.format(position=r"(\d+)"))

CutFunction = Callable[["CutTransformer", Resource], None]


def document_texts(node: Resource) -> list[str]:
    """The texts to merge: one per readable page child, or the node's own file."""
    if node.children:
        return [
            child.path.read_text(encoding="utf-8")
            for child in node.children
            if is_text_node(child) and child.path is not None
        ]
    if node.path is not None:
        return [node.path.read_text(encoding="utf-8")]
    return []


def _drop(node: Resource) -> None:
    """Remove a node from its parent, its content gone with it; a root is left in place, inert."""
    node.children = []
    if node.parent is not None:
        node.parent.children.remove(node)


def exercises(cache: Cache, text: str) -> list[ExerciseSnippet]:
    """A document split into exercises, cached by the document's content hash."""
    content_hash = hash_text(text)

    cached = cache.read(content_hash, EXERCISES_FILE)
    if cached is not None:
        return [ExerciseSnippet(**exercise) for exercise in json.loads(cached)]

    found = split_exercises(text)
    cache.write(content_hash, EXERCISES_FILE, json.dumps([asdict(ex) for ex in found]))
    return found


def _stitch_pages(transformer: "CutTransformer", pages: list[str]) -> str:
    """The pages stitched, the LLM fallback enabled only when asked for."""
    text = stitch(pages)
    if transformer.stitch_fallback:
        text = llm_stitch(text) or text
    return text


def whole_document(transformer: "CutTransformer", node: Resource) -> None:
    """The node becomes one chunk: its pages stitched, or its own file read."""
    pages = document_texts(node)

    if not pages:
        logger.warning(f"Dropping {node.title}: nothing readable to cut whole")
        _drop(node)
        return

    if len(pages) == 1:
        text = pages[0]
    else:
        text = _stitch_pages(transformer, pages)

    if not text.strip():
        logger.warning(f"Dropping {node.title}: no readable content")
        _drop(node)
        return

    if len(text.splitlines()) > MAX_SECTION_LINES:
        logger.warning(f"Dropping {node.title}: {len(text.splitlines())} lines over the {MAX_SECTION_LINES}-line size guard")
        _drop(node)
        return

    node.text = text
    node.children = []


def per_child(transformer: "CutTransformer", node: Resource) -> None:
    """Each readable child becomes a chunk in place, text read from its own file.

    Children carrying their own different cut keep it — an exception inside
    a folder cut per child still applies.
    """
    for child in list(node.children):
        if child.cut not in (None, "per_child"):
            continue
        if not is_text_node(child) or child.path is None:
            continue

        text = child.path.read_text(encoding="utf-8")
        if not text.strip():
            logger.debug(f"Dropping {child.title}: no readable content")
            node.children.remove(child)
            continue

        child.text = text
        child.cut = None


def per_exercise(transformer: "CutTransformer", node: Resource) -> None:
    """The node's pages are stitched, annotated and cut into exercise chunks.

    One chunk per exercise, carrying its number and whether it holds a
    solution; a chunk whose text starts on a pdf page deep-links into it.
    """
    pages = document_texts(node)
    if not pages:
        logger.warning(f"Dropping {node.title}: nothing to annotate")
        _drop(node)
        return

    stitched = _stitch_pages(transformer, pages)

    found = exercises(transformer.cache, stitched)
    if not found:
        logger.warning(f"Dropping {node.title}: no exercises annotated in it")
        _drop(node)
        return

    node.children = []
    for exercise in found:
        chunk = child_inheriting(
            node,
            title=f"{node.title} > Exercise {exercise.number}{' (solution)' if exercise.is_solution else ''}",
            text=exercise.text,
            number=exercise.number,
            is_solution=exercise.is_solution,
            cut=None,
        )

        page = PAGE_MARKER_RE.search(exercise.text)
        if page and node.url:
            chunk.url = f"{node.url}#page={page.group(1)}"

        node.add_child(chunk)


def text(transformer: "CutTransformer", node: Resource) -> None:
    """The node's text is cut at its most common heading level, one chunk per part."""
    full = "\n\n".join(document_texts(node))

    if not full.strip():
        logger.warning(f"Dropping {node.title}: no readable content to cut")
        _drop(node)
        return

    sections = merge_up_to_max_lines(split_by_most_common_heading(full), MAX_SECTION_LINES)

    node.children = []
    for position, section in enumerate(sections, start=1):
        if len(section.splitlines()) > MAX_SECTION_LINES:
            logger.warning(f"Dropping part {position} of {node.title}: over the {MAX_SECTION_LINES}-line size guard")
            continue

        node.add_child(
            child_inheriting(
                node,
                title=f"{node.title} > part {position}",
                text=section,
                cut=None,
            )
        )


# The one cut catalogue; new cuts join here via PR
CUTS: dict[str, CutFunction] = {
    "whole_document": whole_document,
    "per_child": per_child,
    "per_exercise": per_exercise,
    "text": text,
}


@dataclass(frozen=True)
class CutTransformer(Transformer):
    """Cuts the labelled trees into chunk leaves, one registry entry per kind of cut."""

    cache: Cache
    stitch_fallback: bool = False  # the LLM stitch fallback, off by default

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            self._cut(root)
        return roots

    def _cut(self, node: Resource) -> None:
        """Cut one node top-down: a cut consumes the node it applies to."""
        if node.cut is not None:
            try:
                cut = CUTS[node.cut]
            except KeyError:
                raise ValueError(f"Unknown cut '{node.cut}' for {node.title}") from None
            cut(self, node)

        # A consumed node has no children left, or chunk children that are
        # done; its subtree never needs a second cut
        for child in list(node.children):
            self._cut(child)
