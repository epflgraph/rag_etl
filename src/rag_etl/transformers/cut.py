"""Cut: the trees the judge labelled become chunk leaves.

A cut consumes the node it applies to: the node is replaced by chunk
leaves (a chunk is a leaf with text), which the embed step then
annotates with their vectors. The cuts live in one
registry; new kinds of cuts join via PR, never course-local code.
"""

from __future__ import annotations

import json
import logging
import re

from collections.abc import Callable
from dataclasses import asdict, dataclass

import ragtools.mime_types as mt

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.core.resource import hash_text
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.embed import MAX_CHUNK_CHARS
from ragtools.documents import (
    PAGE_MARKER,
    ExerciseSnippet,
    llm_stitch,
    merge_sections,
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
            if mt.is_text(child.mime_type) and child.path is not None
        ]
    if node.path is not None:
        return [node.path.read_text(encoding="utf-8")]
    return []


def _drop(node: Resource) -> None:
    """Remove a node from its tree, its content gone with it, so nothing below it is cut."""
    node.children = []
    node.detach()


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

    if len(text) > MAX_CHUNK_CHARS:
        logger.warning(f"Dropping {node.title}: {len(text):,} chars over the {MAX_CHUNK_CHARS:,}-char chunk limit")
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
        if not mt.is_text(child.mime_type) or child.path is None:
            continue

        text = child.path.read_text(encoding="utf-8")
        if not text.strip():
            logger.debug(f"Dropping {child.title}: no readable content")
            child.detach()
            continue

        child.text = text
        child.cut = None
        # A chunk holds what its file held; a chunk of a corrigé is a solution
        if child.exercise_parts is None:
            child.exercise_parts = node.exercise_parts


def per_exercise(transformer: "CutTransformer", node: Resource) -> None:
    """The node's pages are stitched, annotated and cut into exercise chunks.

    One chunk per exercise, carrying its number and the parts it holds; a
    chunk whose text starts on a pdf page deep-links into it.
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
        # A chunk is text alone: no file behind it, nothing left to cut
        chunk = node.add_child(
            title=f"Exercise {exercise.number} ({exercise.parts})",
            text=exercise.text,
            path=None,
            mime_type=mt.MARKDOWN,
            number=exercise.number,
            exercise_parts=exercise.parts,
            cut=None,
        )

        page = PAGE_MARKER_RE.search(exercise.text)
        if page and chunk.url:
            chunk.url = f"{chunk.url}#page={page.group(1)}"

    # has_solution is derived, not inherited: an exercise with any solution
    # chunk in this sheet — its own or a sibling's — has one
    solved = {chunk.number for chunk in node.children if "solution" in (chunk.exercise_parts or "")}
    for chunk in node.children:
        chunk.has_solution = chunk.number in solved


def text(transformer: "CutTransformer", node: Resource) -> None:
    """The node's text is cut at its most common heading level, one chunk per part."""
    full = "\n\n".join(document_texts(node))

    if not full.strip():
        logger.warning(f"Dropping {node.title}: no readable content to cut")
        _drop(node)
        return

    sections = merge_sections(split_by_most_common_heading(full), MAX_CHUNK_CHARS, size=len)

    node.children = []
    for position, section in enumerate(sections, start=1):
        if len(section) > MAX_CHUNK_CHARS:
            logger.warning(f"Dropping part {position} of {node.title}: over the {MAX_CHUNK_CHARS:,}-char chunk limit")
            continue

        # A chunk is text alone: no file behind it, nothing left to cut
        node.add_child(title=f"part {position}", text=section, path=None, mime_type=mt.MARKDOWN, cut=None)


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

    name = "cut"

    cache: Cache
    stitch_fallback: bool = False  # the LLM stitch fallback, off by default

    def transform(self, roots: list[Resource]) -> list[Resource]:
        for root in roots:
            self._cut(root)
            # Chunks that hold a solution but were not cut by per_exercise
            # (per_child children of a solution file inherit its parts) catch up
            for node in root.walk():
                if "solution" in (node.exercise_parts or ""):
                    node.has_solution = True
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
