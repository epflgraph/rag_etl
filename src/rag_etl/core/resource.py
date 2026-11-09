"""The resource tree model: one recursive dataclass; policy never sits in the data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the sha256 hex digest of utf-8 text."""
    return hash_bytes(text.encode("utf-8"))


@dataclass
class Resource:
    """One node of a resource tree flowing through the ETL pipeline.

    Nodes are created from extracted files and reshaped in place as the
    pipeline runs: operations add, update and remove children (or whole
    nodes), recursively. A chunk is just a leaf with text. Nothing derived
    is stored: `is_leaf` and `content_hash` are computed from the current
    state on every access, so they can never go stale.
    """

    # identity
    title: str                    # the source's name for this resource; "" when it names nothing
    path: Path | None = None      # where the content lives on disk; None for pure containers
    url: str | None = None        # public url the resource is available at
    mime_type: str | None = None

    # lineage
    parent: Resource | None = None
    children: list[Resource] = field(default_factory=list)

    # descriptive metadata (extractors fill some; the judge fills type/subtype/cut; exceptions correct)
    type: str | None = None        # theory / practice / ... (inferred by the judge)
    subtype: str | None = None     # lecture_slides / homework / ...
    cut: str | None = None         # whole_document | per_child | per_exercise | text (inferred by the judge)
    number: str | None = None      # exercise number
    subnumber: str | None = None   # exercise subnumber
    week: int | None = None        # lecture week (slides, recordings)
    exercise_parts: str | None = None  # what the chunk holds: "statement" | "solution" | "statement + solution"
    has_solution: bool = False     # the exercise has a solution chunk (even if this chunk isn't one)
    from_: str | None = None       # release date and time, ISO format
    until: str | None = None       # close date and time

    # content (chunks only)
    text: str | None = None
    embedding: list[float] | None = None  # the chunk's vector, set by embed

    def __post_init__(self) -> None:
        for child in self.children:
            if child.parent is None:
                child.parent = self

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children yet; flips as the tree grows."""
        return not self.children

    @property
    def is_chunk(self) -> bool:
        """Whether this node is a chunk: a node with text."""
        return self.text is not None

    @property
    def content_hash(self) -> str:
        """sha256 of the content this node represents, computed fresh on every access."""
        if self.text is not None:
            return hash_text(self.text)
        if self.children:
            combined = hashlib.sha256()
            for child in self.children:
                combined.update(child.content_hash.encode("utf-8"))
            return combined.hexdigest()
        if self.path is not None:
            return hash_bytes(self.path.read_bytes())
        return hash_text("")

    def add_child(self, child: Resource | None = None, **overrides: Any) -> Resource:
        """Attach a child under this node and return it.

        Without a child, one is built as a copy of this node — every
        field comes down as it is, and the overrides name what differs:
        `pdf.add_child(title="page 1", path=md, mime_type=mt.MARKDOWN)`.
        A child passed in — typically a subtree an extractor built — is
        attached as it is.
        """
        if child is None:
            child = replace(self, **overrides, parent=self, children=[])
        else:
            child.parent = self
        self.children.append(child)
        return child

    def detach(self) -> None:
        """Remove this node, and its subtree with it, from its parent; a root stays as it is."""
        if self.parent is not None:
            self.parent.children.remove(self)

    def walk(self) -> Iterator[Resource]:
        """Yield this node, then its subtree, depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def lineage(self) -> list[Resource]:
        """Return the chain from the root down to this node."""
        chain: list[Resource] = []
        node: Resource | None = self
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain


def iter_chunks(roots: Iterable[Resource]) -> Iterator[Resource]:
    """Yield every chunk in the trees, depth-first."""
    for root in roots:
        yield from (node for node in root.walk() if node.is_chunk)
