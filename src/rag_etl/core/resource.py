"""The resource tree model: one recursive dataclass; policy never sits in the data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


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
    path: Path | None = None       # where the content lives on disk; None for pure containers
    mime_type: str | None = None

    # lineage
    parent: Resource | None = None
    children: list[Resource] = field(default_factory=list)

    # descriptive metadata (extractors fill some; the judge fills type/subtype/is_solution/cut; exceptions correct)
    title: str | None = None
    type: str | None = None        # theory / practice / ... (inferred by the judge)
    subtype: str | None = None     # lecture_slides / homework / ...
    cut: str | None = None         # whole_document | per_child | per_exercise | text (inferred by the judge)
    number: int | None = None      # exercise number (per_exercise annotator)
    sub_number: int | None = None  # exercise sub-number
    week: int | None = None        # lecture week (slides, recordings)
    is_solution: bool = False
    from_: date | None = None      # release date (Moodle availability)
    until: date | None = None      # close date

    # content (chunks only)
    text: str | None = None

    def __post_init__(self) -> None:
        for child in self.children:
            if child.parent is None:
                child.parent = self

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children yet; flips as the tree grows."""
        return not self.children

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

    def add_child(self, child: Resource) -> Resource:
        """Attach a child under this node and return it."""
        child.parent = self
        self.children.append(child)
        return child

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
