"""The extractor contract: fetch one source, return one tree of resources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rag_etl.core import Resource


class SourceUnavailable(RuntimeError):
    """A source could not be reached. Raised, never swallowed, so a run
    either indexes everything or stops loud."""


@dataclass(frozen=True)
class Extractor(ABC):
    """Fetches the raw material of one source and returns it as a resource tree.

    Each extractor returns a single root container holding everything the
    source yielded, with the source's public url on it. Nodes carry facts
    (title, path, mime type, dates) and whatever labels the source itself
    already infers; the judge fills in the rest. Any failure to reach the
    source raises SourceUnavailable.
    """

    dir: Path  # where the extractor hands files over: downloads, copies, clones

    @abstractmethod
    def extract(self) -> Resource:
        """Perform the extraction and return the root container of the tree."""
        raise NotImplementedError
