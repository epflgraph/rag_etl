"""The loader contract: take the transformed trees, produce nothing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_etl.core import Resource


@dataclass(frozen=True)
class Loader(ABC):
    """Publishes or records the trees the last transformer handed over.

    A loader only reads the trees; what it produces lives outside the
    pipeline (an index, a file).
    """

    @abstractmethod
    def load(self, roots: list[Resource]) -> None:
        """Load every tree."""
        raise NotImplementedError
