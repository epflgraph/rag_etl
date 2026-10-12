"""The transformer contract: reshape the trees handed over by the previous step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_etl.core import Resource


@dataclass(frozen=True)
class Transformer(ABC):
    """Reshapes resource trees in place as one step of the fixed sequence.

    A transformer adds, updates and removes nodes (recursively) and hands
    the same trees back; the pipeline runs these in order, so the output
    of one is the input of the next.
    """

    @abstractmethod
    def transform(self, roots: list[Resource]) -> list[Resource]:
        """Perform the transformation over every tree."""
        raise NotImplementedError
