from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from rag_etl.resources import BaseResource


class BaseLoader(ABC):
    """
    Base class for all loaders.

    Loaders take a (possibly transformed) sequence of `Resource` objects and
    persist them somewhere (DB, filesystem, vector store, search index, etc.).
    """

    @abstractmethod
    def load(self, resources: Sequence[BaseResource]) -> None:
        """
        Persist the given resources. Implementations should be side-effecting
        and return nothing.

        Args:
            resources: Ordered sequence of resources to persist.
        """
        raise NotImplementedError
