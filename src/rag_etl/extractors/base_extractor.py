from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_etl.resources import BaseResource


class BaseExtractor(ABC):
    """
    Base class for all extractors.

    Extractors are responsible for fetching raw resources from
    external sources such as APIs, file systems, or LMS platforms.

    Each subclass must implement `extract()` to return a list
    of `Resource` instances.
    """

    @abstractmethod
    def extract(self) -> List[BaseResource]:
        """
        Perform extraction and return a list of Resource objects.

        Returns:
            list[Resource]: The extracted resources ready for transformation.
        """
        raise NotImplementedError
