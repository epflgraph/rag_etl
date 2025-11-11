from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from rag_etl.resources import BaseResource


class BaseTransformer(ABC):
    """
    Base class for all transformers.

    Transformers take a sequence of `Resource` objects,
    apply some modification, enrichment, or normalization,
    and return a new list of transformed `Resource` objects.
    """

    @abstractmethod
    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """
        Apply the transformation to a list of resources.

        Args:
            resources (Sequence[Resource]): Input resources from the previous step.

        Returns:
            List[Resource]: Transformed resources ready for loading.
        """
        raise NotImplementedError
