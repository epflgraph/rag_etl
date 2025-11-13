from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from rag_etl.resources import BaseResource

from rag_etl.utils.cache import get_from_cache, set_to_cache


class BaseTransformer(ABC):
    """
    Base class for all transformers.

    Transformers take a sequence of `Resource` objects,
    apply some modification, enrichment, or normalization,
    and return a new list of transformed `Resource` objects.
    """

    def get_from_cache(self, resource_path, destination_path):
        scope = self.__class__.__name__
        return get_from_cache(scope, resource_path, destination_path)

    def set_to_cache(self, resource_path, source_path):
        scope = self.__class__.__name__
        set_to_cache(scope, resource_path, source_path)

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
