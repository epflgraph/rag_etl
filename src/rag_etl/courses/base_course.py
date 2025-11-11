from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import logging

from rag_etl.resources import BaseResource
from rag_etl.extractors import BaseExtractor
from rag_etl.transformers import BaseTransformer
from rag_etl.loaders import BaseLoader


class BaseCourse(ABC):
    """
    Template for a course-specific RAG pipeline.

    Subclasses must define:
      - `extractors`: sequence of BaseExtractor
      - `transformers`: sequence of BaseTransformer
      - `loaders`: sequence of BaseLoader

    Subclasses can override:
      - `extract`: runs extractors sequentially
      - `transform`: runs transformers sequentially
      - `load`: runs loaders sequentially
      - `run`: orchestrates ETL steps
    """

    ################################################################

    @classmethod
    def from_code(cls, code: str) -> BaseCourse:
        """
        Instantiate the course subclass matching the given code.

        Example:
            course = BaseCourse.from_code("COM309")

        Assumes all course classes are already imported in `rag_etl.courses.__init__.py`.
        """
        for subclass in cls.__subclasses__():
            if subclass.__name__ == f"{code}Course":
                return subclass()

        available = ", ".join(sorted(c.__name__ for c in cls.__subclasses__())) or "<none>"
        raise ValueError(f"Unknown course code '{code}'. Available: {available}")

    ################################################################

    @property
    def course_code(self) -> str:
        """Defaults to the class name (e.g., 'MATH101') without 'Course'."""
        return self.__class__.__name__.removesuffix('Course')

    @property
    @abstractmethod
    def extractors(self) -> List[BaseExtractor]:
        """Ordered sequence of extractor instances to run."""
        raise NotImplementedError

    @property
    @abstractmethod
    def transformers(self) -> List[BaseTransformer]:
        """Ordered sequence of transformer instances to run."""
        raise NotImplementedError

    @property
    @abstractmethod
    def loaders(self) -> List[BaseLoader]:
        """Ordered sequence of loader instances to run."""
        raise NotImplementedError

    ################################################################

    def extract(self) -> List[BaseResource]:
        """Run all extractors and collect their results."""

        resources: List[BaseResource] = []
        for extractor in self.extractors:
            logging.debug(f"Running extractor: {extractor.__class__.__name__}")
            extracted = extractor.extract()
            logging.debug(f"Extractor {extractor.__class__.__name__} returned {len(extracted)} resources")
            resources.extend(extracted)

        return resources

    def transform(self, resources: Sequence[BaseResource]) -> List[BaseResource]:
        """Sequentially apply transformers."""

        resources: List[BaseResource] = list(resources)

        for transformer in self.transformers:
            logging.debug(f"Running transformer: {transformer.__class__.__name__} with {len(resources)} resources")
            resources = transformer.transform(resources)
            logging.debug(f"Transformer {transformer.__class__.__name__} output {len(resources)} resources")

        return resources

    def load(self, resources: Sequence[BaseResource]) -> None:
        """Persist resources using the loaders."""

        for loader in self.loaders:
            logging.debug(f"Running loader: {loader.__class__.__name__} with {len(resources)} resources")
            loader.load(list(resources))

    ################################################################

    def run(self) -> None:
        """
        Execute the pipeline for this course.

        Default guidance:
          1) run all extractors -> List[Resource]
          2) run transformers in order -> List[Resource]
          3) run all loaders

        Subclasses may override for custom logic, filtering, or branching.
        """

        logging.info(f"Starting pipeline for course {self.course_code}")

        # Extract
        resources = self.extract()
        logging.info(f"Extracted {len(resources)} resources")

        # Transform
        resources = self.transform(resources)
        logging.info(f"Transformed to {len(resources)} resources")

        # Load
        self.load(resources)
        logging.info(f"Finished pipeline for course {self.course_code}")
