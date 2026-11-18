"""The loader contract: take the transformed trees, produce nothing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rag_etl.core import Resource


def alias_name(project: str) -> str:
    """The stable name consumers query: rag_{project}."""
    return f"rag_{project}"


def index_name(project: str, timestamp: datetime) -> str:
    """The fresh index one run publishes into: rag_{project}_{YYYYMMDD}_{HHMMSS}."""
    return f"{alias_name(project)}_{timestamp:%Y%m%d_%H%M%S}"


@dataclass(frozen=True)
class Loader(ABC):
    """Publishes or records the trees the last transformer handed over.

    A loader only reads the trees; what it produces lives outside the
    pipeline (an index, a file in the project's output folder).
    """

    project_dir: Path = field(kw_only=True)

    @abstractmethod
    def load(self, roots: list[Resource]) -> None:
        """Load every tree."""
        raise NotImplementedError
