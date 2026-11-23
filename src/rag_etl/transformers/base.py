"""The transformer contract: reshape the trees handed over by the previous step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from rag_etl.cache import Cache
from rag_etl.core import Resource


@dataclass(frozen=True)
class Transformer(ABC):
    """Reshapes resource trees in place as one step of the fixed sequence.

    A transformer adds, updates and removes nodes (recursively) and hands
    the same trees back; the pipeline runs these in order, so the output
    of one is the input of the next. It writes files in its own folder of
    the project's work area and its own cache, never in the sources.
    """

    # The step's name: its work folder, its `--refresh` value, its log label
    name: ClassVar[str]

    # The recipe a caching step's results come from, naming its cache folder
    # (`pdf-ocr-1`): a step that caches declares it, unique across steps.
    # Bumped whenever the recipe changes what the same input yields
    # (prompts, model, settings): the new folder starts empty, the old one
    # stays for rollback
    cache_version: ClassVar[str]

    project_dir: Path = field(kw_only=True)
    refresh: bool = field(default=False, kw_only=True)  # bypass the cache's lookups, overwriting them

    @property
    def work_root(self) -> Path:
        """The step's scratch folder, wiped at every run start."""
        return self.project_dir / "work" / self.name

    @property
    def cache(self) -> Cache:
        """The step's cache, kept across runs, under its recipe's version."""
        return Cache(self.project_dir / "cache" / self.cache_version, refresh=self.refresh)

    @abstractmethod
    def transform(self, roots: list[Resource]) -> list[Resource]:
        """Perform the transformation over every tree."""
        raise NotImplementedError
