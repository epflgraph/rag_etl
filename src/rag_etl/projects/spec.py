"""ProjectSpec: the one declarative spec per project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_etl.core.rules import Rule
from rag_etl.extractors import Extractor


@dataclass(frozen=True)
class ProjectSpec:
    """The declarative spec of one project.

    Sources are extractor instances, each configured for its source; the
    judge infers type/subtype/is_solution/cut per resource; exceptions
    correct specific resources that would otherwise be mislabelled.
    """

    id: str                                  # lowercase, e.g. "com202"; names the index/alias rag_{id}
    language: str | None = None              # the project's language (materialize slide detection)
    exceptions: tuple[Rule, ...] = ()        # manual corrections for resources the judge would mislabel
    sources: tuple[Extractor, ...] = ()      # extractor instances, one per source
    dir: Path | None = None                  # the project's material folder; manifests land in dir/output
