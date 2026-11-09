"""Projects: one declarative `Project` per project, found by its id in the semester folders."""

from __future__ import annotations

import importlib
import pkgutil

from dataclasses import dataclass
from pathlib import Path

from rag_etl.core.overrides import Override
from rag_etl.extractors import Extractor

__all__ = ["Project", "load_project"]


@dataclass(frozen=True)
class Project:
    """The declaration of one project.

    Sources are extractor instances, each configured for its source; the
    judge infers the metadata and the cut per resource; overrides correct
    specific resources that would otherwise be mislabelled.

    Folders are declared relative and resolved when a run starts: the
    project's folder against $BASE_PATH, each source's folder against the
    project's. An absolute path is taken as it is.
    """

    id: str                                  # lowercase, e.g. "micro303"; names the index/alias rag_{id}
    language: str | None = None              # the project's language (materialize slide detection)
    overrides: tuple[Override, ...] = ()     # manual corrections for resources the judge would mislabel
    sources: tuple[Extractor, ...] = ()      # extractor instances, one per source
    dir: Path | None = None                  # the project's folder under $BASE_PATH; defaults to the id


def load_project(project_id: str) -> Project:
    """Find a project by its id, failing loud when unknown.

    Projects live in the semester folders (e.g. ``semester_2627_autumn``),
    one file per project: named after the project in lowercase, defining
    the `Project` under its id in uppercase. A file that exists is a project.
    """

    for info in pkgutil.iter_modules(__path__):
        if not info.ispkg or not info.name.startswith("semester_"):
            continue

        try:
            module = importlib.import_module(f"{__name__}.{info.name}.{project_id.lower()}")
        except ModuleNotFoundError:
            continue

        project = getattr(module, project_id.upper(), None)
        if isinstance(project, Project):
            return project

    raise ValueError(f"Unknown project: {project_id}")
