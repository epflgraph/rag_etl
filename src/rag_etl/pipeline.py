"""The fixed sequence orchestrator.

Extract → normalize → materialize → judge → cut → embed, then load:
one sequence for every project, the steps constructed from the project.
Extractors take nothing and produce trees, transformers take trees and
produce trees, loaders take trees and produce nothing. Failures
propagate: either every source is indexed or the run stops loud.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil

from datetime import UTC, datetime
from pathlib import Path

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.loaders import ElasticsearchLoader, Loader, ManifestLoader
from rag_etl.projects import Project
from rag_etl.transformers import (
    CutTransformer,
    EmbedTransformer,
    JudgeTransformer,
    MaterializeTransformer,
    NormalizeTransformer,
    Transformer,
)
from ragtools.graphai import GraphAISlideDetector

logger = logging.getLogger(__name__)

# The transformers that keep a cache, by name: what `--refresh` accepts
CACHED_STEPS = (NormalizeTransformer.name, MaterializeTransformer.name, CutTransformer.name, EmbedTransformer.name)


def project_dir(project: Project) -> Path:
    """The project's folder: its sources, its work area and its manifests.

    Declared relative to $BASE_PATH (the id when not declared), resolved
    here, at run time; an absolute folder is taken as it is.
    """
    return Path(os.environ["BASE_PATH"]) / (project.dir or project.id)


def run_pipeline(project: Project, dry: bool = False, refresh: str | None = None) -> None:
    """Run the fixed sequence over one project's sources.

    Every run writes a manifest; a dry run's carries a ``_dry`` suffix
    and nothing is published. Failures propagate loud.
    """
    # The work area is pure scratch: everything in it is re-derivable from
    # the sources and the cache, so every run starts from a clean work
    # area — and a crashed run's leftovers never survive into the new one
    shutil.rmtree(project_dir(project) / "work", ignore_errors=True)

    roots = extract(project)
    roots = transform(roots, project, refresh)
    load(roots, project, dry)


def extract(project: Project) -> list[Resource]:
    """Extract every source into its own tree, failing loud on any of them.

    A source's folder is declared relative to the project's folder and
    resolved here; an absolute folder is taken as it is.
    """
    roots: list[Resource] = []
    for source in project.sources:
        source = dataclasses.replace(source, dir=project_dir(project) / source.dir)
        logger.info(f"Extracting {type(source).__name__} into {source.dir}")
        roots.append(source.extract())
    return roots


def transform(roots: list[Resource], project: Project, refresh: str | None = None) -> list[Resource]:
    """Run the fixed transform sequence over the trees."""

    def cache(name: str) -> Cache:
        return Cache(project.id, name, refresh=refresh == name)

    transformers: list[Transformer] = [
        NormalizeTransformer(cache=cache(NormalizeTransformer.name)),
        MaterializeTransformer(
            cache=cache(MaterializeTransformer.name),
            work_root=project_dir(project) / "work",
            detector=GraphAISlideDetector(),
            mediaspace_url=os.environ.get("DEFAULT_MEDIASPACE_BASE_URL"),
            language=project.language,
        ),
        JudgeTransformer(overrides=project.overrides),
        CutTransformer(cache=cache(CutTransformer.name)),
        EmbedTransformer(cache=cache(EmbedTransformer.name)),
    ]

    for transformer in transformers:
        logger.info(f"Running {transformer.name}")
        roots = transformer.transform(roots)

    return roots


def load(roots: list[Resource], project: Project, dry: bool = False) -> None:
    """Record the run in a manifest and, unless dry, publish it."""
    now = datetime.now(UTC)

    loaders: list[Loader] = [
        ManifestLoader(project_dir(project) / "output", project=project.id, timestamp=now, dry=dry)
    ]
    if not dry:
        loaders.append(ElasticsearchLoader(project=project.id, timestamp=now, language=project.language))

    for loader in loaders:
        logger.info(f"Loading {type(loader).__name__}")
        loader.load(roots)
