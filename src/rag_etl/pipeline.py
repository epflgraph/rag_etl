"""The fixed sequence orchestrator.

Extract → normalize → images → pdf → video → judge → cut → embed, then load:
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

from rag_etl.core import Resource, assign_ids
from rag_etl.loaders import ElasticsearchLoader, Loader, ManifestLoader
from rag_etl.projects import Project
from rag_etl.transformers import (
    CutTransformer,
    EmbedTransformer,
    ImagesTransformer,
    JudgeTransformer,
    NormalizeTransformer,
    PdfTransformer,
    Transformer,
    VideoTransformer,
)
from ragtools.graphai import GraphAISlideDetector

logger = logging.getLogger(__name__)

# The transformers that keep a cache, by name: what `--refresh` accepts
CACHED_STEPS = tuple(
    step.name for step in (ImagesTransformer, PdfTransformer, VideoTransformer, CutTransformer, EmbedTransformer)
)


def run_pipeline(project: Project, dry: bool = False, refresh: str | None = None) -> None:
    """Run the fixed sequence over one project's sources.

    The project's folder holds its sources, its work area, its cache and
    its manifests. It is declared relative to $BASE_PATH (the id when not
    declared) and resolved here, once; an absolute folder is taken as it is.
    Every run writes a manifest; a dry run's carries a ``_dry`` suffix
    and nothing is published. Failures propagate loud.
    """
    project_dir = Path(os.environ["BASE_PATH"]) / (project.dir or project.id)

    # The work area is pure scratch: everything in it is re-derivable from
    # the sources and the cache, so every run starts from a clean work
    # area — and a crashed run's leftovers never survive into the new one
    shutil.rmtree(project_dir / "work", ignore_errors=True)

    roots = extract(project, project_dir)
    roots = transform(roots, project, project_dir, refresh)
    load(roots, project, project_dir, dry)


def extract(project: Project, project_dir: Path) -> list[Resource]:
    """Extract every source into its own tree, failing loud on any of them.

    A source's folder is declared relative to the project's folder and
    resolved here; an absolute folder is taken as it is.
    """
    roots: list[Resource] = []
    for source in project.sources:
        source = dataclasses.replace(source, dir=project_dir / source.dir)
        logger.info(f"Extracting {type(source).__name__} into {source.dir}")
        roots.append(source.extract())
    return roots


def transform(
    roots: list[Resource], project: Project, project_dir: Path, refresh: str | None = None
) -> list[Resource]:
    """Run the fixed transform sequence over the trees."""
    transformers: list[Transformer] = [
        NormalizeTransformer(project_dir=project_dir),
        # Images first: the markdown it reads is the sources', never the pages and slides read after
        ImagesTransformer(project_dir=project_dir, refresh=refresh == ImagesTransformer.name),
        PdfTransformer(project_dir=project_dir, refresh=refresh == PdfTransformer.name),
        VideoTransformer(
            project_dir=project_dir,
            refresh=refresh == VideoTransformer.name,
            detector=GraphAISlideDetector(),
            mediaspace_url=os.environ.get("DEFAULT_MEDIASPACE_BASE_URL"),
            language=project.language,
        ),
        JudgeTransformer(project_dir=project_dir, overrides=project.overrides),
        CutTransformer(project_dir=project_dir, refresh=refresh == CutTransformer.name),
        EmbedTransformer(project_dir=project_dir, refresh=refresh == EmbedTransformer.name),
    ]

    for transformer in transformers:
        logger.info(f"Running {transformer.name}")
        roots = transformer.transform(roots)

    return roots


def load(roots: list[Resource], project: Project, project_dir: Path, dry: bool = False) -> None:
    """Record the run in a manifest and, unless dry, publish it."""
    # The trees are final: every chunk gets the id the manifest and the index both carry
    assign_ids(roots)
    now = datetime.now(UTC)

    loaders: list[Loader] = [ManifestLoader(project_dir=project_dir, project=project.id, timestamp=now, dry=dry)]
    if not dry:
        loaders.append(
            ElasticsearchLoader(project_dir=project_dir, project=project.id, timestamp=now, language=project.language)
        )

    for loader in loaders:
        logger.info(f"Loading {type(loader).__name__}")
        loader.load(roots)
