"""The fixed sequence orchestrator.

Extract → normalize → materialize → judge → cut → embed, then load:
one sequence for every project, the steps constructed from the spec.
Extractors take nothing and produce trees, transformers take trees and
produce trees, loaders take trees and produce nothing. Failures
propagate: either every source is indexed or the run stops loud.
"""

from __future__ import annotations

import logging
import os
import shutil

from datetime import UTC, datetime
from pathlib import Path

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.extractors import Extractor
from rag_etl.loaders import ElasticsearchLoader, Loader, ManifestLoader
from rag_etl.projects.spec import ProjectSpec
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.cut.cut_transformer import CutTransformer
from rag_etl.transformers.embed.embed import EmbedTransformer
from rag_etl.transformers.judge.judge_transformer import JudgeTransformer
from rag_etl.transformers.materialize.materialize_transformer import MaterializeTransformer
from rag_etl.transformers.materialize.videos import GraphAISlideDetector
from rag_etl.transformers.normalize.normalize_transformer import NormalizeTransformer

logger = logging.getLogger(__name__)

# The transformers that keep a cache, by name: what `--refresh` accepts
CACHED_STEPS = (NormalizeTransformer.name, MaterializeTransformer.name, CutTransformer.name, EmbedTransformer.name)


def material_dir(spec: ProjectSpec) -> Path:
    """The project's material folder: sources, work area and manifests."""
    return spec.dir or Path(os.environ["BASE_PATH"]) / spec.id


def run_pipeline(spec: ProjectSpec, dry: bool = False, refresh: str | None = None) -> None:
    """Run the fixed sequence over one project's sources.

    Every run writes a manifest; a dry run's carries a ``_dry`` suffix
    and nothing is published. Failures propagate loud.
    """
    # The work area is pure scratch: everything in it is re-derivable from
    # the sources and the cache, so every run starts from a clean work
    # area — and a crashed run's leftovers never survive into the new one
    shutil.rmtree(material_dir(spec) / "work", ignore_errors=True)

    roots = extract(spec.sources)
    roots = transform(roots, spec, refresh)
    load(roots, spec, dry)


def extract(sources: tuple[Extractor, ...]) -> list[Resource]:
    """Extract every source into its own tree, failing loud on any of them."""
    roots: list[Resource] = []
    for source in sources:
        logger.info(f"Extracting {type(source).__name__}")
        roots.append(source.extract())
    return roots


def transform(roots: list[Resource], spec: ProjectSpec, refresh: str | None = None) -> list[Resource]:
    """Run the fixed transform sequence over the trees."""

    def cache(name: str) -> Cache:
        return Cache(spec.id, name, refresh=refresh == name)

    transformers: list[Transformer] = [
        NormalizeTransformer(cache=cache(NormalizeTransformer.name)),
        MaterializeTransformer(
            cache=cache(MaterializeTransformer.name),
            work_root=material_dir(spec) / "work",
            detector=GraphAISlideDetector(),
            mediaspace_url=os.environ.get("DEFAULT_MEDIASPACE_BASE_URL"),
            language=spec.language,
        ),
        JudgeTransformer(overrides=spec.overrides),
        CutTransformer(cache=cache(CutTransformer.name)),
        EmbedTransformer(cache=cache(EmbedTransformer.name)),
    ]

    for transformer in transformers:
        logger.info(f"Running {transformer.name}")
        roots = transformer.transform(roots)

    return roots


def load(roots: list[Resource], spec: ProjectSpec, dry: bool = False) -> None:
    """Record the run in a manifest and, unless dry, publish it."""
    now = datetime.now(UTC)

    loaders: list[Loader] = [ManifestLoader(material_dir(spec) / "output", project=spec.id, timestamp=now, dry=dry)]
    if not dry:
        loaders.append(ElasticsearchLoader(project=spec.id, timestamp=now, language=spec.language))

    for loader in loaders:
        logger.info(f"Loading {type(loader).__name__}")
        loader.load(roots)
