"""The fixed sequence orchestrator.

Extract → normalize → materialize → judge → cut → embed + load: one
sequence for every project, the transforms constructed from the spec.
Failures propagate: either every source is indexed or the run stops loud.
"""

from __future__ import annotations

import logging

from rag_etl.cache import Cache
from rag_etl.config import CONFIG
from rag_etl.core import Resource
from rag_etl.extractors import Extractor
from rag_etl.projects.spec import ProjectSpec
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.cut.cut_transformer import CutTransformer
from rag_etl.transformers.embed.embed import chunks_of, embed
from rag_etl.transformers.judge.judge_transformer import JudgeTransformer
from rag_etl.transformers.materialize.materialize_transformer import MaterializeTransformer
from rag_etl.transformers.materialize.videos import GraphAISlideDetector
from rag_etl.transformers.normalize.normalize_transformer import NormalizeTransformer

logger = logging.getLogger(__name__)


def run_pipeline(spec: ProjectSpec, dry: bool = False, refresh: str | None = None) -> None:
    """Run the fixed sequence over one project's sources.

    The loaders run inside the pipeline (Phase 5) and publish the index;
    nothing content-wise comes back — failures propagate loud. A dry run
    publishes nothing.
    """
    roots = extract(spec.sources)
    roots = transform(roots, spec, refresh)

    pairs = embed(chunks_of(roots), Cache("embed", refresh=refresh == "embed"))
    logger.info(f"{spec.id}: {len(pairs)} chunks embedded")
    # The loaders consume the pairs in Phase 5


def extract(sources: tuple[Extractor, ...]) -> list[Resource]:
    """Extract every source into its own tree, failing loud on any of them."""

    roots: list[Resource] = []
    for source in sources:
        logger.info(f"Extracting {type(source).__name__}")
        roots.append(source.extract())

    return roots


def transform(roots: list[Resource], spec: ProjectSpec, refresh: str | None = None) -> list[Resource]:
    """Run the fixed transform sequence over the trees."""

    normalize = NormalizeTransformer(
        cache=Cache("normalize", refresh=refresh == "normalize"),
    )
    materialize = MaterializeTransformer(
        cache=Cache("materialize", refresh=refresh == "materialize"),
        detector=GraphAISlideDetector(),
        mediaspace_url=CONFIG.get("DEFAULT_MEDIASPACE_BASE_URL"),
        language=spec.language,
    )
    judge = JudgeTransformer(exceptions=spec.exceptions)
    cut = CutTransformer(
        cache=Cache("cut", refresh=refresh == "cut"),
    )

    sequence: list[Transformer] = [normalize, materialize, judge, cut]

    for step in sequence:
        name = type(step).__name__
        logger.info(f"Running {name}")
        roots = step.transform(roots)

    return roots
