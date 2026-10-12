"""The fixed sequence orchestrator.

Extract → normalize → materialize → judge → cut → embed + load: one
sequence for every project, the transforms constructed from the spec.
Failures propagate: either every source is indexed or the run stops loud.
"""

from __future__ import annotations

import logging

from rag_etl.cache import Cache
from rag_etl.config import CONFIG, config_value
from rag_etl.core import Resource
from rag_etl.extractors import Extractor
from rag_etl.projects.spec import ProjectSpec
from rag_etl.transformers.base import Transformer
from rag_etl.transformers.materialize.materialize_transformer import MaterializeTransformer
from rag_etl.transformers.materialize.videos import GraphAISlideDetector
from rag_etl.transformers.normalize.normalize_transformer import NormalizeTransformer
from ragtools import RCPClient

logger = logging.getLogger(__name__)


def run_pipeline(spec: ProjectSpec, dry: bool = False, refresh: str | None = None) -> list[Resource]:
    """Run the fixed sequence over one project's sources and return the trees.

    Judge → cut → embed + load land in Phases 4 and 5; until then the
    sequence runs extract, normalize and materialize. A dry run publishes
    nothing (loaders arrive in Phase 5).
    """

    roots = extract(spec.sources)
    roots = transform(roots, spec, refresh)
    return roots


def extract(sources: tuple[Extractor, ...]) -> list[Resource]:
    """Extract every source into its own tree, failing loud on any of them."""

    roots: list[Resource] = []
    for source in sources:
        logger.info(f"Extracting {type(source).__name__}")
        roots.append(source.extract())

    return roots


def transform(roots: list[Resource], spec: ProjectSpec, refresh: str | None = None) -> list[Resource]:
    """Run the fixed transform sequence over the trees."""

    llm = RCPClient(base_url=config_value("RCP_BASE_URL"), api_key=config_value("RCP_API_KEY"))
    ocr_model = ocr_model_of(spec)

    normalize = NormalizeTransformer(
        cache=Cache("normalize", refresh=refresh == "normalize"),
        llm=llm,
        vision_model=ocr_model,
    )
    materialize = MaterializeTransformer(
        cache=Cache("materialize", refresh=refresh == "materialize"),
        llm=llm,
        vision_model=ocr_model,
        detector=GraphAISlideDetector(),
        mediaspace_url=CONFIG.get("DEFAULT_MEDIASPACE_BASE_URL"),
        language=spec.language,
    )

    sequence: list[Transformer] = [normalize, materialize]
    # judge → cut → embed + load land in Phases 4 and 5

    for step in sequence:
        name = type(step).__name__
        logger.info(f"Running {name}")
        roots = step.transform(roots)

    return roots


def ocr_model_of(spec: ProjectSpec) -> str:
    """Return the OCR model the spec declares, failing loud when it has none."""
    if spec.models.ocr is None:
        raise ValueError(f"The spec of {spec.id} declares no OCR model; set models.ocr")
    return spec.models.ocr.model
