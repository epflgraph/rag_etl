"""ProjectSpec: the one declarative spec per project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rag_etl.core.rules import Rule
from rag_etl.extractors import Extractor


@dataclass(frozen=True)
class ModelConfig:
    """One model invocation config: the model and its parameter overrides.

    The recommended parameters for the model come from
    ``ragtools.llm.defaults``; ``params`` overrides them per call.
    """

    model: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfigs:
    """Model + parameters per concern; every model invocation takes these from the spec."""

    ocr: ModelConfig | None = None        # vision reads: PDF pages, frames, images (materialize)
    judge: ModelConfig | None = None      # type/subtype/is_solution/cut inference (Phase 4)
    annotator: ModelConfig | None = None  # exercise number + is-solution per chunk (Phase 4)
    embedding: ModelConfig | None = None  # chunk embeddings (Phase 4)


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
    models: ModelConfigs = ModelConfigs()    # OCR, judge, annotator, embedding
