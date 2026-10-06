"""ProjectSpec: the one declarative spec per project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rag_etl.core.rules import Rule


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for one source an extractor reads (Phase 2 fixes the shape)."""

    kind: str  # moodle | mooc | mediaspace | local_folder | ed_discussion | git
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """One model invocation config (Phase 3 fixes the shape)."""

    model: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfigs:
    """Model + parameters per concern; every model invocation takes these from the spec."""

    ocr: ModelConfig | None = None
    judge: ModelConfig | None = None
    annotator: ModelConfig | None = None
    embedding: ModelConfig | None = None


@dataclass(frozen=True)
class ProjectSpec:
    """The declarative spec of one project.

    The judge infers type/subtype/is_solution/cut per resource; exceptions
    correct specific resources that would otherwise be mislabelled.
    """

    id: str                                  # lowercase, e.g. "com202"; names the index/alias rag_{id}
    exceptions: tuple[Rule, ...] = ()        # manual corrections for resources the judge would mislabel
    sources: tuple[SourceConfig, ...] = ()   # extractor configs (Phase 2)
    models: ModelConfigs = ModelConfigs()    # OCR, judge, annotator, embedding (Phase 3)
