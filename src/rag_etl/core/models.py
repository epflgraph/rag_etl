"""ModelConfig: one model invocation config, part of the declarative data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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
    judge: ModelConfig | None = None      # type/subtype/is_solution/cut inference (judge)
    annotator: ModelConfig | None = None  # exercise number + is-solution per chunk (per_exercise)
    embedding: ModelConfig | None = None  # chunk embeddings (embed)
