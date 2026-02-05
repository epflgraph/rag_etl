from __future__ import annotations

from dataclasses import dataclass

from rag_etl.resources.base_resource import BaseResource


@dataclass
class MOOCResource(BaseResource):
    source = "mooc"

    chapter: str | None = None
    sequential: str | None = None
    vertical: str | None = None
    tag: str | None = None
