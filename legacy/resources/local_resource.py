from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from rag_etl.resources.base_resource import BaseResource


@dataclass
class LocalResource(BaseResource):
    source = "local"

    tag: Optional[str] = None
