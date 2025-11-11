from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from rag_etl.resources.base_resource import BaseResource


@dataclass
class MOOCResource(BaseResource):
    source = 'mooc'

    chapter: Optional[str] = None
    subchapter: Optional[str] = None
