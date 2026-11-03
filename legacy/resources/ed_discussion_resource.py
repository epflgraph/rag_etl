from __future__ import annotations
from dataclasses import dataclass
from rag_etl.resources.base_resource import BaseResource


@dataclass
class EdDiscussionResource(BaseResource):
    """Resource representing an Ed Discussion Q&A thread."""

    source = "ed_discussion"
    category: str | None = None
    path_to_intermediate_json_file: str | None = None
