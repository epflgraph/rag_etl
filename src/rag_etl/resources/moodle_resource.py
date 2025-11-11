from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from rag_etl.resources.base_resource import BaseResource


@dataclass
class MoodleResource(BaseResource):
    source = 'moodle'

    section_title: Optional[str] = None
    section_text: Optional[str] = None
