from __future__ import annotations

from dataclasses import dataclass, replace

from typing import Optional


@dataclass
class BaseResource:
    """Represents a single resource flowing through the ETL pipeline."""

    title: str
    source: str                 # e.g., "moodle", "mooc", etc.
    url: str                    # public url we can link to, where the resource is available
    path: str                   # path where the resource file is located in the file system
    mime_type: str              # e.g., "application/pdf", "text/markdown", "video/mp4"

    id: Optional[int] = None

    type: Optional[str] = None
    subtype: Optional[str] = None

    is_solution: bool = False
    is_qa: bool = False
    is_video: bool = False
    is_gemini_processed_video: bool = False
    srt_path: Optional[str] = None
    associated_video_lectures: Optional[list] = None

    week: Optional[int] = None
    number: Optional[str] = None
    sub_number: Optional[str] = None

    date: Optional[str] = None
    year: Optional[str] = None
    from_: Optional[str] = None
    until: Optional[str] = None

    processing_method: Optional[str] = None
    model: Optional[str] = None
    tikz: bool = False

    one_chunk_per_page: bool = False
    one_chunk_per_doc: bool = False

    original_link: Optional[str] = None
    pipeline_link: Optional[str] = None

    def copy_with(self, **changes):
        """Return a copy with specified fields replaced."""
        return replace(self, **changes)

    def metadata_dict(self) -> dict[str, Optional[object]]:
        """Return a dictionary containing only metadata-style fields."""
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "type": self.type,
            "subtype": self.subtype,
            "is_solution": self.is_solution,
            "is_qa": self.is_qa,
            "is_video": self.is_video,
            "is_gemini_processed_video": self.is_gemini_processed_video,
            "srt_path": self.srt_path,
            "associated_video_lectures": self.associated_video_lectures,
            "week": self.week,
            "number": self.number,
            "sub_number": self.sub_number,
            "from": self.from_,
            "until": self.until,
            "processing_method": self.processing_method,
            "model": self.model,
            "tikz": self.tikz,
            "one_chunk_per_page": self.one_chunk_per_page,
            "one_chunk_per_doc": self.one_chunk_per_doc,
            "original_link": self.original_link,
            "pipeline_link": self.pipeline_link,
        }
