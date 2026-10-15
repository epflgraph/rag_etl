"""The spec of MICRO-303 (Microfabrication I), 2026-2027 autumn."""

from __future__ import annotations

from pathlib import Path

import ragtools.mime_types as mt

from rag_etl.config import config_value
from rag_etl.extractors import EdDiscussionExtractor, LocalFolderExtractor, MOOCExtractor, MoodleExtractor
from rag_etl.projects.spec import ProjectSpec

# The MOOC export's own labels: type/subtype/cut the export itself infers
# (unlike Moodle files, these are inferred source-side on purpose)
MOOC_TAG_METADATA = {
    "MOOC_QUIZ": {
        "type": "practice",
        "subtype": "mooc_quiz",
        "one_chunk_per_page": False,
        "one_chunk_per_doc": True,
    },
    "MOOC_VIDEO": {
        "type": "theory",
        "subtype": "mooc_video",
        "one_chunk_per_page": False,
        "one_chunk_per_doc": True,
    },
}

course_path = Path(config_value("BASE_PATH")) / "MICRO303"

MICRO303 = ProjectSpec(
    id="micro303",
    language="en",
    sources=(
        MoodleExtractor(dir=course_path / "moodle", course_id=19283),
        MOOCExtractor(
            dir=course_path / "mooc",
            tag_metadata=MOOC_TAG_METADATA,
            mime_types=tuple(mt.DEFAULT_MIME_TYPES + [mt.MP4, mt.JSON]),
            # Week one opens at chapter 2 of the course, the ones
            # before and after it being a preamble and a closing section
            first_week_chapter=2,
            week_count=7,
            include_untagged_documents=True,
            course_url="https://courseware.epfl.ch/learning/course/course-v1:EPFL+mems+2023/home",
            theory_tag="THEORY",
            theory_slides_tag="THEORY_SLIDES",
            language="en",
        ),
        LocalFolderExtractor(dir=course_path / "local"),
        EdDiscussionExtractor(
            ed_discussion_base_path=str(course_path),
            # Previous year's Q&A: dates are inherited from this year's documents via the catalogue
            academic_year="2025_2026",
            categories=[
                "theory",
                "exam",
            ],
            language="en",
            semester=1,
            include_student_endorsed=False,
        ),
    ),
)
