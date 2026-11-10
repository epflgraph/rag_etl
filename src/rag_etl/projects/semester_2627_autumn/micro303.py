"""The MICRO-303 project (Microfabrication I)."""

from __future__ import annotations

from pathlib import Path

from rag_etl.core import Override
from rag_etl.extractors import LocalFolderExtractor, MOOCExtractor, MoodleExtractor  # , EdDiscussionExtractor
from rag_etl.projects import Project

MICRO303 = Project(
    id="micro303",
    language="en",
    dir=Path("MICRO303"),
    # What the MOOC export's tags used to say, until the judge infers it
    overrides=(
        # The quizzes the extractor writes out, statement and solutions alike
        Override('path="*/mooc/*" AND filename="*.quiz*.md"', type="practice", subtype="mooc_quiz", cut="whole_document"),
        # The lecture videos, one chunk per slide like any other video
        Override('path="*/mooc/*" AND mime_type="video/mp4"', type="theory", subtype="mooc_video", cut="per_child"),
        # The untitled blocks of the export: on every video page the one line
        # "Download here the PDF document corresponding to the above video.",
        # and once a hint about the platform's search bar
        Override('title="*Raw HTML"', exclude=True),
    ),
    sources=(
        MoodleExtractor(dir=Path("moodle"), course_id=19283),
        MOOCExtractor(
            dir=Path("mooc"),
            # Week one opens at chapter 2 of the course, the ones
            # before and after it being a preamble and a closing section
            first_week_chapter=2,
            week_count=7,
            course_url="https://courseware.epfl.ch/learning/course/course-v1:EPFL+mems+2023/home",
            language="en",
        ),
        LocalFolderExtractor(dir=Path("local")),
        # Ed Discussion, once the course's export is in place. Previous year's
        # Q&A; which categories to keep (formerly theory and exam) and how to
        # label them become overrides, e.g.
        # Override('path="*/ed_discussion/*"', cut="whole_document") and
        # Override('title="Logistics"', exclude=True)
        # EdDiscussionExtractor(
        #     dir=Path("ed_discussion"),
        #     academic_year="2025_2026",
        #     language="english",
        # ),
    ),
)
