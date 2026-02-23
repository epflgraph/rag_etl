from __future__ import annotations

from datetime import date

import logging
from rag_etl.courses import BaseCourse
from rag_etl.extractors import (
    BaseExtractor,
    MOOCExtractor,
    MoodleExtractor,
)
from rag_etl.transformers import (
    BaseTransformer,
    PDFToMarkdownTransformer,
    VideoToJSONTransformer,
    ExtractZipTransformer,
    SplitExercisesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG
from typing import List, Tuple


class ENV342Course(BaseCourse):
    """
    Course-specific pipeline for ENV-342.
    """

    course_info = {
        "course_title": "Geographic Information System",
        "course_id": "env342",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/geographic-information-system-gis-ENV-342",
        "course_language": "French",  # ask Aitor to add this
    }

    # [SLIDES]
    # [POLYCOPIE]
    # [EXERCISE_SIG_x]
    # [EXERCISE_GEO_x]
    # [PROJET_x]
    # [EXAMEN_x]

    tag_metadata = {
        "SLIDES": {
            "type": "theory",
            "subtype": "lecture_slides",
            "one_chunk_per_page": True,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "POLYCOPIE": {
            "type": "theory",
            "subtype": "polycopie",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "EXERCICE_SIG": {
            "type": "practice",
            "subtype": "exercice_sig",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "EXERCICE_SIG_SOLUTION": {
            "type": "practice",
            "subtype": "exercice_sig",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "EXERCICE_GEO": {
            "type": "practice",
            "subtype": "exercice_geo",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "EXERCICE_GEO_SOLUTION": {
            "type": "practice",
            "subtype": "exercice_geo",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "EXAM": {
            "type": "exam",
            "subtype": "previous_year_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "EXAM_SOLUTION": {
            "type": "exam",
            "subtype": "previous_year_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "PROJET": {
            "type": "practice",
            "subtype": "projet",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "MOOC_VIDEO": {
            "type": "theory",
            "subtype": "video_lecture",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": True,
            "is_gemini_processed_video": True,
            "processing_method": "gemini",
            "model": "gemini-2.5-pro",
        },
        "MOOC_QUIZ": {
            "type": "practice",
            "subtype": "quiz",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
    }

    semester_start_date = date(year=2026, month=2, day=16)
    semester_end_date = date(year=2026, month=6, day=22)

    # now course_path
    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mooc_base_path_gis_1 = f"{course_path}/mooc_gis_1"
    mooc_base_path_gis_2 = f"{course_path}/mooc_gis_2"

    moodle_course_id = 4081

    mime_types = mt.DEFAULT_MIME_TYPES + [mt.PYTHON_SOURCE, mt.MATLAB_SOURCE]

    moodle_base_path = f"{course_path}/moodle"

    @property
    def pdf_to_markdown_type_subtypes(self) -> list[tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get("type"), self.tag_metadata[tag].get("subtype"))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get("pdf_to_markdown")
        ]

    @property
    def split_exercises_type_subtypes(self) -> List[Tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get("type"), self.tag_metadata[tag].get("subtype"))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get("split_exercises")
        ]

    @property
    def extractors(self) -> list[BaseExtractor]:
        """Single MOOC extractor."""
        return [
            MOOCExtractor(
                mooc_base_path=self.mooc_base_path_gis_1,
                tag_metadata=self.tag_metadata,
                mime_types=(mt.DEFAULT_MIME_TYPES + [mt.MP4, mt.MD, mt.JSON]),
            ),
            MOOCExtractor(
                mooc_base_path=self.mooc_base_path_gis_2,
                tag_metadata=self.tag_metadata,
                mime_types=(mt.DEFAULT_MIME_TYPES + [mt.MP4, mt.MD, mt.JSON]),
            ),
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=self.mime_types,
            ),
        ]

    @property
    def transformers(self) -> list[BaseTransformer]:
        """Single transformer that converts PDFs into Markdown text."""
        return [
            VideoToJSONTransformer(cache=self.course_code),
            ExtractZipTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(
                type_subtypes=self.pdf_to_markdown_type_subtypes, cache=self.course_code
            ),
            SplitExercisesTransformer(
                type_subtypes=self.split_exercises_type_subtypes, cache=self.course_code
            ),
        ]

    @property
    def loaders(self) -> list[BaseLoader]:
        """No loaders defined for this course."""
        return [
            ContentMetadataLoader(
                course_path=self.course_path,
                course_info=self.course_info,
            )
        ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("ENV342")
    course.run()
