from __future__ import annotations

from datetime import date

import logging
from typing import List, Tuple
from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MOOCExtractor, MoodleExtractor
from rag_etl.transformers import (
    BaseTransformer,
    PDFToMarkdownTransformer,
    VideoToJSONTransformer,
    ExtractZipTransformer,
    SplitExercisesTransformer,
    JupyterToMarkdownTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class COM202Course(BaseCourse):
    """
    Course-specific pipeline for COM-202.
    """

    course_info = {
        "course_title": "Digital Signal Processing",
        "course_id": "com202",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "",
        "coursebook_link": None,
    }

    tag_metadata = {
        "SLIDES": {
            "type": "theory",
            "subtype": "lecture_slides",
            "one_chunk_per_page": True,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "HOMEWORK": {
            "type": "practice",
            "subtype": "homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "HOMEWORK_SOLUTION": {
            "type": "practice",
            "subtype": "homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "NOTEBOOK": {
            "type": "practice",
            "subtype": "notebook",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_solution": True,
        },
        "CLASS_NOTES": {
            "type": "theory",
            "subtype": "class_notes",
            "one_chunk_per_page": True,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "CHEATSHEET": {
            "type": "theory",
            "subtype": "cheatsheet",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
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
        "RECOMMENDED_READING": {
            "type": "theory",
            "subtype": "recommended_reading",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "MOCK_EXAM": {
            "type": "exam",
            "subtype": "mock_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "MOCK_EXAM_SOLUTION": {
            "type": "exam",
            "subtype": "mock_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "HANDOUT": {
            "type": "theory",
            "subtype": "handout",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "BOOC": {
            "type": "theory",
            "subtype": "booc",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
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
        "MOOC_PRACTICE_HOMEWORK": {
            "type": "practice",
            "subtype": "mooc_practice_homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "MOOC_PRACTICE_HOMEWORK_SOLUTION": {
            "type": "practice",
            "subtype": "mooc_practice_homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_solution": True,
        },
        "MOOC_LECTURE_NOTES": {
            "type": "theory",
            "subtype": "mooc_lecture_notes",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
    }

    semester_start_date = date(year=2026, month=2, day=16)
    semester_end_date = date(year=2026, month=6, day=22)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    moodle_course_id = None
    mooc_base_path = f"{course_path}/mooc"

    mime_types = mt.DEFAULT_MIME_TYPES + [mt.C_SOURCE] + [mt.IPYNB] + [mt.PYTHON_SOURCE]

    moodle_course_id = 18253

    moodle_base_path = f"{course_path}/moodle"

    @property
    def pdf_to_markdown_type_subtypes(self) -> List[Tuple[str, str]]:
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
    def extractors(self) -> List[BaseExtractor]:
        """Single Moodle extractor."""
        return [
            MOOCExtractor(
                mooc_base_path=self.mooc_base_path,
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
    def transformers(self) -> List[BaseTransformer]:
        """Single transformer that converts PDFs into Markdown text."""
        return [
            VideoToJSONTransformer(cache=self.course_code),
            ExtractZipTransformer(cache=self.course_code),
            JupyterToMarkdownTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(
                type_subtypes=self.pdf_to_markdown_type_subtypes, cache=self.course_code
            ),
            SplitExercisesTransformer(
                type_subtypes=self.split_exercises_type_subtypes, cache=self.course_code
            ),
        ]

    @property
    def loaders(self) -> List[BaseLoader]:
        """No loaders defined for this course."""
        return [
            ContentMetadataLoader(
                course_path=self.course_path, course_info=self.course_info
            )
        ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format="[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("COM202")
    course.run()
