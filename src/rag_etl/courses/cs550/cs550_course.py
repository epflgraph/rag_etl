from __future__ import annotations

from datetime import date

import logging

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MoodleExtractor
from rag_etl.transformers import (
    BaseTransformer,
    ExtractZipTransformer,
    JupyterToMarkdownTransformer,
    PDFToMarkdownTransformer,
    SplitExercisesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class CS550Course(BaseCourse):
    """
    Course-specific pipeline for CS550
    """

    course_info = {
        "course_title": "Formal verification",
        "course_id": "CS550",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=13051",
        "coursebook_link": "https://edu.epfl.ch/coursebook/fr/formal-verification-CS-550",
    }

    # [THEORY_SLIDES]
    # [SERIE_x]
    # [SERIE_x_SOLUTION]
    # [EXAM_xxxx]
    # [EXAM_xxxx_SOLUTION]

    tag_metadata = {
        "THEORY": {  # THEORY added just in case
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": True,
            "split_exercises": False,
        },
        "THEORY_SLIDES": {
            "type": "theory",
            "subtype": "theory_slides",
            "one_chunk_per_page": True,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": True,
            "split_exercises": False,
        },
        "SERIE": {
            "type": "practice",
            "subtype": "serie",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "SERIE_SOLUTION": {
            "type": "practice",
            "subtype": "serie",
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
    }

    semester_start_date = date(year=2026, month=9, day=7)
    semester_end_date = date(year=2027, month=1, day=30)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    ################################################################

    moodle_course_id = 13051

    moodle_base_path = f"{course_path}/moodle"

    ################################################################

    @property
    def pdf_to_markdown_type_subtypes(self) -> list[tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get("type"), self.tag_metadata[tag].get("subtype"))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get("pdf_to_markdown")
        ]

    @property
    def split_exercises_type_subtypes(self) -> list[tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get("type"), self.tag_metadata[tag].get("subtype"))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get("split_exercises")
        ]

    @property
    def extractors(self) -> list[BaseExtractor]:
        return [
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=(mt.DEFAULT_MIME_TYPES),
            )
        ]

    @property
    def transformers(self) -> list[BaseTransformer]:
        return [
            ExtractZipTransformer(cache=self.course_code),
            JupyterToMarkdownTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(type_subtypes=self.pdf_to_markdown_type_subtypes, cache=self.course_code),
            SplitExercisesTransformer(type_subtypes=self.split_exercises_type_subtypes, cache=self.course_code),
        ]

    @property
    def loaders(self) -> list[BaseLoader]:
        return [ContentMetadataLoader(course_path=self.course_path, course_info=self.course_info)]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("CS550")
    course.run()
