from __future__ import annotations

from datetime import date

import logging

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MoodleExtractor, LocalFolderExtractor
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


class PHYS201aCourse(BaseCourse):
    """
    Course-specific pipeline for PHYS201a
    """

    course_info = {
        "course_title": "Physique générale : électromagnétisme",
        "course_id": "PHYS201a",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=14288",
        "coursebook_link": "https://edu.epfl.ch/coursebook/fr/physique-generale-electromagnetisme-PHYS-201-A",
    }

    # [QUIZ_x] <- weekly slides with a quiz and its solutions
    # [SERIE_x] <- a weekly series of exercises
    # [SERIE_x_SOLUTION] <- the solutions to these weekly series
    # [EXAM_xxxx] <- past exams
    # [EXAM_xxxx_SOLUTION] <- the solutions to these past exams
    # [MIDTERM_EXAM_xxxx] <- past midterm exams (including, later, this year's midterm)
    # [MIDTERM_EXAM_xxxx_SOLUTION] <- past midterm exams (including, later, this year's midterm)

    tag_metadata = {
        "THEORY": {  # THEORY added just in case
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
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
        "MIDTERM_EXAM": {
            "type": "exam",
            "subtype": "midterm_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "MIDTERM_EXAM_SOLUTION": {
            "type": "exam",
            "subtype": "midterm_exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "QUIZ": {
            "type": "practice",
            "subtype": "quiz",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_solution": True,
        },
    }

    semester_start_date = date(year=2026, month=9, day=7)
    semester_end_date = date(year=2027, month=1, day=30)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    ################################################################

    moodle_course_id = 14288

    moodle_base_path = f"{course_path}/moodle"

    mime_types = mt.DEFAULT_MIME_TYPES

    local_folder_base_path = f"{course_path}/local"

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
                mime_types=self.mime_types,
            ),
            LocalFolderExtractor(
                folder_base_path=self.local_folder_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=self.mime_types,
            ),
            # EdDiscussionExtractor(
            #     ed_discussion_base_path=self.course_path,
            #     tags=self.tag_metadata.keys(),
            #     tag_metadata=self.tag_metadata,
            #     mime_types=self.mime_types,
            #     academic_year="2025-2026",
            #     categories=[
            #         "theory",
            #         "practice",
            #         "exam",
            #     ],
            #     language=self.course_info["course_language"],
            #     semester=self.course_info["semester"],
            #     include_student_endorsed=True,
            # ),
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

    course = BaseCourse.from_code("PHYS201a")
    course.run()
