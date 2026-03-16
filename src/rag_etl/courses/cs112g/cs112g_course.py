from __future__ import annotations

from datetime import date

import logging
from rag_etl.courses import BaseCourse
from rag_etl.extractors import (
    BaseExtractor,
    MOOCExtractor,
    MoodleExtractor,
    EdDiscussionExtractor,
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

from typing import Tuple, List


class CS112gCourse(BaseCourse):
    """
    Course-specific pipeline for CS-112(g).
    """

    course_info = {
        "course_title": "Programmation orientée objet",
        "course_id": "CS112g",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=5571",
        "coursebook_link": "https://edu.epfl.ch/coursebook/fr/programmation-orientee-objet-CS-112-G",
        "course_language": "French",
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
        "PROJECT": {
            "type": "practice",
            "subtype": "project",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
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
        "MOOC_ASSIGNMENT": {
            "type": "practice",
            "subtype": "assignment",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_TUTORIAL_SOLUTION": {
            "type": "practice",
            "subtype": "tutorial",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
            "is_solution": True,
        },
        "MOOC_LECTURE_NOTES": {
            "type": "theory",
            "subtype": "lecture_notes",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_RECOMMENDED_READING": {
            "type": "theory",
            "subtype": "recommended_reading",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
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
        "MOOC_EXERCISES": {
            "type": "practice",
            "subtype": "exercises",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_EXERCISES_SOLUTION": {
            "type": "practice",
            "subtype": "exercises",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
            "is_solution": True,
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
    }

    semester_start_date = date(year=2026, month=2, day=16)
    semester_end_date = date(year=2026, month=6, day=22)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mooc_base_path = f"{course_path}/mooc"

    mime_types = mt.DEFAULT_MIME_TYPES

    moodle_course_id = 5571

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
                mooc_base_path=self.mooc_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=(self.mime_types + [mt.MP4, mt.JSON]),
            ),
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=self.mime_types,
            ),
            # EdDiscussionExtractor(
            #     ed_discussion_base_path=self.course_path,
            #     tags=self.tag_metadata.keys(),
            #     tag_metadata=self.tag_metadata,
            #     mime_types=(mt.DEFAULT_MIME_TYPES),
            #     academic_year="2024-2025",
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
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("CS112g")
    course.run()
