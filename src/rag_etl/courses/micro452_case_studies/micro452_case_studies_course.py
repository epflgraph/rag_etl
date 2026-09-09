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
    SplitPagesTransformer,
    SplitExercisesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class MICRO452CaseStudiesCourse(BaseCourse):
    """
    Course-specific pipeline for MICRO452 case studies
    """

    course_info = {
        "course_title": "Basics of mobile robotics — Case studies",
        "course_id": "MICRO452CaseStudies",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=15293",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/basics-of-mobile-robotics-MICRO-452",
        "course_language": "en",
    }

    # [THEORY] <- theory documents used for case studies
    # [CASE_STUDY_LNCM] <- case study for lecture N case M (e.g. L3C2)
    # [CASE_STUDY_LNCM_SOLUTION] <- solution for that case study
    # [CASE_STUDY_LNCM_MISCONCEPTIONS] <- common misconceptions for that case study

    tag_metadata = {
        "THEORY": {
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "split_pages": True,
        },
        "THEORY_SLIDES": {
            "type": "theory",
            "subtype": "theory_slides",
            "one_chunk_per_page": True,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "split_pages": True,
        },
        "CASE_STUDY": {
            "type": "practice",
            "subtype": "case_study",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "split_pages": True,
        },
        "CASE_STUDY_SOLUTION": {
            "type": "practice",
            "subtype": "case_study",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_solution": True,
        },
        "CASE_STUDY_MISCONCEPTIONS": {
            "type": "practice",
            "subtype": "case_study",
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

    moodle_course_id = 15293

    moodle_base_path = f"{course_path}/moodle"

    mime_types = mt.DEFAULT_MIME_TYPES

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
            SplitPagesTransformer(type_subtypes=self.page_split_type_subtypes, cache=self.course_code),
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

    course = BaseCourse.from_code("MICRO452CaseStudies")
    course.run()
