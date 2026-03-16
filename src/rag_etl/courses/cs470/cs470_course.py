from __future__ import annotations

from datetime import date

import logging
from typing import List, Tuple

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


class CS470Course(BaseCourse):
    """
    Course-specific pipeline for CS470.
    """

    course_info = {
        "course_title": "Advanced computer architecture",
        "course_id": "CS470",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=15017",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/advanced-computer-architecture-CS-470"
    }

    tag_metadata = {
        'SLIDES': {
            'type': 'theory',
            'subtype': 'lecture_slides',
            'one_chunk_per_page': True,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'REFERENCE': {
            'type': 'theory',
            'subtype': 'recommended_reading',
            'one_chunk_per_page': False,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'EXAM': {
            'type': 'exam',
            'subtype': 'previous_year_exam',
            'one_chunk_per_page': False,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': True,
            'split_exercises': True,
        },
        'LAB_ASSIGNMENT': {
            'type': 'practice',
            'subtype': 'lab',
            'one_chunk_per_page': True,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'LAB_SLIDES': {
            'type': 'practice',
            'subtype': 'lab',
            'one_chunk_per_page': True,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'LAB_MATERIAL': {
            'type': 'practice',
            'subtype': 'lab',
            'one_chunk_per_page': False,
            'one_chunk_per_doc': True,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'LAB_SOLUTION': {
            'type': 'practice',
            'subtype': 'lab',
            'is_solution': True,
            'one_chunk_per_page': False,
            'one_chunk_per_doc': True,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'HOMEWORK': {
            'type': 'practice',
            'subtype': 'exercise',
            'one_chunk_per_page': False,
            'one_chunk_per_doc': True,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
        'SUBTITLES': {
            'type': 'theory',
            'subtype': 'video_subtitles',
            'one_chunk_per_page': False,
            'one_chunk_per_doc': False,
            'pdf_to_markdown': False,
            'split_exercises': False,
        },
    }

    semester_start_date = date(year=2026, month=2, day=16)
    semester_end_date = date(year=2026, month=6, day=22)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    ################################################################

    moodle_course_id = 15017

    moodle_base_path = f"{course_path}/moodle"

    ################################################################

    @property
    def pdf_to_markdown_type_subtypes(self) -> List[Tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get('type'), self.tag_metadata[tag].get('subtype'))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get('pdf_to_markdown')
        ]

    @property
    def split_exercises_type_subtypes(self) -> List[Tuple[str, str]]:
        return [
            (self.tag_metadata[tag].get('type'), self.tag_metadata[tag].get('subtype'))
            for tag in self.tag_metadata
            if self.tag_metadata[tag].get('split_exercises')
        ]

    @property
    def extractors(self) -> List[BaseExtractor]:
        return [
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=(mt.DEFAULT_MIME_TYPES),
            )
        ]

    @property
    def transformers(self) -> List[BaseTransformer]:
        return [
            ExtractZipTransformer(cache=self.course_code),
            JupyterToMarkdownTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(type_subtypes=self.pdf_to_markdown_type_subtypes, cache=self.course_code),
            SplitExercisesTransformer(type_subtypes=self.split_exercises_type_subtypes, cache=self.course_code),
        ]

    @property
    def loaders(self) -> List[BaseLoader]:
        return [
            ContentMetadataLoader(
                course_path=self.course_path,
                course_info=self.course_info
            )
        ]


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    course = BaseCourse.from_code('CS470')
    course.run()
