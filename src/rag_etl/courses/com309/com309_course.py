from __future__ import annotations

import logging
from typing import List

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

from rag_etl.courses.com309.com309_metadata_transformer import COM309MetadataTransformer


class COM309Course(BaseCourse):
    """
    Course-specific pipeline for COM309.
    """

    course_info = {
        "course_title": "Introduction to quantum information processing",
        "course_id": "COM-309",
        "academic_course": "2025-2026",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/COM-309",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/introduction-to-quantum-information-processing-COM-309"
    }

    moodle_dump_path = '/Users/hera/Documents/EPFL/2025/COM-309/moodle'

    output_path = '/Users/hera/Documents/EPFL/2025/COM-309'


    pdf_to_markdown_type_subtypes = [
        ('exam', 'previous_year_exam'),
        ('practice', 'homework'),
    ]

    split_exercises_type_subtypes = [
        ('exam', 'previous_year_exam'),
        ('practice', 'homework'),
    ]

    @property
    def extractors(self) -> List[BaseExtractor]:
        """Single Moodle extractor for COM309 course content."""
        return [
            MoodleExtractor(moodle_dump_path=self.moodle_dump_path)
        ]

    @property
    def transformers(self) -> List[BaseTransformer]:
        """Single transformer that converts PDFs into Markdown text."""
        return [
            COM309MetadataTransformer(),
            ExtractZipTransformer(mime_types=['application/pdf', 'application/x-ipynb+json']),
            JupyterToMarkdownTransformer(),
            PDFToMarkdownTransformer(type_subtypes=self.pdf_to_markdown_type_subtypes),
            SplitExercisesTransformer(type_subtypes=self.split_exercises_type_subtypes),
        ]

    @property
    def loaders(self) -> List[BaseLoader]:
        """No loaders defined for this course."""
        return [
            ContentMetadataLoader(output_path=self.output_path, course_info=self.course_info)
        ]


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    course = BaseCourse.from_code('COM309')
    course.run()
