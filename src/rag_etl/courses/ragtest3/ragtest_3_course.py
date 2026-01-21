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

from rag_etl.courses.ragtest3.ragtest_3_metadata_transformer import RAGTEST3MetadataTransformer


class RAGTEST3Course(BaseCourse):
    """
    Course-specific pipeline for RAGTEST 3.
    """

    def __init__(self):
        self.metadata_transformer = RAGTEST3MetadataTransformer(cache=self.course_code)

    @property
    def extractors(self) -> List[BaseExtractor]:
        """Single Moodle extractor."""
        return [
            MoodleExtractor(
                moodle_course_id=self.metadata_transformer.moodle_course_id,
                moodle_base_path=self.metadata_transformer.moodle_base_path
            )
        ]

    @property
    def transformers(self) -> List[BaseTransformer]:
        """Single transformer that converts PDFs into Markdown text."""
        return [
            self.metadata_transformer,
            ExtractZipTransformer(cache=self.course_code),
            JupyterToMarkdownTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(type_subtypes=self.metadata_transformer.pdf_to_markdown_type_subtypes, cache=self.course_code),
            SplitExercisesTransformer(type_subtypes=self.metadata_transformer.split_exercises_type_subtypes, cache=self.course_code),
        ]

    @property
    def loaders(self) -> List[BaseLoader]:
        """No loaders defined for this course."""
        return [
            ContentMetadataLoader(
                output_path=self.metadata_transformer.output_path,
                course_info=self.metadata_transformer.course_info
            )
        ]


if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    course = BaseCourse.from_code('RAGTEST3')
    course.run()
