from __future__ import annotations

from datetime import date

import logging
from typing import List, Tuple

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, LocalFolderExtractor
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


class CMIRESTRICTEDCourse(BaseCourse):
    """
    Course-specific pipeline for CMI (restricted).
    """

    course_info = {
        "course_title": "Centre of MicroNanoTechnology (restricted)",
        "course_id": "CMIRESTRICTED",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://www.epfl.ch/research/facilities/cmi/",
        "coursebook_link": "",
    }

    tags = [
        "WEBSITE",
        "PRIVATE",
    ]
    tag_metadata = {
        tag: {
            "type": "theory",
            "subtype": tag.lower(),
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        }
        for tag in tags
    }

    semester_start_date = date(year=2026, month=7, day=1)
    semester_end_date = date(year=2027, month=7, day=1)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mime_types = mt.DEFAULT_MIME_TYPES

    ################################################################

    local_folder_base_path = f"{course_path}/local"

    ################################################################

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
        return [
            LocalFolderExtractor(
                folder_base_path=self.local_folder_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=self.mime_types,
            ),
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
        return [ContentMetadataLoader(course_path=self.course_path, course_info=self.course_info)]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("CMIRESTRICTED")
    course.run()
