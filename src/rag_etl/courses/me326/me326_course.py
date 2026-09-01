from __future__ import annotations

from datetime import date

import logging

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MoodleExtractor, MOOCExtractor, MediaspaceExtractor
from rag_etl.transformers import (
    BaseTransformer,
    ExtractZipTransformer,
    JupyterToMarkdownTransformer,
    PDFToMarkdownTransformer,
    SplitExercisesTransformer,
    ImageToMarkdownTransformer,
    MergeSlideTranscriptTransformer,
    VideoToFramesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


# ToDo: Add MOOC when Christian shares it
class ME326Course(BaseCourse):
    """
    Course-specific pipeline for ME326
    """

    course_info = {
        "course_title": "Automatique et commande numérique",
        "course_id": "ME326",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=16347",
        "coursebook_link": "https://edu.epfl.ch/coursebook/fr/automatique-et-commande-numerique-ME-326",
        "course_language": "fr",
    }

    # [THEORY_SLIDES]
    # [SERIE_x]
    # [SERIE_x_SOLUTION]
    tag_metadata = {
        "THEORY": {  # THEORY added just in case
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
        },
        "THEORY_SLIDES": {
            "type": "theory",
            "subtype": "theory_slides",
            "one_chunk_per_page": True,
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
            "subtype": "mooc_video",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": True,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MEDIASPACE_VIDEO": {
            "type": "theory",
            "subtype": "mediaspace_video",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": False,
            "split_exercises": False,
            "is_video": True,
            "is_gemini_processed_video": False,
        },
    }

    semester_start_date = date(year=2026, month=9, day=7)
    semester_end_date = date(year=2027, month=1, day=30)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mooc_base_path = f"{course_path}/mooc"

    mime_types = mt.DEFAULT_MIME_TYPES

    ################################################################

    mediaspace_playlist_or_channel_url = (
        "https://mediaspace.epfl.ch/channel/ME-326+Automatique+et+commande+num%C3%A9rique/55706?"
    )

    mediaspace_language = course_info["course_language"]

    # Only keep recordings of this course edition
    mediaspace_created_after = semester_start_date

    mediaspace_base_path = f"{course_path}/mediaspace"

    moodle_course_id = 16347

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
            MOOCExtractor(
                mooc_base_path=self.mooc_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=(self.mime_types + [mt.MP4, mt.JSON]),
                language=self.course_info["course_language"],
            ),
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=self.mime_types,
            ),
            MediaspaceExtractor(
                playlist_or_channel_url=self.mediaspace_playlist_or_channel_url,
                mediaspace_base_path=self.mediaspace_base_path,
                tag_metadata=self.tag_metadata,
                language=self.mediaspace_language,
                created_after=self.mediaspace_created_after,
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
            VideoToFramesTransformer(
                cache=self.course_code,
                language=self.course_info["course_language"],
            ),
            ImageToMarkdownTransformer(cache=self.course_code),
            MergeSlideTranscriptTransformer(cache=self.course_code),
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

    course = BaseCourse.from_code("ME326")
    course.run()
