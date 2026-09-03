from datetime import date

import logging
from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MediaspaceExtractor, MOOCExtractor, MoodleExtractor
from rag_etl.transformers import (
    BaseTransformer,
    PDFToMarkdownTransformer,
    ExtractZipTransformer,
    SplitExercisesTransformer,
    ImageToMarkdownTransformer,
    MergeSlideTranscriptTransformer,
    VideoToFramesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class CS119dCourse(BaseCourse):
    """
    Course-specific pipeline for CS-119d
    """

    course_info = {
        "course_title": "Information, Calcul, Communication",
        "course_id": "CS119d",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=14023",
        "coursebook_link": "https://edu.epfl.ch/coursebook/fr/information-calcul-communication-CS-119-D",
        "course_language": "fr",
    }

    # [SERIE_x]
    # [SERIE_x_SOLUTION]
    # [MIDTERM_EXAM_xxxx]
    # [MIDTERM_EXAM_xxxx_SOLUTION]
    # [EXAM_xxxx]
    # [EXAM_xxxx_SOLUTION]
    # [CASE_STUDY_x_SOLUTION]

    tag_metadata = {
        "THEORY": {  # just in case
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": False,
            "pdf_to_markdown": True,
            "split_exercises": False,
        },
        "THEORY_SLIDES": {  # just in case
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
        "CASE_STUDY": {  # just in case
            "type": "practice",
            "subtype": "case_study",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "CASE_STUDY_SOLUTION": {
            "type": "practice",
            "subtype": "case_study",
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
        ############## MOOC ##############
        "MOOC_THEORY": {  # just in case
            "type": "theory",
            "subtype": "mooc_theory",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
        },
        "MOOC_TUTORIEL": {
            "type": "practice",
            "subtype": "mooc_tutoriel",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_EXERCICE": {
            "type": "practice",
            "subtype": "mooc_exercice",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_EXERCICE_SOLUTION": {
            "type": "practice",
            "subtype": "mooc_exercice",
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
        "MOOC_EXERCICE_FACULTATIF": {
            "type": "practice",
            "subtype": "mooc_exercice_facultatif",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_EXERCICE_FACULTATIF_SOLUTION": {
            "type": "practice",
            "subtype": "mooc_exercice_facultatif",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
            "is_solution": True,
        },
        "MOOC_DEVOIR": {
            "type": "practice",
            "subtype": "mooc_devoir",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
            "is_video": False,
            "is_gemini_processed_video": False,
            "processing_method": None,
            "model": None,
        },
        "MOOC_QUIZ": {
            "type": "practice",
            "subtype": "mooc_quiz",
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

    semester_start_date = date(year=2026, month=2, day=16)
    semester_end_date = date(year=2026, month=6, day=22)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mooc_base_path = f"{course_path}/mooc"

    mime_types = mt.DEFAULT_MIME_TYPES

    mediaspace_language = course_info["course_language"]

    mediaspace_playlist_or_channel_url = "https://mediaspace.epfl.ch/channel/CS-119%2528d%2529%2BInformation_%2Bcalcul_%2Bcommunication%2B%2528SMA%2B%2526%2BSPH%2529/30888"

    # They want all videos in the channel
    mediaspace_created_after = date(year=2020, month=8, day=1)

    mediaspace_base_path = f"{course_path}/mediaspace"

    moodle_course_id = 14023

    moodle_base_path = f"{course_path}/moodle"

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
        """Single MOOC extractor."""
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
        """Documents into Markdown, then videos into timestamped slides."""
        return [
            ExtractZipTransformer(cache=self.course_code),
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

    course = BaseCourse.from_code("CS119d")
    course.run()
