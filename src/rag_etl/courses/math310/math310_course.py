from datetime import date

import logging

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MediaspaceExtractor, MoodleExtractor
from rag_etl.transformers import (
    BaseTransformer,
    ExtractZipTransformer,
    JupyterToMarkdownTransformer,
    PDFToMarkdownTransformer,
    SplitExercisesTransformer,
    VideoToFramesTransformer,
    ImageToMarkdownTransformer,
    MergeSlideTranscriptTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class MATH310Course(BaseCourse):
    """
    Course-specific pipeline for MATH310.
    """

    course_info = {
        "course_title": "Algebra",
        "course_id": "MATH310",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=15441",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/algebra-MATH-310",
        "course_language": "fr",
    }

    # [THEORY] <- lecture notes and other PDF materials
    # [THEORY_SLIDES] <- weekly slides for my lectures
    # [SERIE_x] <- a weekly series of exercises
    # [SERIE_x_SOLUTION] <- the solutions to these weekly series
    # [EXAM_xxxx] <- past exams from the last 3 years
    # [EXAM_xxxx_SOLUTION] <- the solutions to these 3 past exams
    # [HOMEWORK_x] written assignment to submit (but not graded)
    # [HOMEWORK_x_SOLUTION] solution to the assignment
    # [MIDTERM_EXAM_xxxx]  test blanc
    # [MIDTERM_EXAM_xxxx_SOLUTION] solution to the test blanc

    # +entrainement,supplementaire

    tag_metadata = {
        "THEORY": {
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
        "HOMEWORK": {
            "type": "practice",
            "subtype": "homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "HOMEWORK_SOLUTION": {
            "type": "practice",
            "subtype": "homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "SERIE_ENTRAINEMENT": {
            "type": "practice",
            "subtype": "serie_entrainement",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "SERIE_ENTRAINEMENT_SOLUTION": {
            "type": "practice",
            "subtype": "serie_entrainement",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "SERIE_SUPPLEMENTAIRE": {
            "type": "practice",
            "subtype": "serie_supplementaire",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "SERIE_SUPPLEMENTAIRE_SOLUTION": {
            "type": "practice",
            "subtype": "serie_supplementaire",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
            "is_solution": True,
        },
        "EXAM": {
            "type": "exam",
            "subtype": "exam",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "EXAM_SOLUTION": {
            "type": "exam",
            "subtype": "exam",
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

    semester_start_date = date(year=2026, month=9, day=16)
    semester_end_date = date(year=2027, month=2, day=10)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    ################################################################

    mediaspace_playlist_or_channel_url = "https://mediaspace.epfl.ch/channel/MATH-310%2BAlgebra/30044"

    mediaspace_language = course_info["course_language"]

    # Only keep recordings of this course edition
    mediaspace_created_after = semester_start_date
    # mediaspace_created_after = date(year=2025, month=9, day=1)

    mediaspace_base_path = f"{course_path}/mediaspace"

    moodle_course_id = 15441

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
        """Moodle course materials, plus the subtitles of the Mediaspace lecture recordings."""
        return [
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=mt.DEFAULT_MIME_TYPES,
            ),
            MediaspaceExtractor(
                playlist_or_channel_url=self.mediaspace_playlist_or_channel_url,
                mediaspace_base_path=self.mediaspace_base_path,
                tag_metadata=self.tag_metadata,
                language=self.mediaspace_language,
                created_after=self.mediaspace_created_after,
            ),
        ]

    @property
    def transformers(self) -> list[BaseTransformer]:
        """Single transformer that converts PDFs into Markdown text."""
        return [
            ExtractZipTransformer(cache=self.course_code),
            JupyterToMarkdownTransformer(cache=self.course_code),
            PDFToMarkdownTransformer(type_subtypes=self.pdf_to_markdown_type_subtypes, cache=self.course_code),
            SplitExercisesTransformer(type_subtypes=self.split_exercises_type_subtypes, cache=self.course_code),
            VideoToFramesTransformer(
                cache=self.course_code,
                language=self.mediaspace_language,
            ),
            ImageToMarkdownTransformer(cache=self.course_code),
            MergeSlideTranscriptTransformer(cache=self.course_code),
        ]

    @property
    def loaders(self) -> list[BaseLoader]:
        """No loaders defined for this course."""
        return [ContentMetadataLoader(course_path=self.course_path, course_info=self.course_info)]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    course = BaseCourse.from_code("MATH310")
    course.run()
