from datetime import date

import logging

from rag_etl.courses import BaseCourse
from rag_etl.extractors import BaseExtractor, MoodleExtractor, MediaspaceExtractor
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


class MATH111aCourse(BaseCourse):
    """
    Course-specific pipeline for MATH111a
    """

    course_info = {
        "course_title": "Linear Algebra",
        "course_id": "MATH111a",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=18585",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/linear-algebra-MATH-111-A",
        "course_language": "fr",
    }

    # [EXERCISE_x]
    #  [EXERCISE_x_SOLUTION]
    #  [SERIE_x] <- a weekly series of exercises
    #  [SERIE_x_SOLUTION] <- the solutions to these weekly series
    #  [THEORY] <- lecture notes in PDF that I'll update every week

    # He didn't add the previous year exams but they are in the Moodel page, inside a folder.
    # I've added them just in case

    tag_metadata = {
        "THEORY": {
            "type": "theory",
            "subtype": "theory",
            "one_chunk_per_page": False,
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
        "EXERCISE": {
            "type": "practice",
            "subtype": "exercise",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "EXERCISE_SOLUTION": {
            "type": "practice",
            "subtype": "exercise",
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

    ################################################################

    mediaspace_language = course_info["course_language"]

    mediaspace_playlist_or_channel_url = (
        "https://mediaspace.epfl.ch/channel/MATH-111%28a%29+Alg%C3%A8bre+lin%C3%A9aire/90346"
    )

    # Only keep recordings of this course edition
    mediaspace_created_after = semester_start_date

    mediaspace_base_path = f"{course_path}/mediaspace"

    moodle_course_id = 18585

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
            MoodleExtractor(
                moodle_course_id=self.moodle_course_id,
                moodle_base_path=self.moodle_base_path,
                tag_metadata=self.tag_metadata,
                mime_types=(mt.DEFAULT_MIME_TYPES),
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

    course = BaseCourse.from_code("MATH111a")
    course.run()
