from datetime import date

import logging
from rag_etl.courses import BaseCourse
from rag_etl.extractors import (
    BaseExtractor,
    MOOCExtractor,
    MoodleExtractor,
)
from rag_etl.transformers import (
    BaseTransformer,
    ImageToMarkdownTransformer,
    MergeSlideTranscriptTransformer,
    PDFToMarkdownTransformer,
    ExtractZipTransformer,
    SplitExercisesTransformer,
    VideoToFramesTransformer,
)

from rag_etl.loaders import BaseLoader, ContentMetadataLoader

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


class BIO695Course(BaseCourse):
    """
    Course-specific pipeline for BIO695
    """

    course_info = {
        "course_title": "Image Processing for Life Science",
        "course_id": "BIO695",
        "academic_course": "2026-2027",
        "semester": 1,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=18599",
        "coursebook_link": "https://edu.epfl.ch/coursebook/en/image-processing-for-life-science-BIO-695",
        "course_language": "en",
    }

    # [THEORY]
    # [THEORY_SLIDES]
    # [EXERCISE]
    # [EXERCISE_x_SOLUTION]
    # [HOMEWORK_x]
    # [PROJECT]

    tag_metadata = {
        "THEORY": {
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
        "HOMEWORK": {
            "type": "practice",
            "subtype": "homework",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": True,
        },
        "PROJECT": {
            "type": "practice",
            "subtype": "project",
            "one_chunk_per_page": False,
            "one_chunk_per_doc": True,
            "pdf_to_markdown": True,
            "split_exercises": False,
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
    }

    semester_start_date = date(year=2026, month=9, day=7)
    semester_end_date = date(year=2027, month=1, day=31)

    course_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"
    output_path = f"{course_path}/output"

    mooc_base_path = f"{course_path}/mooc"

    mime_types = mt.DEFAULT_MIME_TYPES

    # Videos only from the MOOC

    moodle_course_id = 18599

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

    course = BaseCourse.from_code("BIO695")
    course.run()
