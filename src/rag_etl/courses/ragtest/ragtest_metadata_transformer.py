from datetime import date

from typing import Sequence

from rag_etl.resources import BaseResource, MoodleResource
from rag_etl.transformers import BaseTransformer

from rag_etl.courses.common.utils import (
    infer_date,
    infer_week,
    infer_year,
    get_type_subtype,
    get_is_solution,
    get_processing_method_model,
    get_one_chunk_per_page,
    get_one_chunk_per_doc,
    get_number,
    get_shifted_date,
    get_from,
)

from rag_etl.config import CONFIG


class RAGTESTMetadataTransformer(BaseTransformer):

    course_info = {
        "course_title": "RAG Test Course",
        "course_id": "RAGTEST",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=15403",
        "coursebook_link": None
    }

    semester_start_date = date(year=2026, month=2, day=14)
    semester_end_date = date(year=2026, month=6, day=22)

    weeks = {
        '2026-03-04': 1,
        '2026-03-11': 2,
        '2026-03-18': 3,
        '2026-03-25': 4,
        '2026-04-01': 5,
        '2026-04-08': 6,
        '2026-04-15': 7,
        '2026-04-22': 8,
        '2026-04-29': 9,
        '2026-05-06': 10,
        '2026-05-13': 11,
        '2026-05-20': 12,
        '2026-05-27': 13,
        '2026-06-03': 14,
        '2026-06-10': 15,
    }

    output_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"

    ################################################################

    moodle_course_id = 15403

    moodle_base_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}/moodle"

    moodle_tag_types_subtypes = {
        'LECTURE_SLIDES': ('theory', 'lecture_slides'),
        'LECTURE_NOTES': ('theory', 'lecture_notes'),
        'POLYCOPIE': ('theory', 'polycopie'),

        'EXERCISE': ('practice', 'exercise'),
        'EXERCISE_SOLUTION': ('practice', 'exercise'),
        'SERIES': ('practice', 'series'),
        'SERIES_SOLUTION': ('practice', 'series'),
        'SLT': ('practice', 'slt'),
        'SLT_SOLUTION': ('practice', 'slt'),

        'EXAM': ('exam', 'previous_year_exam'),
        'EXAM_SOLUTION': ('exam', 'previous_year_exam'),
        'MOCK_EXAM': ('exam', 'mock_exam'),
        'MOCK_EXAM_SOLUTION': ('exam', 'mock_exam'),
    }

    ################################################################

    pdf_to_markdown_type_subtypes = [
        ('practice', 'exercise'),
        ('practice', 'series'),
        ('practice', 'slt'),

        ('exam', 'previous_year_exam'),
        ('exam', 'mock_exam'),
    ]

    split_exercises_type_subtypes = [
        ('practice', 'exercise'),
        ('practice', 'series'),
        ('practice', 'slt'),

        ('exam', 'previous_year_exam'),
        ('exam', 'mock_exam'),
    ]

    ################################################################

    def get_type_subtype(self, resource: BaseResource):
        # If Moodle resource, get from dict above according to the tags
        if isinstance(resource, MoodleResource):
            if resource.tag in self.moodle_tag_types_subtypes:
                return self.moodle_tag_types_subtypes[resource.tag]

        # Default to text matching otherwise
        return get_type_subtype(resource)

    def get_is_solution(self, resource: BaseResource):
        # If Moodle resource, get from dict above according to the tags
        if isinstance(resource, MoodleResource):
            if resource.tag and 'SOLUTION' in resource.tag:
                return True

        # Default to text matching otherwise
        return get_is_solution(resource)

    def transform(self, resources: Sequence[BaseResource]) -> Sequence[BaseResource]:
        for resource in resources:
            # Infer time-related fields, like date, week and year
            resource.date = infer_date(resource, start_date=self.semester_start_date, end_date=self.semester_end_date)
            resource.week = infer_week(resource, weeks=self.weeks)
            resource.year = infer_year(resource)

            # Infer type and subtype
            resource.type, resource.subtype = self.get_type_subtype(resource)

            # Infer whether it is a solution
            resource.is_solution = self.get_is_solution(resource)

            # Infer processing method
            resource.processing_method, resource.model = get_processing_method_model(resource)

            # Infer one chunk flags
            resource.one_chunk_per_page = get_one_chunk_per_page(resource)
            resource.one_chunk_per_doc = get_one_chunk_per_doc(resource)

            # Infer number
            resource.number = get_number(resource)

            # If it is a solution resource, we need to add a week to the date
            if resource.is_solution:
                resource.date = get_shifted_date(resource, weeks=self.weeks)

            # Create from field with the datetime
            resource.from_ = get_from(resource)

        return resources
