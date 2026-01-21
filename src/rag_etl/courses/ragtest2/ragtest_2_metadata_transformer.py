from datetime import date

import re

from typing import Sequence, Optional

from rag_etl.resources import BaseResource, MoodleResource
from rag_etl.transformers import BaseTransformer

from rag_etl.courses.common.utils import (
    infer_year,
    get_type_subtype,
    get_is_solution,
    get_processing_method_model,
    get_one_chunk_per_page,
    get_one_chunk_per_doc,
    get_from,
)

from rag_etl.config import CONFIG


class RAGTEST2MetadataTransformer(BaseTransformer):

    course_info = {
        "course_title": "RAG Test 2 Course",
        "course_id": "RAGTEST2",
        "academic_course": "2025-2026",
        "semester": 2,
        "admin_info_link": "https://moodle.epfl.ch/course/view.php?id=19162",
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

    reversed_weeks = {week: date for date, week in weeks.items()}

    output_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}"

    ################################################################

    moodle_course_id = 19162

    moodle_base_path = f"{CONFIG['BASE_PATH']}/{course_info['course_id']}/moodle"

    moodle_tag_types_subtypes = {
        'SLIDES': ('theory', 'lecture_slides'),
        'POLYCOPIE': ('theory', 'polycopie'),

        'SERIE': ('practice', 'exercise'),
        'SERIE_SOLUTION': ('practice', 'exercise'),

        'SERIE_ENTRAINEMENT': ('practice', 'exercise_training'),
        'SERIE_ENTRAINEMENT_SOLUTION': ('practice', 'exercise_training'),

        'QCM': ('practice', 'qcm'),
        'QCM_SOLUTION': ('practice', 'qcm'),

        'EXAM': ('exam', 'previous_year_exam'),
        'EXAM_SOLUTION': ('exam', 'previous_year_exam'),
    }

    ################################################################

    pdf_to_markdown_type_subtypes = [
        ('practice', 'exercise'),
        ('practice', 'exercise_training'),
        ('practice', 'qcm'),

        ('exam', 'previous_year_exam'),
    ]

    split_exercises_type_subtypes = [
        ('practice', 'exercise'),
        ('practice', 'exercise_training'),
        ('practice', 'qcm'),

        ('exam', 'previous_year_exam'),
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
            if resource.tag and 'solution' in resource.tag.lower():
                return True

        # Default to text matching otherwise
        return get_is_solution(resource)

    def infer_week(self, resource: BaseResource) -> Optional[int]:
        # If Moodle resource, extract week number from section_title
        if isinstance(resource, MoodleResource):
            match = re.search(r"\bSemaine\s+(\d+)\b", resource.section_title, flags=re.IGNORECASE)
            return int(match.group(1)) if match else None

        # Default to None otherwise
        return None

    def infer_date(self, resource: BaseResource) -> Optional[str]:
        # If Moodle resource, extract date from dict above
        if isinstance(resource, MoodleResource):
            return self.reversed_weeks.get(resource.week)

        # Default to None otherwise
        return None

    def get_number(self, resource: BaseResource) -> Optional[str]:
        if resource.type == 'exam':
            if resource.year:
                return str(resource.year)
            else:
                return str(self.semester_start_date.year)

        if (resource.type, resource.subtype) == ('practice', 'exercise'):
            return str(resource.week)

        if (resource.type, resource.subtype) == ('practice', 'qcm'):
            return str(resource.year)

        return None

    def transform(self, resources: Sequence[BaseResource]) -> Sequence[BaseResource]:
        for resource in resources:
            # Infer time-related fields, like date, week and year
            resource.week = self.infer_week(resource)
            resource.date = self.infer_date(resource)
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
            resource.number = self.get_number(resource)

            # Create from field with the datetime
            resource.from_ = get_from(resource)

        return resources
