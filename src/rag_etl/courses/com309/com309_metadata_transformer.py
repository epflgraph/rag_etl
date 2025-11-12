from datetime import date, timedelta

import re

from typing import Sequence, Tuple, Optional

from rag_etl.resources import BaseResource, MoodleResource
from rag_etl.transformers import BaseTransformer


class COM309MetadataTransformer(BaseTransformer):

    weeks = {
        '2025-09-10': 1,
        '2025-09-17': 2,
        '2025-09-24': 3,
        '2025-10-01': 4,
        '2025-10-08': 5,
        '2025-10-15': 6,
        '2025-10-22': None,
        '2025-10-29': 7,
        '2025-11-05': 8,
        '2025-11-12': 9,
        '2025-11-19': 10,
        '2025-11-26': 11,
        '2025-12-03': 12,
        '2025-12-10': 13,
        '2025-12-17': 14,
    }

    semester_start_date = date(year=2025, month=9, day=8)
    semester_end_date = date(year=2026, month=2, day=1)

    def _infer_date(self, resource: BaseResource) -> str:
        if isinstance(resource, MoodleResource):
            text = resource.section_title
        else:
            text = resource.title.lower()

        # Regex to capture "start-day - end-day month" (e.g. "3 - 4 Octob")
        date_pattern = re.compile(r"^\s*(\d{1,2})\s*-\s*\d{1,2}\s+([A-Za-z]+)")

        match = date_pattern.match(text)
        if not match:
            return str(self.semester_start_date)

        # Extract day and month strings
        day, month_str = match.groups()

        # Normalise day to int
        day = int(day)

        # Normalise month to three letters, then to int
        month_str = month_str.strip().lower()[:3]  # normalize (Oct → oct, October → oct, octover → oct)
        month_map = {
            "jan": 1, "feb": 2, "fev": 2, "fév": 2, "mar": 3,
            "apr": 4, "avr": 4, "may": 5, "mai": 5, "jun": 6, "jui": 6,
            "jul": 7, "aug": 8, "aou": 8, "aoû": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
        }
        month = month_map[month_str]

        # Infer year based on month and semester
        year = 2025 if month >= 6 else 2026

        # Build inferred date
        inferred_date = date(year=year, month=month, day=int(day))

        # Clamp inferred date to be in semester
        inferred_date = max(self.semester_start_date, inferred_date)
        inferred_date = min(self.semester_end_date, inferred_date)

        return str(inferred_date)

    def _infer_week(self, resource: BaseResource) -> Optional[int]:
        return self.weeks.get(resource.date, None)

    def _infer_year(self, resource: BaseResource) -> Optional[str]:
        text = resource.title.lower()

        years = re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text)

        if years:
            return '-'.join(years)

        return None

    def _get_type_subtype(self, resource: BaseResource) -> Tuple[str, str]:
        if isinstance(resource, MoodleResource):
            text = f"{resource.section_title.lower()}\n{resource.title.lower()}"
        else:
            text = resource.title.lower()

        # 'exam' match
        keyword = 'exam'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match:
            return 'exam', 'previous_year_exam'

        # 'exams' match
        keyword = 'exams'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match:
            return 'exam', 'previous_year_exam'

        # 'solution' match
        keyword = 'solution'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match and resource.week:
            return 'practice', 'homework'

        # 'solutions' match
        keyword = 'solutions'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match and resource.week:
            return 'practice', 'homework'

        # 'homework' match
        keyword = 'homework'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match:
            return 'practice', 'homework'

        # 'problem' match
        keyword = 'problem'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match and resource.week:
            return 'practice', 'homework'

        # 'problems' match
        keyword = 'problems'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match and resource.week:
            return 'practice', 'homework'

        # 'project' match
        keyword = 'project'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match:
            return 'practice', 'project'

        # 'projects' match
        keyword = 'projects'
        match = bool(re.search(rf'\b{keyword}\b', text, re.IGNORECASE))
        if match:
            return 'practice', 'project'

        # 'lecture notes'
        if 'lecture notes' in text:
            return 'theory', 'lecture_notes'

        return 'theory', 'lecture_slides'

    def _get_is_solution(self, resource: BaseResource) -> bool:
        if isinstance(resource, MoodleResource):
            text = f"{resource.section_title.lower()}\n{resource.title.lower()}"
        else:
            text = resource.title.lower()

        if 'solution' in text:
            return True

        return False

    def _get_processing_method(self, resource: BaseResource) -> Optional[str]:
        if resource.mime_type != 'application/pdf':
            return None

        if (resource.type, resource.subtype) == ('theory', 'lecture_notes'):
            return 'google'

        return 'gemini'

    def _get_number(self, resource: BaseResource) -> Optional[str]:
        if resource.week:
            return str(resource.week)

        if resource.year:
            return str(resource.year)

        return None

    def _get_shifted_date(self, resource: BaseResource) -> Optional[str]:
        # Assuming dates and weeks are sorted increasingly
        valid_dates = [d for (d, w) in self.weeks.items() if w]

        shifted_date = resource.date

        if resource.date in valid_dates:
            idx = valid_dates.index(resource.date)

            try:
                shifted_date = valid_dates[idx + 1]
            except IndexError:
                shifted_date = str(date.fromisoformat(resource.date) + timedelta(weeks=1))

        return shifted_date

    def _get_from(self, resource: BaseResource) -> Optional[str]:
        if resource.date and (resource.type, resource.subtype) == ('practice', 'homework'):
            return f"{resource.date}T00:00:00.000000"
        else:
            return None

    def transform(self, resources: Sequence[BaseResource]) -> Sequence[BaseResource]:
        for resource in resources:
            # Infer time-related fields, like date, week and year
            resource.date = self._infer_date(resource)
            resource.week = self._infer_week(resource)
            resource.year = self._infer_year(resource)

            # Infer type and subtype
            resource.type, resource.subtype = self._get_type_subtype(resource)

            # Infer whether it is a solution
            resource.is_solution = self._get_is_solution(resource)

            # Infer processing method
            resource.processing_method = self._get_processing_method(resource)

            # Infer number
            resource.number = self._get_number(resource)

            # If it is a solution resource, we need to add a week to the date
            if (resource.type, resource.subtype) == ('practice', 'homework') and resource.is_solution:
                resource.date = self._get_shifted_date(resource)

            # Create from field with the datetime
            resource.from_ = self._get_from(resource)

        return resources
