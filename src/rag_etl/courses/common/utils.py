from datetime import date, timedelta

import re

from typing import Tuple, Optional

from rag_etl.resources import BaseResource, MoodleResource
import rag_etl.utils.mime_types as mt


def infer_date(resource: BaseResource, start_date: date, end_date: date) -> str:
    if isinstance(resource, MoodleResource):
        text = resource.section_title
    else:
        text = resource.title.lower()

    # Regex to capture "start-day - end-day month" (e.g. "3 - 4 Octob")
    date_pattern = re.compile(r"^\s*(\d{1,2})\s*-\s*\d{1,2}\s+([A-Za-z]+)")

    match = date_pattern.match(text)
    if not match:
        return str(start_date)

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

    # Clamp inferred date to be within given bounds
    inferred_date = max(start_date, inferred_date)
    inferred_date = min(end_date, inferred_date)

    return str(inferred_date)


def infer_week(resource: BaseResource, weeks: dict) -> Optional[int]:
    return weeks.get(resource.date, None)


def infer_year(resource: BaseResource) -> Optional[str]:
    text = resource.title.lower()

    years = re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text)

    if years:
        return '-'.join(years)

    return None


def get_type_subtype(resource: BaseResource) -> Tuple[str, str]:
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


def get_is_solution(resource: BaseResource) -> bool:
    if isinstance(resource, MoodleResource):
        text = f"{resource.section_title.lower()}\n{resource.title.lower()}"
    else:
        text = resource.title.lower()

    if 'solution' in text:
        return True

    return False


def get_processing_method(resource: BaseResource) -> Optional[str]:
    if resource.mime_type != mt.PDF:
        return None

    if (resource.type, resource.subtype) == ('theory', 'polycopie'):
        return 'google'

    return 'gemini'


def get_number(resource: BaseResource) -> Optional[str]:
    if resource.week:
        return str(resource.week)

    if resource.year:
        return str(resource.year)

    return None


def get_shifted_date(resource: BaseResource, weeks: dict) -> Optional[str]:
    # If no date, there is nothing we can do
    if not resource.date:
        return resource.date

    # Assuming dates and weeks are sorted increasingly
    valid_dates = [d for (d, w) in weeks.items() if w]

    shifted_date = resource.date
    if resource.date in valid_dates:
        idx = valid_dates.index(resource.date)

        try:
            shifted_date = valid_dates[idx + 1]
        except IndexError:
            shifted_date = str(date.fromisoformat(resource.date) + timedelta(weeks=1))

    return shifted_date


def get_from(resource: BaseResource) -> Optional[str]:
    if resource.date and (resource.type, resource.subtype) == ('practice', 'homework'):
        return f"{resource.date}T00:00:00.000000"
    else:
        return None
