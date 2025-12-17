from datetime import date, timedelta

import re

from typing import Tuple, Optional

from rag_etl.resources import BaseResource, MoodleResource
import rag_etl.utils.mime_types as mt


def infer_date(resource: BaseResource, start_date: date, end_date: date) -> str:
    assert start_date <= end_date, f"Invalid dates: start_date ({start_date}) should be before end_date ({end_date})"

    if isinstance(resource, MoodleResource):
        text = resource.section_title
    else:
        text = resource.title.lower()

    # Dates in format "15 April - 16 April"
    date_pattern = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]+)\s*-\s*(\d{1,2})\s+([A-Za-z]+)\s*$")
    match = date_pattern.match(text)
    if match:
        start_day, start_month, _, _ = match.groups()
        day = int(start_day)
        month = start_month
    else:
        day = None
        month = None

    # Dates in format "3 - 4 Octob"
    if day is None or month is None:
        date_pattern = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s+([A-Za-z]+)")
        match = date_pattern.match(text)
        if match:
            start_day, _, month = match.groups()
            day = int(start_day)

    # No day and month were found, fall back to start of semester
    if not match:
        return str(start_date)

    # Normalise month to three letters, then to int
    month = month.strip().lower()[:3]  # normalise (Oct → oct, October → oct, octover → oct)
    month_map = {
        "jan": 1, "feb": 2, "fev": 2, "fév": 2, "mar": 3,
        "apr": 4, "avr": 4, "may": 5, "mai": 5, "jun": 6, "jui": 6,
        "jul": 7, "aug": 8, "aou": 8, "aoû": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
    }
    month = month_map[month]

    # Infer year from start_date and end_date
    for year in range(start_date.year, end_date.year + 1):
        try:
            inferred_date = date(year=year, month=month, day=day)
        except ValueError:
            # Invalid date (e.g. Feb 30, Feb 29 in non-leap year)
            continue

        if start_date <= inferred_date <= end_date:
            return str(inferred_date)

    # No year was found, fall back to start of semester
    return str(start_date)


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


def get_processing_method_model(resource: BaseResource) -> Tuple[Optional[str], Optional[str]]:
    if resource.mime_type != mt.PDF:
        return None, None

    if (resource.type, resource.subtype) == ('theory', 'polycopie'):
        return 'google', None

    return 'gemini', 'gemini-2.5-pro'


def get_one_chunk_per_page(resource: BaseResource) -> bool:
    if (resource.type, resource.subtype) == ('theory', 'lecture_slides'):
        return True

    return False


def get_one_chunk_per_doc(resource: BaseResource) -> bool:
    if resource.type == 'practice':
        return True

    return False


def get_number(resource: BaseResource) -> Optional[str]:
    if resource.type == 'theory':
        return None

    if resource.type == 'exam':
        return str(resource.year)

    if resource.type == 'practice':
        if resource.subtype == 'exercise':
            return None

        if resource.subtype == 'series':
            return str(resource.week)

        if resource.subtype == 'slt':
            return None

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
