import re
from datetime import UTC, date, datetime


def caption_matches_language(caption_info: dict, language: str | None) -> bool:
    """
    True when a caption dict matches the wanted language.

    The value is matched case-insensitively against the caption's
    languageCode ("fr"), language ("French") and label ("French") —
    Mediaspace entries are not consistent about which of the three is
    meaningful.

    A falsy language means "no language filter", so everything matches.
    """

    if not language:
        return True

    wanted = str(language).strip().lower()

    for field in ("language_code", "language", "label"):
        value = caption_info.get(field)
        if value and str(value).strip().lower() == wanted:
            return True

    return False


def to_timestamp(value: str | date | datetime | float) -> int:
    """
    Coerce a date-ish value to a UTC unix timestamp, as used by Kaltura's
    createdAt.

    Accepts a datetime, a date, a unix timestamp, or a string such as
    "2022-01-31" or "2022-01-31T14:00:00". Naive datetimes and plain dates
    are read as UTC so that a given input always selects the same entries.
    """

    if isinstance(value, bool):
        raise TypeError(f"Cannot interpret {value!r} as a date")

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        value = datetime.fromisoformat(value.strip())

    # datetime is a subclass of date, so it has to be tested first.
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp())

    if isinstance(value, date):
        return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp())

    raise TypeError(f"Cannot interpret {value!r} as a date")


def entry_created_after(entry, created_after: str | date | datetime | float | None) -> bool:
    """
    True when an entry was created strictly after created_after.

    Kaltura's startDate is unset on Mediaspace entries and updatedAt tracks
    re-processing rather than the recording, so createdAt is the field that
    stands in for the video's date. Entries with no createdAt are kept.
    """

    if not created_after:
        return True

    created_at = getattr(entry, "createdAt", None)
    if not created_at:
        return True

    return int(created_at) > to_timestamp(created_after)


def safe_filename(text: str, max_length: int = 80) -> str:
    """Collapse arbitrary text into something usable as a file name."""

    cleaned = re.sub(r"[^\w\-]+", "_", text or "", flags=re.UNICODE).strip("_")
    return cleaned[:max_length] or "untitled"
