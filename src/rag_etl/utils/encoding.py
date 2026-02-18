from pathlib import Path
import unicodedata


def resolve_path(p: Path) -> Path:
    """Find the actual file on disk by fuzzy-matching the filename."""
    if p.exists():
        return p

    parent = p.parent
    if not parent.exists():
        return p

    target = normalize_for_compare(p.name)
    for candidate in parent.iterdir():
        if normalize_for_compare(candidate.name) == target:
            return candidate

    return p


def normalize_for_compare(name: str) -> str:
    """Normalize a filename for comparison: collapse spaces/underscores/punctuation, NFC unicode."""
    name = unicodedata.normalize("NFC", name)
    # return name
    # Remove characters that may be inconsistently encoded in hrefs
    name = name.replace(" ", "").replace("_", "")
    # straight + curly apostrophes
    name = name.replace("'", "").replace("'", "").replace("'", "")
    name = name.replace("-", "").replace("–", "").replace("—", "")  # hyphens + dashes
    return name.casefold()
