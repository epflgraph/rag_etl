import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOGUE_TYPES = ("theory", "practice", "exam")


@dataclass(frozen=True)
class CatalogueEntry:
    type: str
    subtype: str | None
    number: str | None
    sub_number: str | None
    title: str | None
    from_: str | None
    until: str | None
    week: int | None


def build_catalogue(metadata_dir: Path, sources: list[str] | None = None) -> list[CatalogueEntry]:
    """
    Build a closed-set catalogue of theory/practice/exam documents from a previous pipeline run.

    Empty when there is no previous run yet (or it had no such documents).
    """

    json_paths = find_metadata_json_paths(metadata_dir, sources)

    grouped: dict[tuple[str, str | None, str | None, str | None], CatalogueEntry] = {}
    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            continue

        documents = data.get("documents", [])
        for document in documents:
            merge_document_into_catalogue(document, grouped)

    add_synthesised_parents(grouped)

    catalogue = list(grouped.values())
    catalogue.sort(key=catalogue_sort_key)
    return catalogue


def find_metadata_json_paths(metadata_dir: Path, sources: list[str] | None) -> list[Path]:
    """List the metadata JSON files to read, excluding ed_discussion.json."""

    if not metadata_dir.exists():
        return []

    if sources is not None:
        json_paths = []
        for source in sources:
            source_path = metadata_dir / f"{source}.json"
            if source_path.exists():
                json_paths.append(source_path)
        return json_paths

    json_paths = []
    for json_path in sorted(metadata_dir.glob("*.json")):
        if json_path.name != "ed_discussion.json":
            json_paths.append(json_path)
    return json_paths


def merge_document_into_catalogue(
    document: dict,
    grouped: dict[tuple[str, str | None, str | None, str | None], CatalogueEntry],
) -> None:
    """Add one metadata document to the catalogue, deduping on (type, subtype, number, sub_number)."""

    doc_type = document.get("type")
    if doc_type not in CATALOGUE_TYPES:
        return

    key = (doc_type, document.get("subtype"), document.get("number"), document.get("sub_number"))
    title = document.get("title")
    from_ = document.get("from")
    until = document.get("until")
    week = document.get("week")

    existing = grouped.get(key)
    if existing is None:
        grouped[key] = CatalogueEntry(
            type=doc_type,
            subtype=document.get("subtype"),
            number=document.get("number"),
            sub_number=document.get("sub_number"),
            title=title,
            from_=from_,
            until=until,
            week=week,
        )
        return

    if title is not None and (existing.title is None or len(title) < len(existing.title)):
        grouped[key] = CatalogueEntry(
            type=existing.type,
            subtype=existing.subtype,
            number=existing.number,
            sub_number=existing.sub_number,
            title=title,
            from_=existing.from_,
            until=existing.until,
            week=existing.week,
        )


def add_synthesised_parents(
    grouped: dict[tuple[str, str | None, str | None, str | None], CatalogueEntry],
) -> None:
    """Synthesise a document-level entry for groups that only have sub-numbered rows."""

    children_by_parent: dict[tuple[str, str | None, str | None], list[CatalogueEntry]] = {}
    for entry in grouped.values():
        if entry.number is None or entry.sub_number is None:
            continue
        parent_key = (entry.type, entry.subtype, entry.number)
        children_by_parent.setdefault(parent_key, []).append(entry)

    for parent_key, children in children_by_parent.items():
        full_key = (parent_key[0], parent_key[1], parent_key[2], None)
        if full_key in grouped:
            continue
        grouped[full_key] = synthesise_parent_entry(parent_key, children)


def synthesise_parent_entry(
    parent_key: tuple[str, str | None, str | None],
    children: list[CatalogueEntry],
) -> CatalogueEntry:
    """Build a document-level entry that summarises its sub-numbered children."""

    doc_type, subtype, number = parent_key

    from_values = [child.from_ for child in children if child.from_ is not None]
    until_values = [child.until for child in children if child.until is not None]
    from_ = min(from_values) if from_values else None
    until = max(until_values) if until_values else None
    week = children[0].week

    shortest_title = None
    for child in children:
        if child.title is not None and (shortest_title is None or len(child.title) < len(shortest_title)):
            shortest_title = child.title

    title = strip_last_segment(shortest_title)

    return CatalogueEntry(
        type=doc_type,
        subtype=subtype,
        number=number,
        sub_number=None,
        title=title,
        from_=from_,
        until=until,
        week=week,
    )


def strip_last_segment(title: str | None) -> str | None:
    """Drop a trailing ' > segment' (e.g. an exercise reference) from a document title."""

    if title is None or " > " not in title:
        return title
    return title.rsplit(" > ", 1)[0]


def catalogue_sort_key(entry: CatalogueEntry) -> tuple:
    return (
        entry.type,
        entry.subtype or "",
        natural_sort_key(entry.number),
        natural_sort_key(entry.sub_number),
    )


def natural_sort_key(value: str | None) -> tuple:
    """Sort numeric-looking strings numerically, everything else after, alphabetically."""

    if value is None:
        return (0, "")
    if re.fullmatch(r"-?\d+", value):
        return (1, int(value))
    return (2, value)


def render_catalogue(catalogue: list[CatalogueEntry]) -> str:
    """Render the catalogue as one line per entry, for inclusion in the classification prompt."""

    lines = []
    for i, entry in enumerate(catalogue):
        lines.append(format_entry(i, entry))
    return "\n".join(lines)


def format_entry(catalogue_id: int, entry: CatalogueEntry) -> str:
    """Render a single catalogue entry as it appears in the classification prompt."""

    return f"[{catalogue_id}] {render_catalogue_entry(entry)}"


def render_catalogue_entry(entry: CatalogueEntry) -> str:
    parts = [entry.type]
    if entry.subtype:
        parts.append(f"/ {entry.subtype}")

    number_part = ""
    if entry.number is not None:
        number_part = entry.number
        if entry.sub_number is not None:
            number_part += f" / {entry.sub_number}"
    if number_part:
        parts.append(number_part)

    rendered = " ".join(parts)
    if entry.title:
        rendered += f" — {entry.title}"
    return rendered


def find_entry(
    catalogue: list[CatalogueEntry],
    type: str | None,
    subtype: str | None,
    number: str | None,
    sub_number: str | None,
) -> CatalogueEntry | None:
    """Look up a catalogue entry by its exact (type, subtype, number, sub_number) tuple."""

    for entry in catalogue:
        if (
            entry.type == type
            and entry.subtype == subtype
            and entry.number == number
            and entry.sub_number == sub_number
        ):
            return entry
    return None
