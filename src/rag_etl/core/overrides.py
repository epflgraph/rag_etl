"""Overrides: manual corrections, glob matching over resource attributes, applied on top of the judge.

A `where` is globs over resource attributes joined by AND — no nesting, no
logic beyond AND. Globs are fnmatch patterns; a literal `[` is written `[[]`.
Most-specific wins; equal specificity resolves by declaration order, first
declared wins, loudly.
"""

from __future__ import annotations

import fnmatch
import logging
import re

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_etl.core.resource import METADATA_FIELDS, Resource

logger = logging.getLogger(__name__)

_PREDICATE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))')


class Override:
    """One correction: a where-predicate plus the cut and metadata it sets.

    `Override('title="*[[]HOMEWORK*]*"', type="practice", cut="per_exercise")`
    — any metadata field can be set; a name that is not one fails here, when
    the spec is loaded, rather than silently never applying.
    """

    def __init__(self, where: str, cut: str | None = None, **metadata: Any):
        unknown = set(metadata) - set(METADATA_FIELDS)
        if unknown:
            raise ValueError(f"Not metadata fields: {sorted(unknown)} (expected some of {METADATA_FIELDS})")

        self.where = where
        self.predicates = parse_where(where)
        self.assignments = {
            name: value for name, value in {"cut": cut, **metadata}.items() if value is not None
        }

    def __repr__(self) -> str:
        assignments = ", ".join(f"{name}={value!r}" for name, value in self.assignments.items())
        return f"Override({self.where!r}, {assignments})"

    @property
    def specificity(self) -> int:
        """How many predicates the where carries; more is more specific."""
        return len(self.predicates)

    def matches(self, resource: Resource) -> bool:
        """Whether every glob in the where holds for this resource."""
        attributes = _attributes(resource)
        for name, glob in self.predicates:
            if name not in attributes:
                logger.warning(f"where-field '{name}' is not a resource attribute; override never matches: {self.where}")
                return False
            if not fnmatch.fnmatchcase(attributes[name], glob):
                return False
        return True


def parse_where(where: str) -> list[tuple[str, str]]:
    """Parse 'field=glob AND field=glob' into (field, glob) pairs."""
    return [
        (match.group(1), match.group(2) if match.group(2) is not None else match.group(3))
        for match in _PREDICATE.finditer(where)
    ]


def _as_text(value: Any) -> str:
    """An attribute as a where-glob reads it: None empty, booleans lowercase."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _attributes(resource: Resource) -> dict[str, str]:
    """Every resource attribute a where-glob can match on."""
    path = str(resource.path) if resource.path else ""
    return {
        "path": path,
        "filename": Path(path).name if path else "",
        "mime_type": _as_text(resource.mime_type),
        "title": resource.title,
        "cut": _as_text(resource.cut),
        "is_leaf": _as_text(resource.is_leaf),
        "text": _as_text(resource.text),
        **{name: _as_text(value) for name, value in resource.metadata.items()},
    }


def apply_overrides(resource: Resource, overrides: Sequence[Override]) -> None:
    """Correct one resource with every override that matches it.

    Most-specific wins; equal specificity resolves by declaration order,
    first declared wins, loudly. No match leaves the resource untouched.
    """
    ordered = sorted(
        ((index, override) for index, override in enumerate(overrides) if override.matches(resource)),
        key=lambda pair: (pair[1].specificity, pair[0]),
    )

    depth: dict[str, int] = {}
    for index, override in ordered:
        for name, value in override.assignments.items():
            if name not in depth or override.specificity > depth[name]:
                setattr(resource, name, value)
                depth[name] = override.specificity
            elif getattr(resource, name) != value:
                logger.warning(
                    f"Ambiguous {name} for {resource!r}: {getattr(resource, name)!r} already set at the same "
                    f"specificity, {value!r} from overrides[{index}] ({override.where}); keeping the first"
                )
