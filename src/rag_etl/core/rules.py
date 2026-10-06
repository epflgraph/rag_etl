"""Manual exceptions: glob matching over resource attributes, applied on top of the judge's labels.

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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_etl.core.resource import Resource

logger = logging.getLogger(__name__)

_PREDICATE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))')


@dataclass(frozen=True)
class Rule:
    """One exception: a where-predicate plus the labels it corrects."""

    where: str                    # flat ANDed globs, e.g. 'title="*[[]HOMEWORK*]*" AND mime_type="application/pdf"'
    type: str | None = None       # theory / practice / ...
    subtype: str | None = None    # lecture_slides / homework / ...
    cut: str | None = None        # whole_document | per_child | per_exercise | text
    number: int | None = None
    week: int | None = None
    is_solution: bool | None = None

    @property
    def specificity(self) -> int:
        """How many predicates the where carries; more is more specific."""
        return len(parse_where(self.where))

    def assignments(self) -> dict[str, Any]:
        """The labels this rule sets; where and untouched fields are left out."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "where" and getattr(self, f.name) is not None
        }

    def matches(self, resource: Resource) -> bool:
        """Whether every glob in the where holds for this resource."""
        attrs = _attributes(resource)
        for name, glob in parse_where(self.where):
            if name not in attrs:
                logger.warning(f"where-field '{name}' is not a resource attribute; rule never matches: {self.where}")
                return False
            if not fnmatch.fnmatchcase(attrs[name], glob):
                return False
        return True


def parse_where(where: str) -> list[tuple[str, str]]:
    """Parse 'field=glob AND field=glob' into (field, glob) pairs."""
    return [
        (match.group(1), match.group(2) if match.group(2) is not None else match.group(3))
        for match in _PREDICATE.finditer(where)
    ]


def _attributes(resource: Resource) -> dict[str, str]:
    """Every resource attribute a where-glob can match on."""
    path = str(resource.path) if resource.path else ""
    return {
        "path": path,
        "filename": Path(path).name if path else "",
        "mime_type": resource.mime_type or "",
        "title": resource.title or "",
        "type": resource.type or "",
        "subtype": resource.subtype or "",
        "cut": resource.cut or "",
        "number": "" if resource.number is None else str(resource.number),
        "sub_number": "" if resource.sub_number is None else str(resource.sub_number),
        "week": "" if resource.week is None else str(resource.week),
        "is_solution": "true" if resource.is_solution else "false",
        "is_leaf": "true" if resource.is_leaf else "false",
        "text": resource.text or "",
    }


def apply(resource: Resource, exceptions: Sequence[Rule]) -> None:
    """Correct the labels of one resource with every exception that matches it.

    Most-specific wins; equal specificity resolves by declaration order,
    first declared wins, loudly. No match leaves the resource untouched.
    """
    ordered = sorted(
        ((index, rule) for index, rule in enumerate(exceptions) if rule.matches(resource)),
        key=lambda pair: (pair[1].specificity, pair[0]),
    )

    depth: dict[str, int] = {}
    for index, rule in ordered:
        for name, value in rule.assignments().items():
            if name not in depth or rule.specificity > depth[name]:
                setattr(resource, name, value)
                depth[name] = rule.specificity
            elif getattr(resource, name) != value:
                logger.warning(
                    f"Ambiguous {name} for {resource!r}: {getattr(resource, name)!r} already set at the same "
                    f"specificity, {value!r} from exceptions[{index}] ({rule.where}); keeping the first"
                )
