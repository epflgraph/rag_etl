"""The manifest loader: one JSON manifest per run, written for humans.

The manifest records what a run produced without carrying content: the
header states run identity, the summary drills down per source, type,
subtype, number, subnumber and exercise parts — levels a group does not
split are folded into it, nulls stated, and per group it counts how
many chunks carry a url, week or release dates, stating those values
when the whole group shares one — and the tree mirrors the resource
trees with metadata and content-hash pointers only — text and vectors
are recovered from the cache by hash. Nothing reads a manifest; it is
inspected.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rag_etl.core import Resource, iter_chunks
from rag_etl.loaders.base import Loader, alias_name, index_name

logger = logging.getLogger(__name__)

# Levels of the summary drill-down, outermost first
SUMMARY_LEVELS = ("source", "type", "subtype", "number", "subnumber", "exercise_parts")

# Coverage fields a summary group reports: a with_X count of the chunks
# carrying one, plus X itself when every chunk in the group shares one value
COVERAGE_FIELDS = ("url", "week", "from", "until")


def _pipeline_version() -> str:
    try:
        return version("rag-etl")
    except PackageNotFoundError:
        return "unknown"


def _node(node: Resource, project_dir: Path) -> dict[str, Any]:
    """One tree node: metadata and a content-hash pointer, no text, no vector.

    The path is relative to the project folder, so it reads the same on
    every machine and shows no local layout.
    """
    return {
        "id": node.id,
        "title": node.title,
        "url": node.url,
        "path": str(node.path.relative_to(project_dir, walk_up=True)) if node.path is not None else None,
        "mime_type": node.mime_type,
        "cut": node.cut,
        **node.metadata,
        "chunk": node.is_chunk,
        "content_hash": node.content_hash,
        "children": [_node(child, project_dir) for child in node.children],
    }


def _summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """One summary node: shared levels, coverage counts, unanimous values, drill-down.

    Levels every row shares, null included, are stated on the node itself:
    a single subgroup would only repeat the node's counts. Each coverage
    field reports how many chunks in the group carry a non-null value, and
    the value itself only when they all share one. The drill-down groups
    rows by the first level that splits them — None sorts last, so
    manifests diff cleanly — and recurses; a node no level splits carries
    no counts key.
    """
    node: dict[str, Any] = {}
    while keys and len(values := {row[keys[0]] for row in rows}) == 1:
        node[keys[0]] = values.pop()
        keys = keys[1:]

    node["chunks"] = len(rows)
    for name in COVERAGE_FIELDS:
        node[f"with_{name}"] = sum(1 for row in rows if row[name] is not None)
        values = {row[name] for row in rows}
        if len(values) == 1 and (value := next(iter(values))) is not None:
            node[name] = value
    if not keys:
        return node

    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[keys[0]], []).append(row)

    # each group states its own value for the splitting level as a shared level
    node["counts"] = [
        _summary(grouped[value], keys)
        for value in sorted(grouped, key=lambda v: (v is not None, v))
    ]
    return node


@dataclass(frozen=True)
class ManifestLoader(Loader):
    """Writes one manifest per run into the project folder's output/."""

    project: str
    timestamp: datetime
    dry: bool

    def load(self, roots: list[Resource]) -> None:
        """Write the manifest for one run's trees."""
        manifest = {
            "header": {
                "project": self.project,
                "dry": self.dry,
                "timestamp": self.timestamp.isoformat(),
                "index": index_name(self.project, self.timestamp),
                "alias": alias_name(self.project),
                "pipeline_version": _pipeline_version(),
            },
            "summary": _summary(
                [
                    # The metadata fields come from their one declaration: a new one reaches the rows by itself
                    {"source": chunk.lineage()[0].title, "url": chunk.url, **chunk.metadata}
                    for chunk in iter_chunks(roots)
                ],
                SUMMARY_LEVELS,
            ),
            "tree": [_node(root, self.project_dir) for root in roots],
        }

        output = self.project_dir / "output"
        output.mkdir(parents=True, exist_ok=True)
        name = f"{self.timestamp:%Y%m%dT%H%M%S}_manifest"
        if self.dry:
            name += "_dry"
        path = output / f"{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Manifest written: {path}")
