"""The manifest loader: one JSON manifest per run, written for humans.

The manifest records what a run produced without carrying content: the
header states run identity, the summary drills down per source, type,
subtype and number, and the tree mirrors the resource trees with
metadata and content-hash pointers only — text and vectors are
recovered from the cache by hash. Nothing reads a manifest; it is
inspected.
"""

from __future__ import annotations

import json
import logging

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rag_etl.core import Resource
from rag_etl.loaders.elasticsearch import alias_name, index_name

logger = logging.getLogger(__name__)

# Levels of the summary drill-down, outermost first
SUMMARY_LEVELS = ("source", "type", "subtype", "number")


def _pipeline_version() -> str:
    try:
        return version("rag-etl")
    except PackageNotFoundError:
        return "unknown"


def _chunks(roots: list[Resource]) -> list[tuple[Resource, Resource]]:
    """Every chunk, paired with the root of its tree."""
    return [(root, node) for root in roots for node in root.walk() if node.text is not None]


def _without_url(node: Resource) -> bool:
    """Whether no node on the chain down to this one carries a url."""
    return all(ancestor.url is None for ancestor in node.lineage())


def _node(node: Resource) -> dict[str, Any]:
    """One tree node: metadata and a content-hash pointer, no text, no vector."""
    return {
        "title": node.title,
        "url": node.url,
        "path": str(node.path) if node.path is not None else None,
        "mime_type": node.mime_type,
        "type": node.type,
        "subtype": node.subtype,
        "cut": node.cut,
        "number": node.number,
        "subnumber": node.subnumber,
        "week": node.week,
        "is_solution": node.is_solution,
        "from": node.from_,
        "until": node.until,
        "chunk": node.text is not None,
        "content_hash": node.content_hash,
        "children": [_node(child) for child in node.children],
    }


def _level(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """One nesting level: group rows by the next key, aggregate, recurse.

    Rows are chunk facts; each node aggregates its group and nests the
    remaining keys. None sorts last, so manifests diff cleanly.
    """
    if not keys:
        return []

    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[keys[0]], []).append(row)

    nodes: list[dict[str, Any]] = []
    for value in sorted(grouped, key=lambda v: (v is not None, v)):
        members = grouped[value]
        nodes.append(
            {
                keys[0]: value,
                "chunks": len(members),
                "without_url": sum(member["without_url"] for member in members),
                "is_solution": sum(member["is_solution"] for member in members),
                "counts": _level(members, keys[1:]),
            }
        )

    return nodes


class ManifestLoader:
    """Writes one manifest per run into the directory it is given."""

    def __init__(self, path: Path, project: str, timestamp: datetime, dry: bool):
        self.path = Path(path)
        self.project = project
        self.timestamp = timestamp
        self.dry = dry

    def load(self, roots: list[Resource]) -> None:
        """Write the manifest for one run's trees."""
        chunks = _chunks(roots)
        manifest = {
            "header": {
                "project": self.project,
                "dry": self.dry,
                "timestamp": self.timestamp.isoformat(),
                "index": index_name(self.project, self.timestamp),
                "alias": alias_name(self.project),
                "pipeline_version": _pipeline_version(),
            },
            "summary": {
                "sources": len(roots),
                "chunks": len(chunks),
                "without_url": sum(1 for _, chunk in chunks if _without_url(chunk)),
                "counts": _level(
                    [
                        {
                            "source": root.title or root.url or "unknown",
                            "type": chunk.type or "unknown",
                            "subtype": chunk.subtype or "unknown",
                            "number": chunk.number,
                            "without_url": _without_url(chunk),
                            "is_solution": chunk.is_solution,
                        }
                        for root, chunk in chunks
                    ],
                    SUMMARY_LEVELS,
                ),
            },
            "tree": [_node(root) for root in roots],
        }

        self.path.mkdir(parents=True, exist_ok=True)
        name = f"{self.timestamp:%Y%m%dT%H%M%S}_manifest"
        if self.dry:
            name += "_dry"
        path = self.path / f"{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Manifest written: {path}")
