"""The manifest loader: one JSON manifest per run, written for humans.

The manifest records what a run produced without carrying content: the
header states run identity, the summary drills down per source, type,
subtype, number, subnumber and exercise parts — per group counting how
many chunks carry a url, week or release dates, and stating those values
when the whole group shares one — and the tree mirrors the resource
trees with metadata and content-hash pointers only — text and vectors
are recovered from the cache by hash. Nothing reads a manifest; it is
inspected.
"""

from __future__ import annotations

import json
import logging

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rag_etl.core import Resource, iter_chunks
from rag_etl.loaders.elasticsearch import alias_name, index_name

logger = logging.getLogger(__name__)

# Levels of the summary drill-down, outermost first
SUMMARY_LEVELS = ("source", "type", "subtype", "number", "subnumber", "parts")

# Coverage fields a summary group reports: a with_X count of the chunks
# carrying one, plus X itself when every chunk in the group shares one value
COVERAGE_FIELDS = ("url", "week", "from", "until")


def _pipeline_version() -> str:
    try:
        return version("rag-etl")
    except PackageNotFoundError:
        return "unknown"


def _node(node: Resource) -> dict[str, Any]:
    """One tree node: metadata and a content-hash pointer, no text, no vector."""
    return {
        "title": node.title,
        "url": node.url,
        "path": str(node.path) if node.path is not None else None,
        "mime_type": node.mime_type,
        "cut": node.cut,
        **node.metadata,
        "chunk": node.is_chunk,
        "content_hash": node.content_hash,
        "children": [_node(child) for child in node.children],
    }


def _summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """One summary node: coverage counts, unanimous values, drill-down.

    Each coverage field reports how many chunks in the group carry a
    non-null value, and the value itself only when they all share one.
    The drill-down groups rows by the next key — None sorts last, so
    manifests diff cleanly — and recurses; the last level, where no keys
    remain, carries no counts key.
    """
    node: dict[str, Any] = {"chunks": len(rows)}
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

    node["counts"] = [
        {keys[0]: value, **_summary(grouped[value], keys[1:])}
        for value in sorted(grouped, key=lambda v: (v is not None, v))
    ]
    return node


class ManifestLoader:
    """Writes one manifest per run into the directory it is given."""

    def __init__(self, path: Path, project: str, timestamp: datetime, dry: bool):
        self.path = Path(path)
        self.project = project
        self.timestamp = timestamp
        self.dry = dry

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
                    {
                        "source": chunk.lineage()[0].title,
                        "type": chunk.type,
                        "subtype": chunk.subtype,
                        "number": chunk.number,
                        "subnumber": chunk.subnumber,
                        "parts": chunk.exercise_parts,
                        "url": chunk.url,
                        "week": chunk.week,
                        "from": chunk.from_,
                        "until": chunk.until,
                    }
                    for chunk in iter_chunks(roots)
                ],
                SUMMARY_LEVELS,
            ),
            "tree": [_node(root) for root in roots],
        }

        self.path.mkdir(parents=True, exist_ok=True)
        name = f"{self.timestamp:%Y%m%dT%H%M%S}_manifest"
        if self.dry:
            name += "_dry"
        path = self.path / f"{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Manifest written: {path}")
