"""Content-hash disk cache: the one cache every transform shares.

The scope is the project and the transform's name; the key is the sha256
of the content the transform would recompute from, so identical content
within one project shares an entry. Values are files or whole
directories (nested), copied in and out as they are. A per-run
`--refresh TRANSFORM` flag bypasses lookups for the named transform;
there is no eviction — the cache grows until cleaned.

A transform whose output depends on a recipe (prompts, model, rendering
settings) works in a scope named after the recipe's version: bumping the
version starts a fresh scope, so nothing made under the old recipe is ever
served again.

Subtrees are cached as Merkle trees: the leaves are ordinary entries, keyed
by their own content, and a parent's entry is a small node record listing
its ordered (name, leaf hash) children. A node hit adopts the whole subtree
while the leaves stay the single copy of the content.
"""

from __future__ import annotations

import copy
import json
import os
import shutil

from collections.abc import Sequence
from pathlib import Path

# The file a node record is stored under, next to nothing else in its entry
NODE_FILE = "children.json"


class Cache:
    """Disk cache for one project's transform, keyed by content hash only."""

    def __init__(self, project: str, transform: str, refresh: bool = False):
        self.project = project
        self.transform = transform
        self.refresh = refresh

        cache_path = Path(os.environ["CACHE_DIR"])
        if not cache_path.exists():
            raise ValueError(f"Cache path {cache_path} does not exist.")
        self.cache_path = cache_path / project / transform

    def scope(self, name: str) -> Cache:
        """The cache nested under this one in its own folder, e.g. one per recipe version."""
        scoped = copy.copy(self)
        scoped.cache_path = self.cache_path / name
        return scoped

    def _entry_path(self, content_hash: str, value: Path) -> Path:
        return self.cache_path / content_hash / value.name

    def get(self, content_hash: str, value: Path) -> bool:
        """Copy the cached output for `content_hash` to `value`; False on miss or refresh."""
        value = Path(value)

        if self.refresh:
            return False

        entry = self._entry_path(content_hash, value)
        if not entry.is_file() and not entry.is_dir():
            return False

        if entry.is_dir():
            value.mkdir(parents=True, exist_ok=True)
            shutil.copytree(entry, value, dirs_exist_ok=True)
        else:
            value.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, value)

        return True

    def set(self, content_hash: str, value: Path) -> None:
        """Store the output produced for `content_hash` in the cache.

        A path that was never produced (e.g. no frames extracted) is not
        cached: the next run recomputes it.
        """
        value = Path(value)
        if not value.exists():
            return

        entry = self._entry_path(content_hash, value)

        if value.is_dir():
            entry.mkdir(parents=True, exist_ok=True)
            shutil.copytree(value, entry, dirs_exist_ok=True)
        else:
            entry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(value, entry)

    def read(self, content_hash: str, name: str) -> str | None:
        """Return the cached text value for `content_hash`, or None on miss or refresh.

        A text value is read in memory rather than copied to a path, for
        outputs that live nowhere on disk (e.g. the vector of one chunk).
        """
        if self.refresh:
            return None

        entry = self.cache_path / content_hash / name
        if not entry.is_file():
            return None

        return entry.read_text(encoding="utf-8")

    def write(self, content_hash: str, name: str, text: str) -> None:
        """Store a text value for `content_hash` in the cache."""
        entry = self.cache_path / content_hash / name
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(text, encoding="utf-8")

    def adopt(self, key: str, work_dir: Path) -> list[Path] | None:
        """Copy the subtree recorded under `key` into work_dir, its leaves in order.

        None — so the caller recomputes — when there is no node record, it
        is malformed, or any leaf it names is missing (refresh included).
        Leaves already copied before a gap are harmless: the recompute
        reads them back from the same entries.
        """
        text = self.read(key, NODE_FILE)
        if text is None:
            return None

        try:
            children = [(str(name), str(leaf_hash)) for name, leaf_hash in json.loads(text)]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

        paths: list[Path] = []
        for name, leaf_hash in children:
            path = work_dir / name
            if not self.get(leaf_hash, path):
                return None
            paths.append(path)

        return paths

    def write_node(self, key: str, children: Sequence[tuple[str, str]]) -> None:
        """Record the subtree under `key`: its ordered (name, leaf hash) children."""
        self.write(key, NODE_FILE, json.dumps([[name, leaf_hash] for name, leaf_hash in children]))
